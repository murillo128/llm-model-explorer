"""One app-owned native instance; an exclusive lease covers each complete experiment."""

import asyncio
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch.utils.hooks import RemovableHandle
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from ..model_files import ModelError
from ..operations import Cancellation, DeviceScheduler, OperationCancelled, _complete, _drain
from ..tensor_analysis import CPU_ALLOCATION_FAILURE
from .admission import AdmittedModel, NeuronInput, Unavailable, runtime_identity

ActivationHook = Callable[[torch.Tensor], torch.Tensor | None]


def numerical_failure() -> ModelError:
    return ModelError(
        "internal_error",
        "Neuron computation produced non-finite values.",
        500,
        details={"reason": "numerical_failure"},
    )


def execution_failure(exc: Exception) -> ModelError:
    if isinstance(exc, MemoryError | torch.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and CPU_ALLOCATION_FAILURE.search(str(exc))
    ):
        return ModelError("resource_exhausted", "Insufficient execution memory.", 503)
    # Framework/hook exceptions may contain tensors, tokens or private paths.
    return ModelError("internal_error", "Neuron execution failed.", 500)


def load_model(admitted: AdmittedModel, cancellation: Cancellation) -> Any:
    """Use only the packaged class and guarded numeric reader, never HF path loading.

    Meta construction avoids random initialization and an extra resident model.
    The sole non-weight buffer (default RoPE) is rebuilt on the actual device.
    """
    cancellation.check()
    admitted.source.check_unchanged()
    device = admitted.execution["device"]
    config = LlamaConfig(**admitted.config)  # type: ignore[no-untyped-call]
    with torch.device("meta"):
        model: Any = LlamaForCausalLM(config)  # type: ignore[no-untyped-call]
        model.to(dtype=torch.float32)
    model.to_empty(device=device)
    model.tie_weights()
    model.model.rotary_emb = LlamaRotaryEmbedding(config, device=torch.device(device))
    parameters = dict(model.named_parameters())
    if set(parameters) != set(admitted.tensors):
        raise ModelError("unsupported_representation", "Runtime weight bindings are incompatible.")
    for name, parameter in parameters.items():
        cancellation.check()
        descriptor = admitted.tensors[name]
        if tuple(parameter.shape) != descriptor.shape:
            raise ModelError("unsupported_representation", "Runtime weight shape is incompatible.")
        offset = 0
        for block in admitted.source.iter_tensor(descriptor.id):
            cancellation.check()
            if not torch.isfinite(block).all():
                raise numerical_failure()
            parameter.view(-1)[offset : offset + block.numel()].copy_(block)
            offset += block.numel()
        if offset != parameter.numel():
            raise ModelError("unsupported_representation", "Incomplete native weight.")
    model.eval()
    model.requires_grad_(False)
    admitted.source.check_unchanged()
    cancellation.check()
    return model


class NeuronLease:
    """Blocking, operation-local forward/hook primitives for capture and comparison.

    Hooks see [batch=1, absolute-prefix-position, intermediate-coordinate]. Every
    forward recomputes the exact prefix without a KV cache. No model or hook may
    escape this lease; output tensors belong only to the calling operation.
    """

    def __init__(self, admitted: AdmittedModel, model: Any, cancellation: Cancellation) -> None:
        self.admitted = admitted
        self._model = model
        self.cancellation = cancellation
        self._hooks: list[RemovableHandle] = []

    def _check(self) -> None:
        self.cancellation.check()
        if self._model is None:
            raise RuntimeError("execution lease is closed")
        self.admitted.source.check_unchanged()

    @contextmanager
    def hook(self, module_id: str, callback: ActivationHook) -> Iterator[None]:
        """Attach only at the reviewed MLP post-product/pre-down point."""
        self._check()
        module = next((m for m in self.admitted.modules if m["module_id"] == module_id), None)
        if module is None:
            raise ModelError("validation_error", "Unknown neuron module.")
        down = self._model.model.layers[module["layer_index"]].mlp.down_proj

        def invoke(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> Any:
            self.cancellation.check()
            values = args[0]
            if not torch.isfinite(values).all():
                raise numerical_failure()
            replacement = callback(values)
            if replacement is None:
                replacement = values
            if (
                replacement.shape != values.shape
                or replacement.dtype != values.dtype
                or replacement.device != values.device
            ):
                raise ModelError("internal_error", "Invalid neuron hook result.", 500)
            if not torch.isfinite(replacement).all():
                raise numerical_failure()
            self.cancellation.check()
            return (replacement,)

        handle = down.register_forward_pre_hook(invoke)
        self._hooks.append(handle)
        try:
            yield
        finally:
            handle.remove()
            self._hooks.remove(handle)

    def forward(self, token_ids: tuple[int, ...]) -> torch.Tensor:
        self._check()
        self.admitted.validate_tokens(token_ids, continuation=True)
        inputs = torch.tensor(
            [token_ids], dtype=torch.long, device=self.admitted.execution["device"]
        )
        # No generation defaults, BOS insertion, sampling, dropout or persistent KV state.
        output = self._model(input_ids=inputs, use_cache=False, return_dict=True)
        logits: torch.Tensor = output.logits
        if not torch.isfinite(logits).all():
            raise numerical_failure()
        self._check()
        return logits.detach().to(device="cpu", dtype=torch.float32, copy=True)

    def close(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._model = None


class NeuronRuntime:
    def __init__(self, scheduler: DeviceScheduler, device: str) -> None:
        self.scheduler = scheduler
        self.device = device
        self._lock = asyncio.Lock()
        self._model: Any = None
        self._key: tuple[str, str, str, str] | None = None
        self._tasks: dict[asyncio.Task[Any], Cancellation] = {}
        self._closed = False

    async def run[T](
        self,
        request: NeuronInput,
        cancellation: Cancellation,
        operation: Callable[[NeuronLease], T],
    ) -> T:
        """Run from Consumer.prepare so session/operation deletion owns cancellation.

        Callers finish request preflight before this method, then guard all result
        delivery with the admitted source. This method also drains direct task
        cancellation, including repeated disconnect signals.
        """
        if self._closed:
            raise ModelError("resource_exhausted", "Neuron runtime is shutting down.", 503)
        request.validate()
        admitted = request.model
        try:
            identity = runtime_identity(self.device)
        except Unavailable:
            raise ModelError(
                "unsupported_representation", "Execution environment unavailable."
            ) from None
        if admitted.execution != identity:
            raise ModelError("unsupported_representation", "Execution environment changed.")
        cancellation.check()
        task = asyncio.create_task(self._run(admitted, cancellation, operation))
        self._tasks[task] = cancellation
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            cancellation.cancel()
            await _drain(task)
            raise
        finally:
            self._tasks.pop(task, None)

    async def _run[T](
        self,
        admitted: AdmittedModel,
        cancellation: Cancellation,
        operation: Callable[[NeuronLease], T],
    ) -> T:
        acquire = asyncio.create_task(self._lock.acquire())
        cancelled = asyncio.create_task(cancellation._wake.wait())
        acquired = False
        try:
            await asyncio.wait([acquire, cancelled], return_when=asyncio.FIRST_COMPLETED)
            if acquire.done():
                acquired = acquire.result()
            cancellation.check()
            return await self.scheduler.run(
                self.device, cancellation, self._execute, admitted, cancellation, operation
            )
        finally:
            if not acquire.done():
                acquire.cancel()
            cancelled.cancel()
            await asyncio.gather(acquire, cancelled, return_exceptions=True)
            if acquired:
                self._lock.release()

    def _execute[T](
        self,
        admitted: AdmittedModel,
        cancellation: Cancellation,
        operation: Callable[[NeuronLease], T],
    ) -> T:
        lease = None
        failure: Exception | None = None
        try:
            with torch.inference_mode(), torch.autocast(self.device.split(":")[0], enabled=False):
                admitted.source.check_unchanged()
                cancellation.check()
                if self._key != admitted.key:
                    # Drop the preceding instance before allocating its replacement.
                    self._model, self._key = None, None
                    self._model = load_model(admitted, cancellation)
                    self._key = admitted.key
                admitted.source.check_unchanged()
                cancellation.check()
                lease = NeuronLease(admitted, self._model, cancellation)
                result = operation(lease)
                admitted.source.check_unchanged()
                cancellation.check()
        except (ModelError, OperationCancelled) as exc:
            self._model, self._key = None, None
            failure = (
                ModelError(exc.code, str(exc), exc.status, details=exc.details)
                if isinstance(exc, ModelError)
                else OperationCancelled()
            )
            traceback.clear_frames(exc.__traceback__)
        except Exception as exc:
            self._model, self._key = None, None
            failure = execution_failure(exc)
            traceback.clear_frames(exc.__traceback__)
        finally:
            try:
                if self.device != "cpu":
                    torch.cuda.synchronize(self.device)
                admitted.source.check_unchanged()
            except ModelError as exc:
                self._model, self._key = None, None
                failure = ModelError(exc.code, str(exc), exc.status, details=exc.details)
                traceback.clear_frames(exc.__traceback__)
            except Exception as exc:
                self._model, self._key = None, None
                failure = execution_failure(exc)
                traceback.clear_frames(exc.__traceback__)
            finally:
                if lease is not None:
                    lease.close()
        # Raise outside the framework exception context: even a retained public
        # error must not retain a failed model, prompt or activation via traceback.
        if failure is not None:
            raise failure
        return result

    async def aclose(self) -> None:
        self._closed = True
        for cancellation in self._tasks.values():
            cancellation.cancel()

        async def settle() -> None:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)
            self._model, self._key = None, None

        await _complete(asyncio.create_task(settle()))
