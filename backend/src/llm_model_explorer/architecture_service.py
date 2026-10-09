"""Application-owned startup/demand preparation and immutable result lookup."""

import asyncio
import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass
from time import perf_counter

from .architecture_analysis import AnalysisInput, DescriptionRegistry, register_dense_descriptions
from .architecture_analysis.clm import register_clm
from .architecture_analysis.core import AnalysisResult, Scope
from .architecture_analysis.deepseek_v2 import register_deepseek_v2
from .architecture_analysis.glm4_moe_lite import register_glm4_moe_lite
from .architecture_analysis.kev import register_kev
from .architecture_analysis.kimi_linear import register_kimi_linear
from .architecture_analysis.model_defined import (
    ModelDefinedValidator,
    diagnostic_for_model_error,
    producer_for,
)
from .architecture_analysis.model_defined_schema import MAX_DEFINITION_BYTES
from .architecture_analysis.qwen35 import register_qwen35
from .architecture_analysis.validation import MAX_BYTES, BindingContext, GraphError, model_finding
from .architecture_analysis.vjepa2 import register_vjepa2
from .artifacts import ArchitectureArtifactSpec, ArtifactStore
from .execution import BlockingWork
from .model_files import ModelError
from .models import CatalogueEntry, ModelCatalogue
from .tensor_source import ModelSource

logger = logging.getLogger(__name__)


def unavailable(
    model_id: str, reason: str, message: str, *, requires_restart: bool = False
) -> dict[str, object]:
    return {
        "status": "unavailable",
        "model_id": model_id,
        "reason": reason,
        "requires_restart": requires_restart,
        "diagnostics": [{"code": reason, "message": message}],
    }


@dataclass(frozen=True)
class Prepared:
    spec: ArchitectureArtifactSpec | None
    context: BindingContext | None
    failure: dict[str, object] | None = None


class PreparationStopped(Exception):
    """Internal safe-boundary cancellation; never a terminal model outcome."""


def packaged_registry() -> DescriptionRegistry:
    registry = DescriptionRegistry()
    register_dense_descriptions(registry)
    register_clm(registry)
    register_deepseek_v2(registry)
    register_glm4_moe_lite(registry)
    register_kimi_linear(registry)
    register_kev(registry)
    register_qwen35(registry)
    register_vjepa2(registry)
    return registry


class ArchitectureService:
    def __init__(
        self, store: ArtifactStore, work: BlockingWork, *, stop: threading.Event | None = None
    ) -> None:
        self.store = store
        self.work = work
        self.registry = packaged_registry()
        self._prepared: dict[tuple[str, str], Prepared] = {}
        self._stop = stop if stop is not None else threading.Event()
        # Event-loop owned tasks outlive individual HTTP waiters. Only one worker
        # may analyze at a time, without occupying pool threads while queued.
        self._pending: dict[tuple[str, str], asyncio.Task[None]] = {}
        self._analysis_lock = asyncio.Lock()

    async def get(self, source: ModelSource) -> bytes:
        """Ensure this exact session snapshot, then perform guarded retrieval."""
        await self.work.run(source.check_unchanged)
        self._check_stop()
        key = (source.model_id, source.fingerprint)
        if key not in self._prepared:
            task = self._pending.get(key)
            if task is None:
                task = asyncio.create_task(self._ensure(source))
                self._pending[key] = task
                task.add_done_callback(lambda done: self._finished(key, done))
            # Disconnect/cancellation never cancels the shared blocking writer.
            await asyncio.shield(task)
        return await self.work.run(self.lookup, source)

    async def _ensure(self, source: ModelSource) -> None:
        async with self._analysis_lock:
            self._check_stop()
            await self.work.run(self._prepare_one, source)

    def _finished(self, key: tuple[str, str], task: asyncio.Task[None]) -> None:
        del self._pending[key]
        # Retrieve errors even if all waiters disconnected. Active waiters still
        # receive the original exception through shield.
        if not task.cancelled():
            task.exception()

    async def aclose(self) -> None:
        """Stop at publication boundaries and settle writers before pool teardown."""
        self._stop.set()
        pending = asyncio.gather(*self._pending.values(), return_exceptions=True)
        cancelled = False
        while not pending.done():
            try:
                await asyncio.shield(pending)
            except asyncio.CancelledError:
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError()

    def _check_stop(self) -> None:
        if self._stop.is_set():
            raise PreparationStopped()

    def _check_storage(self) -> None:
        # Probe global usability independently of any one damaged artifact entry.
        # Own and remove only this probe, never other producers' temporary files.
        with tempfile.TemporaryDirectory(
            prefix=".tmp-architecture-probe-", dir=self.store.root
        ) as d:
            with open(os.path.join(d, "probe"), "w+b") as stream:
                stream.write(b"cache")
                stream.flush()
                os.fsync(stream.fileno())
                stream.seek(0)
                if stream.read() != b"cache":
                    raise OSError("Artifact cache probe failed")

    async def prepare(self, catalogue: ModelCatalogue) -> None:
        # Shield the worker so cancellation cannot abandon a live writer. The stop
        # flag reaches publication checks before resources are released.
        task = asyncio.create_task(self.work.run(self._prepare_all, catalogue))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            self._stop.set()
            while not task.done():
                try:
                    await asyncio.shield(task)
                except (asyncio.CancelledError, Exception):
                    pass
            # Retrieve any worker failure while preserving the cancellation outcome.
            if not task.cancelled():
                task.exception()
            raise

    def _prepare_all(self, catalogue: ModelCatalogue) -> None:
        self._check_storage()
        for entry in catalogue.discover():
            self._check_stop()
            self._prepare_one(entry)
        self._check_stop()

    def _prepare_one(self, entry: CatalogueEntry | ModelSource) -> None:
        model_id = entry.model_id if isinstance(entry, ModelSource) else entry.summary.id
        source = None
        model_supplied = False
        outcome = "unavailable"
        mode = "analysis"
        hashing = analysis = cache_read = 0.0
        started = perf_counter()
        try:
            stamp = perf_counter()
            try:
                source = entry if isinstance(entry, ModelSource) else entry.pin()
                source.check_unchanged()
            finally:
                hashing += perf_counter() - stamp
            self._check_stop()
            inputs = AnalysisInput.from_source(
                source, tokenizer_available=source.tokenizer_available
            )
            try:
                raw_definition = source.architecture_definition(max_bytes=MAX_DEFINITION_BYTES)
            except ModelError as exc:
                if exc.code == "unsupported_size":
                    model_supplied = True
                    raise model_finding(
                        "unsupported_size", "resource", "", "Definition exceeds the 8 MiB limit."
                    ) from exc
                raise
            model_supplied = raw_definition is not None
            validator = (
                ModelDefinedValidator.from_bytes(raw_definition)
                if raw_definition is not None
                else None
            )
            definition = validator.definition if validator is not None else None
            selected = None if model_supplied else self.registry.select(inputs)
            key = (model_id, source.fingerprint)
            if selected is None and definition is None:
                self._prepared[key] = Prepared(
                    None,
                    None,
                    unavailable(
                        model_id,
                        "unsupported_architecture",
                        "No verified description matches. Install a compatible analyzer "
                        "or supply a valid architecture.json definition.",
                    ),
                )
                return
            scope: Scope
            if definition is not None:
                producer = producer_for(definition)
                scope = "model_defined"
            else:
                assert selected is not None
                producer, scope = selected.producer, selected.scope
            spec = ArchitectureArtifactSpec(
                model_fingerprint=source.fingerprint,
                producer=producer,
                scope=scope,
                analysis_options={},
            )
            # Reserve the exact response envelope, including JSON-escaped model ID.
            budget = self.response_budget(model_id)
            stamp = perf_counter()
            try:
                reader = self.store.lookup_graph(spec, inputs.bindings)
            finally:
                cache_read += perf_counter() - stamp
            if reader is not None:
                with reader:
                    if reader.available_bytes > budget:
                        raise GraphError("unsupported_size", "Architecture response exceeds limit.")
                source.check_unchanged()
                self._check_stop()
                mode, outcome = "cache", "available"
            else:
                stamp = perf_counter()
                try:
                    result = (
                        validator.validate(inputs, byte_limit=budget)
                        if validator is not None
                        else self.registry.analyze(inputs, byte_limit=budget)
                    )
                finally:
                    analysis += perf_counter() - stamp
                self._check_stop()
                if result.graph is None:
                    self._prepared[key] = Prepared(None, None, self._failure(model_id, result))
                    return

                def check_publication() -> None:
                    nonlocal hashing
                    self._check_stop()
                    stamp = perf_counter()
                    try:
                        source.check_unchanged(rehash=True)
                    finally:
                        hashing += perf_counter() - stamp
                    self._check_stop()

                with self.store.begin_graph_write(
                    spec, inputs.bindings, check_source=check_publication
                ) as writer:
                    writer.write_graph(result.graph)
                    self._check_stop()
                    writer.commit()
                outcome = result.status
            self._prepared[key] = Prepared(spec, inputs.bindings)
        except PreparationStopped:
            outcome = "cancelled"
            raise
        except Exception as exc:
            if isinstance(entry, ModelSource) and isinstance(exc, (OSError, MemoryError)):
                # Request-time infrastructure failures retain the ordinary HTTP
                # error path; they are not terminal model-analysis diagnostics.
                raise
            if isinstance(exc, OSError):
                self._check_storage()  # A global storage failure aborts startup.
            if source is not None:
                reason = (
                    "unsupported_size"
                    if isinstance(exc, (GraphError, ModelError)) and exc.code == "unsupported_size"
                    else "analysis_failed"
                )
                failure = unavailable(model_id, reason, "Architecture preparation failed.")
                if model_supplied and isinstance(exc, GraphError):
                    failure["diagnostics"] = [diagnostic_for_model_error(exc).document()]
                self._prepared[(model_id, source.fingerprint)] = Prepared(
                    None,
                    None,
                    failure,
                )
        finally:
            logger.info(
                "Architecture model=%s mode=%s outcome=%s hashing_seconds=%.6f "
                "analysis_seconds=%.6f cache_read_seconds=%.6f preparation_seconds=%.6f",
                model_id,
                mode,
                outcome,
                hashing,
                analysis,
                cache_read,
                perf_counter() - started,
            )

    @staticmethod
    def _failure(model_id: str, result: AnalysisResult) -> dict[str, object]:
        response = unavailable(model_id, result.reason or "analysis_failed", "Analysis failed.")
        response["diagnostics"] = [d.document() for d in result.diagnostics]
        return response

    @staticmethod
    def response_budget(model_id: str) -> int:
        """Leave room for the architecture response envelope around a graph."""
        return MAX_BYTES - len(ArchitectureService._prefix(model_id)) - 1

    @staticmethod
    def _prefix(model_id: str) -> bytes:
        return (
            b'{"status":"available","model_id":'
            + json.dumps(model_id).encode()
            + b',"diagnostics":[],"graph":'
        )

    def lookup(self, source: ModelSource) -> bytes:
        """Read prepared state/cache, with the session source guarded on both sides."""
        source.check_unchanged()
        prepared = self._prepared[(source.model_id, source.fingerprint)]
        if prepared.failure is not None:
            result = json.dumps(prepared.failure).encode()
        else:
            assert prepared.spec is not None and prepared.context is not None
            reader = self.store.lookup_graph(prepared.spec, prepared.context)
            if reader is None:
                result = json.dumps(
                    unavailable(
                        source.model_id,
                        "cache_unavailable",
                        "Prepared cache is unavailable; restart required.",
                        requires_restart=True,
                    )
                ).encode()
            else:
                with reader:
                    prefix = self._prefix(source.model_id)
                    if reader.available_bytes > MAX_BYTES - len(prefix) - 1:
                        result = json.dumps(
                            unavailable(
                                source.model_id,
                                "cache_unavailable",
                                "Prepared response exceeds its limit.",
                                requires_restart=True,
                            )
                        ).encode()
                    else:
                        result = prefix + reader.read_available(reader.available_bytes) + b"}"
        source.check_unchanged()
        return result
