"""Application-owned blocking preparation and immutable prepared-result lookup."""

import asyncio
import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass
from time import perf_counter

from .architecture_analysis import AnalysisInput, DescriptionRegistry, register_dense_descriptions
from .architecture_analysis.core import AnalysisResult
from .architecture_analysis.deepseek_v2 import register_deepseek_v2
from .architecture_analysis.glm4_moe_lite import register_glm4_moe_lite
from .architecture_analysis.kimi_linear import register_kimi_linear
from .architecture_analysis.qwen35 import register_qwen35
from .architecture_analysis.validation import MAX_BYTES, BindingContext, GraphError
from .architecture_analysis.vjepa2 import register_vjepa2
from .artifacts import ArchitectureArtifactSpec, ArtifactStore
from .execution import BlockingWork
from .models import CatalogueEntry, ModelCatalogue
from .tensor_source import ModelSource

logger = logging.getLogger(__name__)


def unavailable(model_id: str, reason: str, message: str) -> dict[str, object]:
    return {
        "status": "unavailable",
        "model_id": model_id,
        "reason": reason,
        "requires_restart": True,
        "diagnostics": [{"code": reason, "message": message}],
    }


@dataclass(frozen=True)
class Prepared:
    spec: ArchitectureArtifactSpec | None
    context: BindingContext | None
    failure: dict[str, object] | None = None


class PreparationStopped(Exception):
    """Internal safe-boundary cancellation; never a terminal model outcome."""


class ArchitectureService:
    def __init__(
        self, store: ArtifactStore, work: BlockingWork, *, stop: threading.Event | None = None
    ) -> None:
        self.store = store
        self.work = work
        self.registry = DescriptionRegistry()
        register_dense_descriptions(self.registry)
        register_deepseek_v2(self.registry)
        register_glm4_moe_lite(self.registry)
        register_kimi_linear(self.registry)
        register_qwen35(self.registry)
        register_vjepa2(self.registry)
        self._prepared: dict[tuple[str, str], Prepared] = {}
        self._stop = stop if stop is not None else threading.Event()

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

    def _prepare_one(self, entry: CatalogueEntry) -> None:
        model_id = entry.summary.id
        source = None
        outcome = "unavailable"
        mode = "analysis"
        hashing = analysis = cache_read = 0.0
        started = perf_counter()
        try:
            stamp = perf_counter()
            try:
                source = entry.pin()
            finally:
                hashing += perf_counter() - stamp
            self._check_stop()
            inputs = AnalysisInput.from_source(
                source, tokenizer_available=entry.summary.tokenizer_available
            )
            selected = self.registry.select(inputs)
            key = (model_id, source.fingerprint)
            if selected is None:
                self._prepared[key] = Prepared(
                    None,
                    None,
                    unavailable(
                        model_id,
                        "unsupported_architecture",
                        "No verified description matches. Install a compatible analyzer "
                        "before restarting.",
                    ),
                )
                return
            spec = ArchitectureArtifactSpec(
                model_fingerprint=source.fingerprint,
                producer=selected.producer,
                scope=selected.scope,
                analysis_options={},
            )
            # Reserve the exact response envelope, including JSON-escaped model ID.
            budget = MAX_BYTES - len(self._prefix(model_id)) - 1
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
                    result = self.registry.analyze(inputs, byte_limit=budget)
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
            if isinstance(exc, OSError):
                self._check_storage()  # A global storage failure aborts startup.
            if source is not None:
                reason = (
                    "unsupported_size"
                    if isinstance(exc, GraphError) and exc.code == "unsupported_size"
                    else "analysis_failed"
                )
                self._prepared[(model_id, source.fingerprint)] = Prepared(
                    None, None, unavailable(model_id, reason, "Architecture preparation failed.")
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
    def _prefix(model_id: str) -> bytes:
        return (
            b'{"status":"available","model_id":'
            + json.dumps(model_id).encode()
            + b',"diagnostics":[],"graph":'
        )

    def lookup(self, source: ModelSource) -> bytes:
        """Only read startup state/cache, with the session's source guarded on both sides."""
        source.check_unchanged()
        prepared = self._prepared.get((source.model_id, source.fingerprint))
        if prepared is None:
            result = json.dumps(
                unavailable(
                    source.model_id, "restart_required", "Restart to prepare this model content."
                )
            ).encode()
        elif prepared.failure is not None:
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
                            )
                        ).encode()
                    else:
                        result = prefix + reader.read_available(reader.available_bytes) + b"}"
        source.check_unchanged()
        return result
