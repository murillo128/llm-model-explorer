"""Faithful local Hugging Face tokenization, owned by one application lifespan."""

import json
import shutil
from _thread import LockType
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Any

from transformers import AutoTokenizer

from .model_files import ModelError, open_local, read_json
from .tensor_source import MAX_SAFE_INTEGER, ModelSource


def unavailable() -> ModelError:
    return ModelError("unsupported_representation", "Local tokenizer is missing or unsupported.")


def load_tokenizer(root: Path, directory: Path) -> Any:
    """Load only conventional, guarded assets; never hand weights or code to HF.

    A private temporary copy also excludes untracked chat templates and prevents
    the library from following source symlinks after snapshot validation.
    """
    with TemporaryDirectory(prefix="model-explorer-tokenizer-") as temporary:
        target = Path(temporary)
        for path in directory.iterdir():
            if path.suffix not in {".json", ".model", ".txt", ".tiktoken"}:
                continue
            if path.name.endswith(".safetensors.index.json"):
                continue
            with open_local(root, path) as source, (target / path.name).open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
        config_path = target / "tokenizer_config.json"
        if config_path.exists():
            try:
                config = read_json(target, config_path)
            except ModelError as exc:
                raise unavailable() from exc
            # HF can honor file overrides in saved init kwargs. Only staged,
            # immediate conventional assets may be referenced by that metadata.
            for key, value in config.items():
                if value is None:
                    continue
                if key == "fast_tokenizer_files":
                    if not isinstance(value, list) or any(
                        not isinstance(name, str)
                        or Path(name).name != name
                        or not (target / name).is_file()
                        for name in value
                    ):
                        raise unavailable()
                elif key.endswith("_file"):
                    if (
                        not isinstance(value, str)
                        or Path(value).name != value
                        or not (target / value).is_file()
                    ):
                        raise unavailable()
                    config[key] = str(target / value)
            # Keep all configured tokenizer behavior; resolve only file locations.
            config_path.write_text(json.dumps(config), encoding="utf-8")
        try:
            return AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
                target, local_files_only=True, trust_remote_code=False, use_fast=True
            )
        except MemoryError:
            raise
        except Exception as exc:
            # The Rust tokenizer deserializer also raises plain Exception for
            # invalid/unsupported tokenizer.json files, not just ValueError.
            raise unavailable() from exc


@dataclass
class _Loaded:
    tokenizer: Any
    # HF wrappers may update internal padding/truncation state even on a call
    # requesting neither. Serialize use of an instance, never set shared options.
    lock: LockType = field(default_factory=Lock)


class TokenizerService:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._loaded: dict[str, _Loaded] = {}
        self._lock = Lock()

    def tokenize(
        self, source: ModelSource, text: str, add_special_tokens: bool = True
    ) -> dict[str, Any]:
        """Run on BlockingWork; snapshot guards also wrap every cached use."""
        with source.local_directory() as directory:
            with self._lock:
                loaded = self._loaded.get(source.fingerprint)
                if loaded is None:
                    loaded = _Loaded(load_tokenizer(self._root, directory))
                    source.check_unchanged()
                    self._loaded[source.fingerprint] = loaded
            with loaded.lock:
                tokenizer = loaded.tokenizer
                options = dict(
                    add_special_tokens=add_special_tokens,
                    padding=False,
                    truncation=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
                if tokenizer.is_fast:
                    try:
                        encoded = tokenizer(text, return_offsets_mapping=True, **options)
                    except NotImplementedError:
                        encoded = tokenizer(text, **options)
                else:
                    encoded = tokenizer(text, **options)
                ids = encoded["input_ids"]
                offsets = encoded.get("offset_mapping")
                if offsets is not None and len(offsets) != len(ids):
                    offsets = None
                special_ids = set(tokenizer.all_special_ids)
                tokens = []
                for index, token_id in enumerate(ids):
                    if type(token_id) is not int or not 0 <= token_id <= MAX_SAFE_INTEGER:
                        raise unavailable()
                    token = dict(
                        index=index,
                        id=token_id,
                        token=tokenizer.convert_ids_to_tokens(token_id),
                        decoded=tokenizer.decode(
                            [token_id],
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        ),
                        special=token_id in special_ids,
                    )
                    if offsets is not None:
                        span = offsets[index]
                        # HF fast offsets use original-source Unicode code points.
                        # Keep overlaps and source-typed specials; (0, 0) is a sentinel.
                        if (
                            isinstance(span, (tuple, list))
                            and len(span) == 2
                            and all(type(n) is int for n in span)
                            and 0 <= span[0] <= span[1] <= len(text)
                            and tuple(span) != (0, 0)
                        ):
                            token.update(start=span[0], end=span[1])
                    tokens.append(token)
                return dict(text=text, add_special_tokens=add_special_tokens, tokens=tokens)
