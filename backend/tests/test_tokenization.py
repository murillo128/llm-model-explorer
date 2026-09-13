import asyncio
import builtins
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import Any
from unittest.mock import Mock
from uuid import uuid4

import httpx
import pytest
import requests
import torch
from fastapi.testclient import TestClient
from test_models import make_model, mutate_last_byte
from tokenizers import (  # type: ignore[import-untyped]
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from llm_model_explorer import tokenization
from llm_model_explorer.app import create_app
from llm_model_explorer.model_files import ModelError
from llm_model_explorer.models import ModelCatalogue
from llm_model_explorer.settings import Settings
from llm_model_explorer.tokenization import TokenizerService

TEXTS = [
    "",
    "  hello   world ",
    "\nhello\tworld\n",
    "España, pingüino, café",
    "A😀e\u0301<special>",
    "你好世界",
    "😀🚀",
    "HELLO Café",
    "<special>",
]


def make_tokenizer(root: Path, kind: str = "byte", name: str = "tiny") -> tuple[Path, Any]:
    directory = make_model(root, name, identity=f"test/{name}")
    if kind == "byte":
        native = Tokenizer(models.BPE(unk_token="<unk>"))
        native.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        native.decoder = decoders.ByteLevel()
        native.train_from_iterator(
            ["hello world"],
            trainers.BpeTrainer(
                vocab_size=260,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                special_tokens=["<unk>", "<bos>", "<special>"],
            ),
        )
    else:
        native = Tokenizer(models.WordPiece(unk_token="<unk>"))
        native.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase()])
        native.pre_tokenizer = pre_tokenizers.Whitespace()
        native.decoder = decoders.WordPiece()
        native.train_from_iterator(
            TEXTS,
            trainers.WordPieceTrainer(
                vocab_size=100,
                special_tokens=["<unk>", "<bos>", "<special>"],
            ),
        )
    native.post_processor = processors.TemplateProcessing(
        single="<bos> $A",
        special_tokens=[("<bos>", native.token_to_id("<bos>"))],
    )
    tokenizer = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=native,
        unk_token="<unk>",
        bos_token="<bos>",
        additional_special_tokens=["<special>"],
        model_max_length=2,
    )
    tokenizer.save_pretrained(directory)
    return directory, AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        directory,
        local_files_only=True,
        trust_remote_code=False,
    )


def expected_tokens(reference: Any, text: str, specials: bool) -> list[dict[str, Any]]:
    encoded = reference(text, add_special_tokens=specials, return_offsets_mapping=True)
    result = []
    for index, (token_id, span) in enumerate(
        zip(
            encoded["input_ids"],
            encoded["offset_mapping"],
            strict=True,
        )
    ):
        token = dict(
            index=index,
            id=token_id,
            token=reference.convert_ids_to_tokens(token_id),
            decoded=reference.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
            special=token_id in reference.all_special_ids,
        )
        if span != (0, 0):
            token.update(start=span[0], end=span[1])
        result.append(token)
    return result


@pytest.mark.parametrize("kind", ["byte", "normalized"])
@pytest.mark.parametrize("specials", [False, True])
def test_real_endpoint_unicode_and_exact_reference(
    settings: Settings,
    kind: str,
    specials: bool,
) -> None:
    _, reference = make_tokenizer(settings.model_root, kind)
    with TestClient(create_app(settings)) as client:
        session = client.post("/sessions", json={"model_id": "test/tiny"}).json()["id"]
        for text in TEXTS:
            response = client.post(
                f"/sessions/{session}/tokenize",
                json={"text": text, "add_special_tokens": specials},
            )
            assert response.status_code == 200, response.text
            assert response.json() == dict(
                text=text,
                add_special_tokens=specials,
                tokens=expected_tokens(reference, text, specials),
            )
            assert response.headers["cache-control"] == "no-store"
            assert "x-operation-id" not in response.headers
            for token in response.json()["tokens"]:
                assert ("start" in token) == ("end" in token)
                if "start" in token:
                    assert 0 <= token["start"] <= token["end"] <= len(text)


def test_byte_overlaps_literal_special_and_normalized_mapping(settings: Settings) -> None:
    _, reference = make_tokenizer(settings.model_root)
    source = ModelCatalogue(settings.model_root).pin("test/tiny")
    service = TokenizerService(settings.model_root)
    tokens = service.tokenize(source, "A😀é<special>")["tokens"]
    emoji = [t for t in tokens if t.get("start") == 1 and t.get("end") == 2]
    assert len(emoji) > 1  # Multiple bytes cover the same source code point.
    assert tokens[0]["special"] and "start" not in tokens[0]
    assert tokens[-1]["special"] and (tokens[-1]["start"], tokens[-1]["end"]) == (4, 13)
    assert tokens == expected_tokens(reference, "A😀é<special>", True)
    _, reference = make_tokenizer(settings.model_root, "normalized", "normalized")
    source = ModelCatalogue(settings.model_root).pin("test/normalized")
    result = service.tokenize(source, "CAFÉ")
    assert result["text"] == "CAFÉ"
    assert result["tokens"] == expected_tokens(reference, "CAFÉ", True)
    assert "".join(t["token"] for t in result["tokens"]) != "CAFÉ"


@pytest.mark.parametrize("fast", [True, False])
def test_offsets_unavailable_preserves_ids_and_text(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    fast: bool,
) -> None:
    _, reference = make_tokenizer(settings.model_root)

    class WithoutOffsets:
        is_fast = fast
        all_special_ids = reference.all_special_ids
        convert_ids_to_tokens = reference.convert_ids_to_tokens
        decode = reference.decode

        def __call__(self, text: str, **kwargs: Any) -> Any:
            if kwargs.pop("return_offsets_mapping", False):
                raise NotImplementedError("offsets unavailable")
            return reference(text, **kwargs)

    monkeypatch.setattr(tokenization, "load_tokenizer", lambda *args: WithoutOffsets())
    result = TokenizerService(settings.model_root).tokenize(
        ModelCatalogue(settings.model_root).pin("test/tiny"),
        TEXTS[4],
        False,
    )
    expected = expected_tokens(reference, TEXTS[4], False)
    for token in expected:
        token.pop("start", None)
        token.pop("end", None)
    assert result["tokens"] == expected


@pytest.mark.parametrize("offsets", [[(-1, 2)], [(0, 999)], [(1,)], [(False, 1)], [], [(0, 0)]])
def test_invalid_offsets_are_omitted(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    offsets: Any,
) -> None:
    make_model(settings.model_root)
    fake = Mock(
        is_fast=True,
        all_special_ids=[],
        return_value={
            "input_ids": [5],
            "offset_mapping": offsets,
        },
    )
    fake.convert_ids_to_tokens.return_value = "x"
    fake.decode.return_value = "x"
    monkeypatch.setattr(tokenization, "load_tokenizer", lambda *args: fake)
    result = TokenizerService(settings.model_root).tokenize(
        ModelCatalogue(settings.model_root).pin("test/tiny"),
        "x",
    )
    assert result["tokens"] == [dict(index=0, id=5, token="x", decoded="x", special=False)]


@pytest.mark.parametrize("failure", ["missing", "corrupt", "remote-code", "external-file"])
def test_unavailable_tokenizer_is_path_free(
    settings: Settings,
    failure: str,
) -> None:
    directory = make_model(settings.model_root)
    if failure == "corrupt":
        (directory / "tokenizer.json").write_text("broken")
    elif failure == "remote-code":
        (directory / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "tokenizer_class": "PrivateTokenizer",
                    "auto_map": {"AutoTokenizer": ["private.PrivateTokenizer", None]},
                }
            )
        )
        (directory / "private.py").write_text("raise AssertionError('must not execute')")
    elif failure == "external-file":
        (directory / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "tokenizer_class": "PreTrainedTokenizerFast",
                    "tokenizer_file": "/private/tokenizer.json",
                }
            )
        )
    with TestClient(create_app(settings)) as client:
        session = client.post("/sessions", json={"model_id": "test/tiny"}).json()["id"]
        response = client.post(f"/sessions/{session}/tokenize", json={"text": "hello"})
        assert response.status_code == 422, response.text
        assert response.json() == {
            "code": "unsupported_representation",
            "message": "Local tokenizer is missing or unsupported.",
        }
        assert response.headers["cache-control"] == "no-store"


def test_snapshot_reuse_and_new_session_invalidation(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, _ = make_tokenizer(settings.model_root)
    loader = Mock(wraps=tokenization.load_tokenizer)
    monkeypatch.setattr(tokenization, "load_tokenizer", loader)
    with TestClient(create_app(settings)) as client:
        a, b = [
            client.post("/sessions", json={"model_id": "test/tiny"}).json()["id"] for _ in range(2)
        ]
        for session in (a, b):
            assert (
                client.post(f"/sessions/{session}/tokenize", json={"text": "x"}).status_code == 200
            )
        assert loader.call_count == 1
        config = directory / "tokenizer_config.json"
        data = json.loads(config.read_text())
        data["add_prefix_space"] = True
        config.write_text(json.dumps(data))
        assert (
            client.post(f"/sessions/{a}/tokenize", json={"text": "x"}).json()["code"]
            == "model_content_changed"
        )
        c = client.post("/sessions", json={"model_id": "test/tiny"}).json()["id"]
        assert client.post(f"/sessions/{c}/tokenize", json={"text": "x"}).status_code == 200
        assert loader.call_count == 2
        mutate_last_byte(directory / "model.safetensors")
        assert client.post(f"/sessions/{c}/tokenize", json={"text": "x"}).status_code == 409


def test_change_during_encode_is_rejected(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, reference = make_tokenizer(settings.model_root)

    class Mutating:
        is_fast = True
        all_special_ids = reference.all_special_ids
        convert_ids_to_tokens = reference.convert_ids_to_tokens
        decode = reference.decode

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            result = reference(*args, **kwargs)
            mutate_last_byte(directory / "model.safetensors")
            return result

    monkeypatch.setattr(tokenization, "load_tokenizer", lambda *args: Mutating())
    source = ModelCatalogue(settings.model_root).pin("test/tiny")
    with pytest.raises(ModelError, match="Model content changed"):
        TokenizerService(settings.model_root).tokenize(source, "hello")


def test_concurrent_sessions_options_and_single_load(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, reference = make_tokenizer(settings.model_root)
    loader = Mock(wraps=tokenization.load_tokenizer)
    monkeypatch.setattr(tokenization, "load_tokenizer", loader)
    expected = {
        specials: expected_tokens(reference, TEXTS[4], specials) for specials in (True, False)
    }
    with TestClient(create_app(settings)) as client:
        sessions = [
            client.post("/sessions", json={"model_id": "test/tiny"}).json()["id"] for _ in range(2)
        ]

        def request(index: int) -> None:
            specials = bool(index % 2)
            response = client.post(
                f"/sessions/{sessions[index % 2]}/tokenize",
                json={"text": TEXTS[4], "add_special_tokens": specials},
            )
            assert response.status_code == 200
            assert response.json()["tokens"] == expected[specials]
            assert response.json()["add_special_tokens"] == specials

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(request, range(40)))
        assert loader.call_count == 1


def test_no_network_weights_or_operation_runtime(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    make_tokenizer(settings.model_root)
    app = create_app(settings)
    with TestClient(app) as client:
        session = client.post("/sessions", json={"model_id": "test/tiny"}).json()["id"]
        forbidden = Mock(side_effect=AssertionError("forbidden I/O or computation"))
        monkeypatch.setattr(requests.Session, "request", forbidden)
        monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
        monkeypatch.setattr(torch, "load", forbidden)
        monkeypatch.setattr(torch, "frombuffer", forbidden)
        monkeypatch.setattr(app.state.services.operation_delivery, "subscribe", forbidden)
        original_open = builtins.open

        def checked_open(file: Any, *args: Any, **kwargs: Any) -> Any:
            assert not str(file).endswith(".safetensors")
            return original_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", checked_open)
        response = client.post(f"/sessions/{session}/tokenize", json={"text": "hello"})
        assert response.status_code == 200, response.text
        assert response.json()["add_special_tokens"] is True
        forbidden.assert_not_called()


def test_slow_encode_metadata_and_request_cancellation(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    make_tokenizer(settings.model_root)
    entered, release = Event(), Event()
    original = TokenizerService.tokenize

    def slow(self: TokenizerService, *args: Any, **kwargs: Any) -> Any:
        entered.set()
        assert release.wait(10)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TokenizerService, "tokenize", slow)

    async def scenario() -> None:
        app = create_app(settings)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                session = (await client.post("/sessions", json={"model_id": "test/tiny"})).json()[
                    "id"
                ]
                task = asyncio.create_task(
                    client.post(f"/sessions/{session}/tokenize", json={"text": "x"})
                )
                try:
                    assert await asyncio.to_thread(entered.wait, 5)
                    metadata = await asyncio.wait_for(client.get("/models"), timeout=2)
                    assert metadata.status_code == 200
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await asyncio.wait_for(task, timeout=1)
                    assert (
                        not release.is_set()
                    )  # Interest released while native work is still running.
                finally:
                    release.set()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "body", [{}, {"text": 3}, {"text": "x", "add_special_tokens": 1}, {"text": "x", "other": True}]
)
def test_validation(settings: Settings, body: dict[str, Any]) -> None:
    with TestClient(create_app(settings)) as client:
        response = client.post(f"/sessions/{uuid4()}/tokenize", json=body)
        assert response.status_code == 422
        assert response.json()["code"] == "validation_error"


def test_missing_session_malformed_json_and_uuid(settings: Settings) -> None:
    with TestClient(create_app(settings)) as client:
        response = client.post(f"/sessions/{uuid4()}/tokenize", json={"text": "x"})
        assert response.status_code == 404 and response.json()["code"] == "session_not_found"
        response = client.post("/sessions/bad/tokenize", json={"text": "x"})
        assert response.status_code == 422
        response = client.post(
            f"/sessions/{uuid4()}/tokenize",
            content="{",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400 and response.json()["code"] == "malformed_json"


def test_http_disconnect_releases_interest(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_tokenizer(settings.model_root)
    entered, release = Event(), Event()
    original = TokenizerService.tokenize

    def slow(self: TokenizerService, *args: Any, **kwargs: Any) -> Any:
        entered.set()
        assert release.wait(10)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TokenizerService, "tokenize", slow)

    async def scenario() -> None:
        app = create_app(settings)
        async with app.router.lifespan_context(app):
            session = await app.state.services.sessions.create("test/tiny")
            messages: asyncio.Queue[Any] = asyncio.Queue()
            await messages.put(
                {"type": "http.request", "body": b'{"text":"hello"}', "more_body": False}
            )
            sent: list[Any] = []

            async def send(message: Any) -> None:
                sent.append(message)

            task = asyncio.create_task(
                app(
                    {
                        "type": "http",
                        "asgi": {"version": "3.0"},
                        "http_version": "1.1",
                        "method": "POST",
                        "scheme": "http",
                        "path": f"/sessions/{session.id}/tokenize",
                        "query_string": b"",
                        "headers": [(b"content-type", b"application/json")],
                        "client": ("127.0.0.1", 123),
                        "server": ("test", 80),
                    },
                    messages.get,
                    send,
                )
            )
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                await messages.put({"type": "http.disconnect"})
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=1)
                assert sent == []
                assert not release.is_set()
            finally:
                release.set()

    asyncio.run(scenario())


def test_load_mutation_does_not_poison_reused_snapshot(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, _ = make_tokenizer(settings.model_root)
    source = ModelCatalogue(settings.model_root).pin("test/tiny")
    service = TokenizerService(settings.model_root)
    original = tokenization.load_tokenizer

    def load(root: Path, directory: Path) -> Any:
        tokenizer = original(root, directory)
        mutate_last_byte(directory / "model.safetensors")
        return tokenizer

    monkeypatch.setattr(tokenization, "load_tokenizer", load)
    with pytest.raises(ModelError, match="Model content changed"):
        service.tokenize(source, "hello")
    assert not service._loaded


@pytest.mark.parametrize("asset", ["tokenizer.json", "tokenizer_config.json"])
def test_invalid_real_tokenizer_assets(settings: Settings, asset: str) -> None:
    directory, _ = make_tokenizer(settings.model_root)
    (directory / asset).write_text("invalid JSON")
    with TestClient(create_app(settings)) as client:
        session = client.post("/sessions", json={"model_id": "test/tiny"}).json()["id"]
        response = client.post(f"/sessions/{session}/tokenize", json={"text": "hello"})
        assert response.status_code == 422
        assert response.json()["code"] == "unsupported_representation"
