import asyncio
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from test_models import make_model, mutate_last_byte

from llm_model_explorer.app import create_app
from llm_model_explorer.model_files import ModelError
from llm_model_explorer.services import open_services
from llm_model_explorer.settings import Settings


def test_sessions_inventory_refresh_isolation_and_restart(settings: Settings) -> None:
    first = make_model(settings.model_root)
    make_model(settings.model_root, "second", identity="test/second")
    app = create_app(settings)
    with TestClient(app) as client:
        a = client.post("/sessions", json={"model_id": "test/tiny"})
        b = client.post("/sessions", json={"model_id": "test/second"})
        assert a.status_code == b.status_code == 201
        a_id, b_id = a.json()["id"], b.json()["id"]
        assert UUID(a_id) != UUID(b_id)
        assert a.json() == {"id": a_id, "model_id": "test/tiny"}
        assert client.get(f"/sessions/{a_id}").json() == a.json()
        inventory = client.get(f"/sessions/{a_id}/tensors")
        assert inventory.status_code == 200
        assert inventory.json()["coverage"] == "complete"
        assert inventory.json()["diagnostics"] == []
        assert client.get(f"/sessions/{a_id}/architecture").status_code == 200
        tensor = inventory.json()["tensors"][0]
        assert tensor["name"] == "layer.weight"
        assert tensor["path"] == ["layer", "weight"]
        assert tensor["shape"] == [2, 2]
        assert tensor["rank"] == 2 and tensor["numel"] == 4
        assert tensor["logical_dtype"] == "float32"
        assert str(settings.model_root) not in inventory.text + a.text
        mutate_last_byte(first / "model.safetensors")
        assert client.get(f"/sessions/{a_id}").status_code == 409
        assert client.get(f"/sessions/{a_id}/tensors").json()["code"] == "model_content_changed"
        assert client.get(f"/sessions/{b_id}").status_code == 200
        assert client.delete(f"/sessions/{a_id}").status_code == 204
        assert client.delete(f"/sessions/{a_id}").status_code == 404
        assert client.get(f"/sessions/{a_id}").status_code == 404
        assert client.get(f"/sessions/{b_id}").status_code == 200
    with TestClient(app) as client:
        assert client.get(f"/sessions/{b_id}").status_code == 404


@pytest.mark.parametrize("body", [{}, {"model_id": ""}, {"model_id": 1}, {"model_id": "x", "x": 1}])
def test_create_validation(settings: Settings, body: dict[str, object]) -> None:
    with TestClient(create_app(settings)) as client:
        response = client.post("/sessions", json=body)
        assert response.status_code == 422
        assert response.json()["code"] == "validation_error"
        assert response.headers["cache-control"] == "no-store"


def test_errors_and_idempotent_operation_delete(settings: Settings) -> None:
    with TestClient(create_app(settings)) as client:
        missing = client.post("/sessions", json={"model_id": "unknown"})
        assert missing.status_code == 404
        assert missing.json()["code"] == "model_not_found"
        malformed = client.post(
            "/sessions", content="{", headers={"Content-Type": "application/json"}
        )
        assert malformed.status_code == 400
        assert malformed.json()["code"] == "malformed_json"
        for method, path in [
            ("GET", "/sessions/bad"),
            ("DELETE", "/sessions/bad"),
            ("GET", "/sessions/bad/tensors"),
            ("DELETE", "/operations/bad"),
        ]:
            response = client.request(method, path)
            assert response.status_code == 422
            assert response.json()["code"] == "validation_error"
            assert response.headers["cache-control"] == "no-store"
        for path in [f"/sessions/{uuid4()}", f"/sessions/{uuid4()}/tensors"]:
            assert client.get(path).json()["code"] == "session_not_found"
        operation = uuid4()
        for _ in range(2):
            response = client.delete(f"/operations/{operation}")
            assert response.status_code == 204 and not response.content
            assert response.headers["cache-control"] == "no-store"


def test_unknown_tensor_and_model_binding(settings: Settings) -> None:
    make_model(settings.model_root)

    async def scenario() -> None:
        async with open_services(settings) as services:
            assert services.sessions is not None
            session = await services.sessions.create("test/tiny")
            with pytest.raises(ModelError) as error:
                await services.sessions.tensor(session.id, "missing")
            assert error.value.code == "tensor_not_found"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "error,status,code",
    [
        (MemoryError(), 503, "resource_exhausted"),
        (OSError("/private/model/path"), 500, "internal_error"),
    ],
)
def test_safe_errors(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, error: Exception, status: int, code: str
) -> None:
    def fail(model_id: str) -> None:
        raise error

    app = create_app(settings)
    with TestClient(app) as client:
        monkeypatch.setattr(app.state.services.catalogue, "pin", fail)
        response = client.post("/sessions", json={"model_id": "anything"})
        assert response.status_code == status
        assert response.json()["code"] == code
        assert "/private" not in response.text
        assert response.headers["cache-control"] == "no-store"
