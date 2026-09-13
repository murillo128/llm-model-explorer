import asyncio
from uuid import UUID

from fastapi.testclient import TestClient
from test_models import make_model

from llm_model_explorer.app import create_app
from llm_model_explorer.artifacts import ArtifactSpec
from llm_model_explorer.operations import Consumer, OperationCancelled, ProducerContext
from llm_model_explorer.services import Services
from llm_model_explorer.settings import Settings


def test_http_delete_releases_only_its_session_and_operation(settings: Settings) -> None:
    make_model(settings.model_root)
    app = create_app(settings)
    with TestClient(app) as client:
        assert client.portal is not None
        first = UUID(client.post("/sessions", json={"model_id": "test/tiny"}).json()["id"])
        second = UUID(client.post("/sessions", json={"model_id": "test/tiny"}).json()["id"])
        services: Services = app.state.services
        assert services.sessions is not None
        assert services.operation_delivery is not None
        sessions, operations = services.sessions, services.operation_delivery
        finish = asyncio.Event()
        calls = 0

        async def produce(ctx: ProducerContext) -> None:
            nonlocal calls
            calls += 1
            await ctx.append(b"abcd")
            await ctx.cancellation.wait(finish)
            await ctx.append(b"efgh")

        async def start(session_id: UUID) -> Consumer:
            session = sessions.require(session_id)
            artifact = ArtifactSpec(
                model_fingerprint=session.source.fingerprint,
                source="weight",
                operation="test",
                parameters={},
                dtype="float32",
                layout="C",
                shape=(2,),
                expected_bytes=8,
                producer="test",
            )
            return await sessions.subscribe(session_id, artifact, produce)

        a = client.portal.call(start, first)
        b = client.portal.call(start, second)
        c = client.portal.call(start, second)
        assert client.portal.call(a.read, 4) == b"abcd"
        assert client.delete(f"/sessions/{first}").status_code == 204
        assert a.terminal == "cancelled"
        assert client.delete(f"/operations/{b.operation_id}").status_code == 204
        assert b.terminal == "cancelled"
        assert not c._flight.cancellation.requested.is_set()
        client.portal.call(finish.set)

        async def collect() -> bytes:
            try:
                data = b""
                while chunk := await c.read():
                    data += chunk
                return data
            finally:
                await c.aclose()

        assert client.portal.call(collect) == b"abcdefgh"
        assert calls == 1
        assert not operations._operations
        assert not operations._flights
        assert client.delete(f"/operations/{c.operation_id}").status_code == 204
        assert c.terminal == "complete"

        async def cannot_read_cancelled() -> None:
            try:
                await b.read()
            except OperationCancelled:
                return
            raise AssertionError("cancelled operation delivered data")

        client.portal.call(cannot_read_cancelled)
