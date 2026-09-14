"""Real TCP embedding delivery with an independent arithmetic/wire oracle."""

import json

import numpy as np
import pytest

from acceptance.test_network import Frames
from acceptance.test_network import service as service


def test_ordered_embedding_rows_are_progressive_bounded_and_ephemeral(service):
    session = service.session()
    ids = [258, 1, 258, 0]
    service.arm("input_embeddings")
    with service.client.stream(
        "POST",
        f"/sessions/{session}/embeddings",
        json={"token_ids": ids},
        headers={"Origin": "http://127.0.0.1:4175"},
    ) as response:
        assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:4175"
        assert "x-operation-id" in response.headers["access-control-expose-headers"].lower()
        frames = Frames(response)
        kind, body = frames.next()
        metadata = json.loads(body)
        assert kind == 1 and metadata["token_ids"] == ids
        assert metadata["shape"] == [len(ids), 576]
        assert metadata["byte_length"] == len(ids) * 576 * 4
        first = frames.next()
        assert first[0] == 2 and len(first[1]) < metadata["byte_length"]
        state = service.wait(lambda s: s["control"]["entered"])
        assert state["readers"] == 1 and state["operations"] == 1
        assert not state["artifacts"] and not state["source_reads"]["full_tensors"]
        service.release()
        rest = frames.rest()
        assert rest[-1] == (4, b"")
        payload = first[1] + b"".join(body for kind, body in rest if kind == 2)
    # Arithmetic is independent of fixture generation and production row lookup.
    indices = np.array(ids, dtype=np.int64)[:, None] * 576 + np.arange(576)
    expected = ((indices * 17 % 257 - 128) / 128).astype("<f4")
    assert payload == expected.tobytes()
    actual = np.frombuffer(payload, dtype="<f4").reshape(len(ids), 576)
    assert np.array_equal(actual[0], actual[2])
    state = service.idle()
    assert state["source_reads"]["row_elements"] == len(ids) * 576
    assert state["source_reads"]["max_row_block"] <= len(ids) * 576
    assert not state["source_reads"]["full_tensors"] and not state["artifacts"]


@pytest.mark.parametrize("action", ["operation", "session", "disconnect"])
def test_embedding_cancellation_releases_reader_without_publishing(service, action):
    session = service.session()
    service.arm("input_embeddings")
    with service.client.stream(
        "POST", f"/sessions/{session}/embeddings", json={"token_ids": [1, 2, 1]}
    ) as response:
        frames = Frames(response)
        assert frames.next()[0] == 1 and frames.next()[0] == 2
        service.wait(lambda s: s["control"]["entered"])
        if action == "disconnect":
            response.close()
        else:
            path = (
                f"/operations/{frames.operation}"
                if action == "operation"
                else f"/sessions/{session}"
            )
            assert service.client.delete(path).status_code == 204
            assert frames.rest()[-1] == (6, b"")
    state = service.idle()
    assert not state["artifacts"] and not state["source_reads"]["full_tensors"]
