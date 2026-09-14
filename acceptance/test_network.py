"""Product acceptance over TCP; no TestClient or ASGI in-process transport."""

import json
import math
import os
import shutil
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
from transformers import AutoTokenizer

from acceptance.fixtures import MODEL_ID, SHAPES, TEXT, generate, values

REPO = Path(__file__).resolve().parents[1]
MATRIX = "model.layers.0.mlp.down_proj.weight"


class Service:
    def __init__(self, root, device="cpu", model_root=None, startup_timeout=30):
        self.root = root
        self.startup_timeout = startup_timeout
        self.device = device
        self.model_root = model_root
        if model_root is None:
            generate(root / "models")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.client = httpx.Client(base_url=f"http://127.0.0.1:{self.port}", timeout=20)
        self.process = None
        try:
            self.start()
        except BaseException:
            self.stop()
            self.client.close()
            raise

    def start(self):
        self.log = (self.root / "service.log").open("a")
        command = [
            sys.executable,
            "-m",
            "acceptance.server",
            "--root",
            str(self.root),
            "--port",
            str(self.port),
            "--device",
            self.device,
        ]
        if self.model_root is not None:
            command = [
                sys.executable,
                "-m",
                "llm_model_explorer",
                "--model-root",
                str(self.model_root),
                "--cache-dir",
                str(self.root / "cache"),
                "--port",
                str(self.port),
                "--device",
                self.device,
            ]
        self.process = subprocess.Popen(
            command,
            cwd=REPO,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "HF_HUB_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
            },
        )
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise AssertionError((self.root / "service.log").read_text())
            try:
                if self.client.get("/models").status_code == 200:
                    return
            except httpx.TransportError:
                pass
            time.sleep(0.05)
        raise AssertionError("Service did not start")

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
                raise
            finally:
                self.log.close()
            self.process = None

    def state(self):
        response = self.client.get("/__test/state")
        response.raise_for_status()
        return response.json()

    def wait(self, predicate):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            state = self.state()
            if predicate(state):
                return state
            time.sleep(0.02)
        raise AssertionError(f"State did not converge: {state}")

    def idle(self):
        return self.wait(
            lambda s: all(
                s[k] == 0
                for k in (
                    "operations",
                    "consumers",
                    "readers",
                    "tasks",
                    "flights",
                    "temporary",
                )
            )
        )

    def session(self):
        response = self.client.post("/sessions", json={"model_id": MODEL_ID})
        assert response.status_code == 201
        return response.json()["id"]

    def path(self, session, name=MATRIX, kind="data"):
        response = self.client.get(f"/sessions/{session}/tensors")
        assert response.status_code == 200
        descriptor = next(t for t in response.json()["tensors"] if t["name"] == name)
        return f"/sessions/{session}/tensors/{descriptor['id']}/{kind}"

    def arm(self, kind="logical_tensor", mode="barrier"):
        assert self.client.post("/__test/arm", json={"kind": kind, "mode": mode}).status_code == 200

    def release(self):
        assert self.client.post("/__test/release").status_code == 200


@pytest.fixture
def service(tmp_path):
    server = Service(tmp_path)
    try:
        yield server
    finally:
        server.stop()
        server.client.close()


class Frames:
    """Independent wire reader checks framing across arbitrary HTTP chunks."""

    def __init__(self, response):
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["content-type"] == "application/vnd.llm-model-explorer.stream"
        self.operation = response.headers["x-operation-id"]
        self.chunks = response.iter_bytes()
        self.pending = bytearray()

    def exact(self, n):
        while len(self.pending) < n:
            self.pending.extend(next(self.chunks))
        result = bytes(self.pending[:n])
        del self.pending[:n]
        return result

    def next(self):
        magic, kind, flags, reserved, size = struct.unpack("<4sBBHI", self.exact(12))
        assert (magic, flags, reserved) == (b"LMEX", 0, 0)
        assert kind in range(1, 7)
        assert kind != 2 or size % 4 == 0
        return kind, self.exact(size)

    def rest(self):
        frames = []
        while True:
            frame = self.next()
            frames.append(frame)
            if frame[0] in (4, 5, 6):
                assert frame[0] == 5 or frame[1] == b""
                assert not self.pending
                assert list(self.chunks) == []
                return frames


def result(service, path):
    with service.client.stream("GET", path) as response:
        frames = Frames(response).rest()
    assert frames[0][0] == 1 and frames[-1] == (4, b"")
    meta = json.loads(frames[0][1])
    payload = b"".join(body for kind, body in frames if kind == 2)
    assert len(payload) == meta["byte_length"]
    return meta, payload


def test_ten_operations_scientific_values_and_unicode(service):
    models = service.client.get("/models").json()["models"]
    assert [m["id"] for m in models] == [MODEL_ID]
    assert str(service.root) not in json.dumps(models)
    session = service.session()
    assert service.client.get(f"/sessions/{session}").json()["model_id"] == MODEL_ID
    for name, shape in SHAPES.items():
        meta, payload = result(service, service.path(session, name))
        assert meta["shape"] == list(shape)
        assert payload == values(shape).numpy().astype("<f4").tobytes()
    meta, payload = result(service, service.path(session, "science.weight"))
    scientific = np.frombuffer(payload, dtype="<f4")
    finite = scientific[np.isfinite(scientific)].astype(np.float64)
    stats, data = result(service, service.path(session, "science.weight", "statistics"))
    assert data == b"" and stats["count"] == 12 and stats["finite_count"] == 10
    for key, expected in {
        "minimum": finite.min(),
        "maximum": finite.max(),
        "mean": finite.mean(),
        "stddev": finite.std(),
    }.items():
        assert stats[key] == pytest.approx(expected, rel=1e-6, abs=1e-7)
    for key, q in zip(
        ["p01", "p05", "p50", "p95", "p99"], [0.01, 0.05, 0.5, 0.95, 0.99], strict=True
    ):
        assert stats["percentiles"][key] == pytest.approx(
            np.quantile(finite, q), rel=1e-6, abs=1e-7
        )
    meta, payload = result(service, service.path(session, "science.weight", "distributions"))
    expected_rows, expected_cols = (
        np.zeros((3, 100), dtype="<u4"),
        np.zeros((100, 4), dtype="<u4"),
    )
    for i, value in enumerate(scientific):
        if math.isfinite(value):
            bin_index = min(99, math.floor((float(value) + 2) / 8 * 100))
            expected_rows[i // 4, bin_index] += 1
            expected_cols[bin_index, i % 4] += 1
    assert payload == expected_rows.tobytes() + expected_cols.tobytes()
    assert [s["shape"] for s in meta["sections"]] == [[3, 100], [100, 4]]
    tokenizer = AutoTokenizer.from_pretrained(
        service.root / "models/fixture", local_files_only=True
    )
    response = service.client.post(f"/sessions/{session}/tokenize", json={"text": TEXT}).json()
    native = tokenizer(TEXT, return_offsets_mapping=True)
    assert response["text"] == TEXT
    assert [t["id"] for t in response["tokens"]] == native["input_ids"]
    for token, (start, end) in zip(response["tokens"], native["offset_mapping"], strict=True):
        if end > start:
            assert (token["start"], token["end"]) == (start, end)
            assert 0 <= start < end <= len(TEXT)
    assert response["tokens"][0]["special"] and "start" not in response["tokens"][0]
    assert (
        service.client.delete("/operations/00000000-0000-0000-0000-000000000001").status_code == 204
    )
    assert service.client.delete(f"/sessions/{session}").status_code == 204
    assert service.client.get(f"/sessions/{session}").status_code == 404
    service.idle()


def test_progressive_shared_late_and_slow_consumers(service, record_property):
    first, second = service.session(), service.session()
    path = service.path(first)
    service.arm()
    start = time.monotonic()
    with service.client.stream("GET", path) as a:
        af = Frames(a)
        assert af.next()[0] == 1
        prefix = af.next()
        assert prefix[0] == 2
        first_data = time.monotonic() - start
        state = service.wait(lambda s: s["control"]["entered"])
        assert not state["artifacts"] and sum(state["productions"].values()) == 1
        with service.client.stream("GET", service.path(second)) as b:
            bf = Frames(b)
            assert bf.operation != af.operation
            assert bf.next()[0] == 1
            assert bf.next() == prefix  # late joiner reads from offset zero
            service.release()
            # Leave a deliberately unread while b completes the entire producer.
            b_rest = bf.rest()
            assert b_rest[-1][0] == 4
            assert (
                len(prefix[1]) + sum(len(body) for kind, body in b_rest if kind == 2)
                == 576 * 1536 * 4
            )
            completed = time.monotonic() - start
            parked = service.state()
            assert sum(parked["productions"].values()) == 1
            assert len(parked["artifacts"]) == 1
            # Kernel buffers may already own the remaining bytes; early release is valid.
            assert parked["readers"] <= 1
            a_rest = af.rest()
            assert a_rest[-1][0] == 4
            assert (
                prefix[1] + b"".join(body for kind, body in a_rest if kind == 2)
                == values((576, 1536)).numpy().astype("<f4").tobytes()
            )
    assert first_data < completed
    record_property("first_data_seconds", first_data)
    record_property("completion_seconds", completed)
    before = service.idle()
    result(service, path)
    after = service.idle()
    assert after["artifacts"] == before["artifacts"]  # key, digest, inode and mtime unchanged
    assert after["productions"] == before["productions"]


@pytest.mark.parametrize("action", ["one", "all", "session", "disconnect"])
def test_cancel_and_disconnect_release_owned_work(service, action):
    a, b = service.session(), service.session()
    service.arm()
    with service.client.stream("GET", service.path(a)) as response:
        frames = Frames(response)
        assert frames.next()[0] == 1
        assert frames.next()[0] == 2
        service.wait(lambda s: s["control"]["entered"])
        if action in ("one", "session"):
            with service.client.stream("GET", service.path(b)) as other:
                remaining = Frames(other)
                assert remaining.next()[0] == 1 and remaining.next()[0] == 2
                target = (
                    f"/sessions/{a}" if action == "session" else f"/operations/{frames.operation}"
                )
                assert service.client.delete(target).status_code == 204
                assert frames.rest()[-1][0] == 6
                assert service.state()["tasks"] == 1
                service.release()
                assert remaining.rest()[-1][0] == 4
        elif action == "all":
            with service.client.stream("GET", service.path(b)) as other:
                remaining = Frames(other)
                assert remaining.next()[0] == 1 and remaining.next()[0] == 2
                assert service.client.delete(f"/operations/{frames.operation}").status_code == 204
                assert (
                    service.client.delete(f"/operations/{remaining.operation}").status_code == 204
                )
                assert frames.rest()[-1][0] == 6
                assert remaining.rest()[-1][0] == 6
        else:
            response.close()
    state = service.idle()
    assert len(state["artifacts"]) == (1 if action in ("one", "session") else 0)


@pytest.mark.parametrize(
    "kind,mode",
    [("tensor_statistics", "pre-meta-error"), ("logical_tensor", "midstream-error")],
)
def test_real_stream_failures_never_publish_partial_artifacts(service, kind, mode):
    session = service.session()
    service.arm(kind, mode)
    path = service.path(session, kind="statistics" if kind == "tensor_statistics" else "data")
    with service.client.stream("GET", path) as response:
        frames = Frames(response)
        if mode == "midstream-error":
            assert frames.next()[0] == 1 and frames.next()[0] == 2
            service.wait(lambda s: s["control"]["entered"])
            service.release()
        terminal = frames.rest()
        assert len(terminal) == 1 and terminal[0][0] == 5
        error = json.loads(terminal[0][1])
        assert error["code"] == "internal_error" and str(service.root) not in str(error)
    state = service.idle()
    assert not any(a["manifest"]["spec"]["operation"] == kind for a in state["artifacts"].values())


def test_restart_invalidation_and_disposable_cache(service):
    session = service.session()
    path = service.path(session)
    original = result(service, path)[1]
    before = service.idle()["artifacts"]
    service.stop()
    service.start()
    assert service.client.get(f"/sessions/{session}").status_code == 404
    session = service.session()
    path = service.path(session)
    assert result(service, path)[1] == original
    assert service.idle()["artifacts"] == before
    assert not service.state()["productions"]
    asset = service.root / "models/fixture/model.safetensors"
    with asset.open("r+b") as file:
        file.seek(-2, 2)
        byte = file.read(1)
        file.seek(-1, 1)
        file.write(bytes([byte[0] ^ 1]))
    response = service.client.get(path)
    assert response.status_code == 409 and response.json()["code"] == "model_content_changed"
    new_session = service.session()
    result(service, service.path(new_session))
    after = service.idle()["artifacts"]
    assert len(after) == 2 and set(before) < set(after)
    assert len({a["manifest"]["spec"]["model_fingerprint"] for a in after.values()}) == 2
    service.stop()
    shutil.rmtree(service.root / "cache")
    service.start()
    assert not service.state()["artifacts"]
    result(service, service.path(service.session()))
    assert len(service.idle()["artifacts"]) == 1


def test_cors_and_preflight_error(service):
    origin = "http://127.0.0.1:4175"
    response = service.client.options(
        "/sessions",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
    bad = service.client.get("/models", headers={"Origin": "http://untrusted.invalid"})
    assert "access-control-allow-origin" not in bad.headers
    session = service.session()
    response = service.client.get(
        service.path(session, "model.norm.weight", "distributions"),
        headers={"Origin": origin},
    )
    assert response.status_code == 422 and response.json()["code"] == "unsupported_rank"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["access-control-allow-origin"] == origin


def test_cuda_fixture(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip(
            "CUDA unavailable in the installed PyTorch/runtime; CPU acceptance is mandatory"
        )
    service = Service(tmp_path, "cuda:0")
    try:
        session = service.session()
        meta, _ = result(service, service.path(session, "science.weight", "statistics"))
        assert meta["mean"] == pytest.approx(1.8, abs=1e-7, rel=1e-6)
        service.idle()
    finally:
        service.stop()
        service.client.close()


def test_distribution_bytes_arrive_before_publication(service):
    session = service.session()
    service.arm("tensor_distributions")
    with service.client.stream("GET", service.path(session, kind="distributions")) as response:
        frames = Frames(response)
        kind, body = frames.next()
        assert kind == 1 and json.loads(body)["kind"] == "tensor_distributions"
        assert frames.next()[0] == 2
        state = service.wait(lambda s: s["control"]["entered"])
        assert not state["control"]["released"]
        assert all(
            a["manifest"]["spec"]["operation"] != "tensor_distributions"
            for a in state["artifacts"].values()
        )
        service.release()
        assert frames.rest()[-1][0] == 4
    service.idle()
