import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from fastapi import FastAPI

from llm_model_explorer.cli import main


def test_installed_help_needs_no_paths_or_torch() -> None:
    executable = Path(sys.executable).parent / "llm-model-explorer-backend"
    result = subprocess.run([str(executable), "--help"], capture_output=True, text=True, check=True)
    for option in ["--model-root", "--cache-dir", "--device", "--host", "--port", "--cors-origin"]:
        assert option in result.stdout
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import llm_model_explorer.cli; "
            "assert 'torch' not in sys.modules; assert 'uvicorn' not in sys.modules",
        ],
        check=True,
    )


@pytest.mark.parametrize("extra", [["--port", "0"], ["--port", "x"], ["--device", "invalid"]])
def test_invalid_cli_exits_with_local_diagnostic(
    model_root: Path, tmp_path: Path, extra: list[str]
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "llm_model_explorer",
            "--model-root",
            str(model_root),
            "--cache-dir",
            str(tmp_path / "cache"),
            *extra,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "cache").exists()


def test_cli_rejects_unavailable_cuda(
    model_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--model-root",
                str(model_root),
                "--cache-dir",
                str(tmp_path / "cache"),
                "--device",
                "cuda:0",
            ]
        )
    assert error.value.code == 2
    assert "cuda:0 is unavailable" in capsys.readouterr().err
    assert not (tmp_path / "cache").exists()


def test_cli_passes_network_settings(
    model_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = Mock()
    monkeypatch.setattr("uvicorn.run", run)
    main(
        [
            "--model-root",
            str(model_root),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--host",
            "0.0.0.0",
            "--port",
            "8123",
            "--cors-origin",
            "http://localhost:5173",
            "--cors-origin",
            "https://remote.example",
        ]
    )
    app = run.call_args.args[0]
    assert isinstance(app, FastAPI)
    assert run.call_args.kwargs == {"host": "0.0.0.0", "port": 8123}
    assert app.state.settings.cors_origins == ("http://localhost:5173", "https://remote.example")


def test_installed_server_starts_and_stops_on_cpu(model_root: Path, tmp_path: Path) -> None:
    # Reserve a free loopback port, then hand it to the actual installed CLI.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    executable = Path(sys.executable).parent / "llm-model-explorer-backend"
    with (tmp_path / "server.log").open("w+") as log:
        process = subprocess.Popen(
            [
                str(executable),
                "--model-root",
                str(model_root),
                "--cache-dir",
                str(tmp_path / "cache"),
                "--port",
                str(port),
            ],
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 15
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", trust_env=False) as client:
                while True:
                    if process.poll() is not None or time.monotonic() > deadline:
                        log.seek(0)
                        pytest.fail(f"server failed to start: {log.read()}")
                    try:
                        response = client.get("/models", timeout=0.2)
                        break
                    except httpx.TransportError:
                        time.sleep(0.05)
            assert response.status_code == 404
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
                pytest.fail("server did not shut down cleanly")
        log.seek(0)
        output = log.read()
    assert "Application startup complete" in output
    assert "Application shutdown complete" in output
    # Uvicorn re-raises the original termination signal after graceful cleanup.
    assert process.returncode in (0, -signal.SIGTERM)
