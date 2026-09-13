# Logical tensor data evidence

`tests/test_tensor_data.py` exercises the real application endpoint and the
logical materialization service using generated, independently encoded local
safetensors. No model weights or runtime cache entries are committed.

- F32/F16/BF16 matrix, vector, scalar, empty and rank-3 results match independent
  `struct`-encoded float32 bytes, including asymmetric C-order values and signed
  zero. Repeated requests preserve the original source bytes.
- Every finite 16-bit F16 and BF16 encoding is widened and compared exactly
  against an independent expected result. F32 direct transport preserves positive
  and negative infinity, signed zero, subnormal, maximum finite, and NaN payload
  bit patterns, including a signaling NaN encoding.
- Real loopback HTTP reads a DATA prefix before a gated cold conversion finishes
  or publishes its artifact. Generated `[131075, 1]` tensors exceed common texture
  dimensions; concatenated DATA contains the complete expected tensor.
- Source instrumentation observes at most 65,536 elements per block. LMEX reads
  and frames are at most 256 KiB. Production retains bounded source/output blocks
  and spools to disk; slow readers do not accumulate a tensor-sized queue in RAM.
- Two sessions share exactly one F16/BF16 conversion but receive different public
  operation IDs. Cancelling one returns CANCELLED while the other completes.
  A subsequent warm request performs no source iteration/conversion. F32 requests
  create no artifact, and setting the runtime device to CUDA creates no GPU queue.
- Internal producer dependencies consume the same materialization seam without
  HTTP, receive no public operation ID, and release their runtime interests.
- Invalid session/tensor IDs, representation, dimensions and safe sizes fail
  preflight. Changed sources fail before headers and during active direct/warm
  reads; a newly pinned revision gets a different artifact key.
- Injected allocation, native PyTorch OOM, disk begin/append/commit, short output
  and source mutation failures never emit COMPLETE or publish partial artifacts.
  Memory/disk exhaustion is path-free `resource_exhausted`; source changes use
  `model_content_changed`. Partial cancellation cleans the spool. Session deletion
  closes an already-open direct source iterator.

Validation uses the repository-pinned Python 3.12 environment, Ruff, mypy and
pytest. The host was nearly full, so a fresh `uv sync --locked` could not extract
PyTorch; existing installed pinned packages were used with `PYTHONPATH=src` and
mypy caching disabled. No dependency or lockfile changes are needed. CUDA
hardware and a downloaded reference model are not required or exercised: the
accepted source seam widens ordinary safetensors on CPU. Quantized encodings
remain explicitly unsupported by the existing catalogue and service.

Local checks: Ruff check/format and strict mypy pass. The complete source and
installed-wheel suites each pass **354 tests** (38 new tensor-data cases).
Both suites were run serially with an explicit temporary fixture directory after
initial overlapping runs hit the host's disk limit. The wheel was built with
`uv build`, installed without dependencies into an isolated target, and tested
from `/tmp` using that target as `PYTHONPATH`. Existing warnings concern optional
NumPy absence and Starlette/AnyIO deprecations.
