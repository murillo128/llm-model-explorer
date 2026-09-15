# Admitted packed tensor decoding

## Numeric reference and ownership

`quantized_decoding.decode_range` reads a bounded flat range of a logical
`[output, input]` matrix and returns owned CPU float32 values. It accepts only
the two storage layouts validated by checkpoint admission. Configuration and
structural admission are unchanged.

The independent scalar oracles in `tests/quantized_oracles.py` use `struct` and
scalar arithmetic, without production conversion helpers or PyTorch lookup
tables. They record these exact reviewed implementations:

- [AutoGPTQ qlinear_cuda.py at 9f7d370](https://github.com/AutoGPTQ/AutoGPTQ/blob/9f7d37072917ab3a7545835f23e808294a542153/auto_gptq/nn_modules/qlinear/qlinear_cuda.py):
  GPTQ v1 low-first Int4 packing, restoration of zero points by adding one
  after extracting the nibble (without modulo wrapping), and stored `g_idx`.
- [GPTQModel at 40759cd](https://github.com/ModelCloud/GPTQModel/blob/40759cdf06c17ea6c57637fa075f8c10547b4a6f/gptqmodel/utils/model.py):
  the selected producer's GPTQ v1 convention. Canonical float32 decoding avoids
  the additional half-precision output rounding of a half-precision matmul path.
- [ModelOpt NVFP4 at 82f1d21](https://github.com/NVIDIA/Model-Optimizer/blob/82f1d216d1a9022e60b8e1143a77f36c00b885a4/modelopt/torch/quantization/qtensor/nvfp4_tensor.py):
  low-first E2M1, E4M3 block scales multiplied by the global weight scale in
  float32 before multiplication by the decoded value. `input_scale` is absent
  from weight dequantization. Standard E2M1 negative zero is preserved; the
  upstream Python lookup collapses it to positive zero, a sign-bit distinction
  called out separately from nonzero numerical agreement.

Production arithmetic and gathering use PyTorch kernels. Each gather coalesces
requested unique storage positions into contiguous file spans and reads them
into owned bytes, then restores requested order with PyTorch. Python loops over
I/O spans only. Read lengths and total copied bytes are proportional to the
requested chunk, including strided GPTQ columns. Snapshot and short-read checks
detect concurrent mutation/truncation without dereferencing live file mappings.
There is no full-checkpoint float32 allocation or new cache product.

Fixture tests cover complete matrices, every nibble position, signed zeros,
zero-point endpoints, multiple/nontrivial repeated group mappings, half and
float32 scales, E4M3 boundaries, arbitrary range alignment, separated companion
shards, invalid group indices, safe ranges, source mutation, and bounded owned
output. Their synthetic bytes are distinct from actual-checkpoint evidence.

The corrected standalone decoder checkpoint passes 34 focused tests, Ruff and
mypy using the locked Python 3.12 CPU environment. The six concurrent-truncation
cases cover both formats before a file read, after a read, and immediately before
the vectorized gather; each yields `model_content_changed` without a process
crash. The earlier live-mapping implementation was rejected at intermediate
review after a SIGBUS reproduction and is replaced by owned file reads.
