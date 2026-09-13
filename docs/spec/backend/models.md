# Local models and tensors

## Supported model source

The proof of concept supports local Hugging Face model directories only. No generic provider abstraction is required for other formats.

The backend receives one local model root through configuration. Models are discovered beneath that root and exposed to the API without revealing local filesystem paths to the UI.

Discovery examines immediate child directories only. A candidate must have a
`config.json` object with a nonempty string `model_type`, plus safetensors files
or one `*.safetensors.index.json` with a nonempty `weight_map`. With an index,
the referenced shard inventory must exactly match its mapping; without one,
all immediate `*.safetensors` files form the checkpoint. Referenced shards may
use subdirectories. Duplicate tensor names, missing shards, invalid headers,
gaps/overlaps, and inconsistent tensor byte lengths invalidate the candidate.
Unrelated directories are ignored; invalid candidates are omitted and diagnosed
in backend logs. An empty valid catalogue returns an empty model list.

Shard references must be relative, canonical POSIX paths without traversal,
drive prefixes, or backslashes. Model directories, configuration, tokenizer
assets, and shards may use symlinks only when they resolve within the configured
root. The backend reads regular files only and never executes local Python code.

## Model identity

The public model identity should be derived from Hugging Face metadata when sufficient metadata is available. The local directory name is only a fallback identity when the model does not provide a suitable logical identity.

Filesystem paths are private backend implementation details.

The concrete identity fields are `config.json`'s `_name_or_path`, then
`name_or_path`. An eligible value is a name or `organization/name`, with each
segment starting in an ASCII letter, digit, or underscore and otherwise using
letters, digits, underscores, dots, or hyphens. `..`, absolute paths, URIs,
drive paths, and values identifying an existing path relative to the model
directory or backend working directory are ineligible. Without an eligible
value, use the immediate model directory's name. `model_type` and architecture
class names are never model identities. An eligible single-segment
`_commit_hash` (or `revision` when that field is absent) is appended as
`@revision`; display name excludes the revision suffix.

Identical copies with the same identity collapse to one entry. Equal identities
with different content fingerprints fail catalogue listing with a path-free
`validation_error`; the backend must not select one silently. Optional
`parameter_count` is omitted because checkpoint entries may include buffers or
omit tied parameters. `size_bytes`, when safe, counts the snapshot's selected
local asset bytes. Unknown/unusable architectures are represented by the
required empty list. Tokenizer availability indicates the presence of
conventional tokenizer assets (`tokenizer.json`, `tokenizer.model`,
`spiece.model`, `vocab.txt`, or the pair `vocab.json`/`merges.txt`), not a promise
that every tokenizer implementation can load them.

Each discovered model also has a content fingerprint used for cache validity. Moving an unchanged model to another local path must not invalidate artifacts merely because its path changed, while changing the actual model content must result in a different fingerprint and therefore different artifact keys.

The fingerprint is SHA-256 over a domain marker followed by sorted relative
asset names, framed name lengths/file sizes, and streamed file contents. Assets
include immediate `.json`, `.model`, `.txt`, `.tiktoken`, and `.safetensors`
files, plus every index-referenced shard. Absolute paths, resolved symlink
targets, inode numbers, and timestamps do not enter the digest. Hash reads are
bounded to 1 MiB. Source-layout or relevant metadata changes may invalidate the
fingerprint even when mathematical weights are unchanged.

Ordinary metadata listing reads configuration, indexes, and tensor headers only.
Full-file hashing is deferred to source pinning and duplicate-identity comparison.
Pinning captures a coherent snapshot for a session. Before/after hashing,
inventory access, guarded tokenizer work, and each tensor chunk, check the asset
set, resolved targets, device/inode, size, modification time, and change time.
Replacement, removal, addition, or mutation invalidates the pinned source with
`model_content_changed`; it never silently refreshes a session. Explicit rehash
is also available. This assumes ordinary filesystem change tracking in the
trusted deployment environment; a full rehash is not performed for each chunk.
The consumer must exhaust the iterator and pass the final check before treating
generation as complete or publishing an artifact.

## Loading strategy

Opening a model must not load the complete model into RAM or GPU memory. Metadata and tensor structure are discovered first; tensor content is accessed lazily and materialized only when requested.

Where the Hugging Face storage format allows it, tensor access should use memory mapping or equivalent lazy file access rather than copying the complete weight files into process memory.

The API describes tensor shape generically for any rank. The proof-of-concept UI only needs direct visualization for 1D and 2D tensors.

Tensor IDs are opaque URL-safe SHA-256 digests of UTF-8 logical tensor names,
stable across checkpoint relocation and independent of shard filenames. Logical
path segments come from splitting the name on dots. Inventory order is sorted
by the original tensor name; dimensions and C-order element order remain native.
Scalars have shape `[]` and one element, and any zero dimension gives zero
elements. Dimensions, products, offsets, and logical byte lengths must fit the
API safe-integer bounds before content allocation. Metadata/header input is
bounded to 100,000,000 bytes per file/header.

## Physical and logical representations

The backend knows the physical representation used by the model, including ordinary floating-point formats and future quantized formats such as INT8 or NF4.

The main visualization path exposes logical tensor values rather than requiring the UI to understand every physical quantization format. The initial canonical visualization representation is `float32`.

Initial accepted physical dtypes are safetensors `F32`, `F16`, and `BF16`.
Other encodings or a declared `quantization_config` fail explicitly as
unsupported representations in local discovery diagnostics. No integer-to-float
reinterpretation or quantization decoder is supplied by this catalogue.
Content access uses bounded file reads and PyTorch conversion into flat owned
CPU `float32` chunks (at most 262,144 elements). Later computation may transfer
these logical chunks to the configured device; this seam performs no GPU work.

A tensor already stored as compatible floating-point values may be streamed with the minimum necessary transformation. Quantized tensors are dequantized/materialized by the backend into logical `float32` values for visualization. A future raw-storage inspection mode may expose physical representations, but it is outside the proof-of-concept scope.

Materialized logical representations may be persisted as derived artifacts when doing so avoids repeating meaningful work.
