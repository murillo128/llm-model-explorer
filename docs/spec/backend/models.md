# Local models and tensors

## Supported model source

The product supports complete local Hugging Face checkpoint directories only. No generic provider abstraction, remote-ID onboarding, or automatic download is required. The backend receives one model root and never reveals its filesystem paths to the UI.

Discovery examines immediate child directories. A candidate needs a `config.json` object with a nonempty string `model_type`, plus Safetensors files or one `*.safetensors.index.json` with a nonempty `weight_map`. With an index, the referenced shard inventory must exactly match its mapping; without one, all immediate `*.safetensors` files form the checkpoint. Referenced shards may use subdirectories. Duplicate tensor names, missing shards, invalid headers, gaps/overlaps, and inconsistent physical byte lengths invalidate the candidate. Unrelated directories are ignored; invalid candidates are omitted and diagnosed in logs. An empty valid catalogue returns an empty list.

Shard references must be relative canonical POSIX paths without traversal, drive prefixes, or backslashes. Model directories, configuration, tokenizer assets, and shards may use symlinks only when they resolve within the configured root. Read regular files only; never execute checkpoint Python code.

Complete means all files of the chosen checkpoint variant, not every alternative format or quantization published in its repository. Metadata-only directories are not substitutes. For the selected V-JEPA 2 Transformers variant, do not also require the alternative `original` weights. Preserve local processor configuration as metadata without running it. Missing text-tokenizer assets do not invalidate a non-text model or its architecture; advertise only capabilities justified by available assets.

## Model identity

Public identity derives from HF metadata when suitable, with immediate directory-name fallback. Filesystem paths remain backend-private.

The identity fields are `config.json`'s `_name_or_path`, then `name_or_path`. An eligible value is a name or `organization/name`, with each segment starting in an ASCII letter, digit, or underscore and otherwise using letters, digits, underscores, dots, or hyphens. `..`, absolute paths, URIs, drive paths, and values identifying an existing path relative to the model directory or backend working directory are ineligible. Without an eligible value, use the immediate directory name. `model_type` and architecture class names are not model identities. An eligible single-segment `_commit_hash` (or `revision` when absent) is appended as `@revision`; display name excludes that suffix.

Identical copies with the same identity collapse to one entry. Equal identities with different content fingerprints fail catalogue listing with a path-free `validation_error`; never select one silently. Optional `parameter_count` is omitted because checkpoint entries may include buffers or omit tied parameters. `size_bytes`, when safe, counts selected local asset bytes. Unknown/unusable architecture names are represented by the required empty list. Tokenizer availability means conventional assets exist (`tokenizer.json`, `tokenizer.model`, `spiece.model`, `vocab.txt`, or `vocab.json`/`merges.txt`), not a guarantee every implementation can load them.

Each model has an internal content fingerprint. Relocation alone must not invalidate derived results; actual content changes must yield different artifact identities. The fingerprint is SHA-256 over a domain marker, sorted relative asset names, framed name lengths/file sizes, and streamed contents. Assets include immediate `.json`, `.model`, `.txt`, `.tiktoken`, and `.safetensors` files plus every indexed shard. Absolute paths, resolved targets, inode numbers, and timestamps do not enter the digest. Hash reads are bounded to 1 MiB. Layout/metadata changes can invalidate a fingerprint even when mathematical weights are unchanged.

Ordinary metadata listing reads configuration, indexes, and headers only. Full-file hashing is deferred to pinning, duplicate-identity comparison, and the accepted startup architecture preparation. Preparation must establish the same content identity; a config-only or stat-only digest is not an allowed shortcut for graph cache validity.

Before/after hashing, inventory access, guarded tokenizer work, and each tensor chunk, check the asset set, resolved targets, device/inode, size, modification time, and change time. Replacement, removal, addition, or mutation invalidates a pinned source with `model_content_changed`; never silently refresh a session. Explicit rehash remains available. This assumes ordinary filesystem change tracking in the trusted deployment environment; no full rehash is required per chunk. Exhaust iterators and pass final checks before declaring successful generation/publication.

## Loading strategy

Opening a model must not load the entire checkpoint into RAM or GPU. Discover metadata and structure first, and access values lazily. Use memory mapping or equivalent bounded reads where the format permits. Startup graph preparation does not change this rule: sequential integrity hashing is permitted, resident full-weight materialization is not.

Tensor shapes remain generic for arbitrary rank; direct UI visualization is limited to complete 1D/2D tensors. Tensor IDs are opaque URL-safe SHA-256 digests of UTF-8 logical names, stable across relocation and independent of shards. Logical paths split names on dots; native inventory order is sorted by original name. Scalars have shape `[]` and one element; a zero dimension means zero elements. Dimensions, products, offsets, and logical lengths must fit API safe-integer limits before allocation. Metadata/header input remains bounded to 100,000,000 bytes per file/header.

## Physical and logical representations

The backend owns physical formats. The main numeric visualization representation remains canonical `float32`; the UI never decodes quantization.

The baseline native materialization path supports Safetensors `F32`, `F16`, and `BF16`. It uses bounded file reads and PyTorch conversion to flat owned CPU float32 chunks (at most 262,144 elements), performing no GPU work in this seam. Existing native descriptors, IDs, element order, values, tokenization, embeddings, and byte streaming must remain unchanged.

The accepted architecture increment extends *admission and structural interpretation* to the selected GPTQ Int4 and NVFP4 checkpoint layouts listed in [product.md](../product.md#architecture-reference-checkpoints-and-bounded-coverage). It supersedes blanket rejection solely because `quantization_config` exists or a selected packed dtype differs from the three native types. It does not promise support for every quantization with those names. Validate actual configuration, physical dtype widths, physical shapes/offsets, complete shard coverage, and encoding-specific metadata before admitting a variant. Unknown encodings remain explicitly unsupported; never relax header, path, size, or integrity validation.

Keep a complete physical inventory internally, separate from logical parameters and their numeric availability. A logical parameter can use packed data, scales, an alias, or a fused region; physical byte counts must be checked against physical storage geometry, never against guessed dequantized dimensions. Graph bindings and provenance are governed by [architecture-analysis.md](architecture-analysis.md#parameter-binding-and-provenance).

For the newly admitted quantized variants, expose in the existing logical tensor inventory only complete native mathematical tensors with a verified supported data path. Mark this inventory partial through the [API extension](../api/architecture-explorer.md#quantized-checkpoint-capability-separation); describe excluded packed/unresolved parameters in the graph. Do not expose auxiliary scales or packed arrays as substitute logical weight matrices. A parameter without an existing actionable numeric tensor identity receives metadata and an unavailable reason, not a fabricated ID. Baseline unquantized inventories are not filtered or renamed.

Architecture completeness is independent from numeric inspection. Native 1D/2D weights remain inspectable where verified. Higher-rank native weights retain exact metadata and existing generic tensor-data behavior, but the architecture modal does not flatten or slice them. Decoder-dependent or region-dependent views remain unavailable in this increment. No new quantization decoder, decompressed checkpoint copy, raw-storage explorer, or volume viewer is required.

Where a logical representation is supported, numeric materialization remains backend-owned and may be persisted as a derived artifact. The architecture view must never feed its descriptive graph into embedding lookup as if that were an executable or numerically supported model.
