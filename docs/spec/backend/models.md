# Local models and tensors

## Supported model source

The proof of concept supports local Hugging Face model directories only. No generic provider abstraction is required for other formats.

The backend receives one local model root through configuration. Models are discovered beneath that root and exposed to the API without revealing local filesystem paths to the UI.

## Model identity

The public model identity should be derived from Hugging Face metadata when sufficient metadata is available. The local directory name is only a fallback identity when the model does not provide a suitable logical identity.

Filesystem paths are private backend implementation details.

Each discovered model also has a content fingerprint used for cache validity. Moving an unchanged model to another local path must not invalidate artifacts merely because its path changed, while changing the actual model content must result in a different fingerprint and therefore different artifact keys.

## Loading strategy

Opening a model must not load the complete model into RAM or GPU memory. Metadata and tensor structure are discovered first; tensor content is accessed lazily and materialized only when requested.

Where the Hugging Face storage format allows it, tensor access should use memory mapping or equivalent lazy file access rather than copying the complete weight files into process memory.

The API describes tensor shape generically for any rank. The proof-of-concept UI only needs direct visualization for 1D and 2D tensors.

## Physical and logical representations

The backend knows the physical representation used by the model, including ordinary floating-point formats and future quantized formats such as INT8 or NF4.

The main visualization path exposes logical tensor values rather than requiring the UI to understand every physical quantization format. The initial canonical visualization representation is `float32`.

A tensor already stored as compatible floating-point values may be streamed with the minimum necessary transformation. Quantized tensors are dequantized/materialized by the backend into logical `float32` values for visualization. A future raw-storage inspection mode may expose physical representations, but it is outside the proof-of-concept scope.

Materialized logical representations may be persisted as derived artifacts when doing so avoids repeating meaningful work.
