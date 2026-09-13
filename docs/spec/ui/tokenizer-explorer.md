# Tokenizer Explorer

## Status

**Specification intentionally pending.**

The Tokenizer Explorer has already been the subject of dedicated design work outside this general architecture discussion. Its detailed behavior, layout, controls, interaction, and reusable UI components must be specified from that dedicated work rather than reconstructed here.

This document currently inherits only the cross-cutting constraints defined by the product, API, backend, and UI architecture specifications.

## General boundary

The proof of concept executes the real Hugging Face tokenizer associated with the session's model. The contract exposes the resulting tokens, IDs, offsets when available, decoded/text representation, and special-token information. Reconstructing or teaching the internal BPE/SentencePiece algorithm step by step is outside the current proof-of-concept scope.

Further requirements belong in the dedicated Tokenizer Explorer specification.
