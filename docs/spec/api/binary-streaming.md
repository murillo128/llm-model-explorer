# Binary streaming protocol

## Purpose

Long operations return progressive binary responses so the UI can consume and display useful data before the complete result has been produced or transferred.

The protocol is common infrastructure for tensor transfer now and later for other large or progressive results such as activations and inference intermediates.

## Transport model

A long operation is initiated by an HTTP request and the same HTTP response carries its binary stream. WebSockets are not required for the proof of concept.

The browser must be able to consume the response incrementally through streaming Fetch APIs and cancel the consumer without waiting for the full body.

The protocol aims for zero-copy where the browser/runtime permits it and minimal-copy otherwise. It must avoid textual numeric serialization, Base64, and unnecessary application-owned intermediate buffers.

## Framing

The response is a sequence of framed records rather than an unstructured byte dump. Every frame has enough framing information to identify its kind and payload length, and the stream begins with an identifying marker sufficient to reject data that is not this protocol.

The proof of concept does not maintain multiple protocol versions or backwards-compatible negotiation. Backend and UI implement the current framing definition together.

The common semantic frame categories are:

- initial result metadata, describing what is being streamed and the expected logical payload;
- binary data frames containing contiguous bytes of the result;
- optional progress or auxiliary metadata frames when an operation needs them;
- successful completion;
- execution error;
- cancellation outcome where applicable.

The exact byte widths and field encoding of the frame header must be fixed in the machine-readable/API implementation contract before coding the parser and writer. They must remain deliberately small and must not wrap binary tensor data in JSON.

## Tensor semantics

A tensor request is logically for one complete tensor even though transport may divide its bytes across many data frames. Transport chunks are not independently addressable tensor tiles and are not persistent cache artifacts.

For the initial visualization path, tensor data frames carry the canonical logical `float32` representation in deterministic tensor order. The metadata frame carries at least the tensor identity, shape, representation, and total expected logical byte length needed for the UI to allocate and progressively fill its destination buffer.

The UI may begin rendering populated portions before the full tensor has arrived.

Tensor statistics are logically independent results and are not required to block the tensor stream.
