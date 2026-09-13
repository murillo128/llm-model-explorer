# Tensor rendering

## Pixel mapping

The fundamental proof-of-concept rule is exact spatial mapping: **one tensor weight equals one rendered pixel**.

A 2D tensor of width `W` and height `H` therefore occupies `W × H` rendered pixels. The renderer must not fit the matrix to the viewport, resample it, aggregate weights, or use zoom as a substitute for the one-to-one mapping.

If the rendered dimensions exceed the available viewport, the containing UI uses normal horizontal and/or vertical scrolling.

If a tensor dimension exceeds a WebGL2 texture or render-target limit, the renderer may partition the representation internally into multiple textures/bands. This is an implementation detail and must preserve the visible one-weight-to-one-pixel mapping and the logical identity of one complete matrix.

## Numeric representation

The renderer consumes the logical visualization representation supplied by the backend. The initial representation is one `float32` logical value per weight regardless of the model's physical storage representation.

The tensor value remains authoritative. The renderer must not rewrite tensor values in order to change color, brightness, selection state, or another visual property.

## Luminosity

Weight value is represented through luminosity rather than semantic color.

The renderer maps tensor values to a normalized `[0, 1]` visual intensity using a configurable nonlinear, sigmoid-like transfer function. The default transfer uses robust tensor statistics such as percentiles rather than relying only on raw minimum and maximum values, so a small number of extreme values does not collapse most weights around the same mid-level intensity.

Tensor statistics are supplied independently by the backend and may arrive after tensor data has already started rendering. The renderer may use a provisional mapping and update the visual transfer function when the statistics become available without retransmitting or rewriting tensor values.

## Color

Color is reserved as an independent semantic channel. Future uses include selection, activation state, clusters, highlighting, and other overlays.

Applying or changing semantic color must not change the luminosity-derived underlying weight information and must not require modification of the stored tensor values.

## Interaction scope

The proof of concept does not require zoom or pan. Detailed Tensor Explorer interaction behavior is owned by `tensor-explorer.md` and must not be inferred from this common rendering specification.
