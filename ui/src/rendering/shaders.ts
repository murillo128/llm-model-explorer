import { amber } from './chroma';

export const vertexShader = `#version 300 es
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

// Shared by scalar and count shaders, including the inspection framebuffer.
const selectionShader = `
uniform ivec2 bandOrigin;
uniform ivec2 selection;
uniform vec2 guideOpacity;
ivec2 sampledCell() {
  vec2 pixel = vec2(floor(gl_FragCoord.x), float(viewHeight - 1) - floor(gl_FragCoord.y));
  vec2 cell = floor((pixel + cameraOffset) / cellScale);
  // Correct division at exact boundaries using the same float32 raster edges
  // as CPU hit testing and band scissors. Origins retain integer cell identity.
  cell += vec2(greaterThanEqual(pixel, ceil((cell + 1.0) * cellScale - cameraOffset)));
  cell -= vec2(lessThan(pixel, ceil(cell * cellScale - cameraOffset)));
  return ivec2(cell) + cellOrigin - bandOrigin;
}
vec3 semanticColor(float y, ivec2 cell) {
  vec2 pixel = vec2(floor(gl_FragCoord.x), float(viewHeight - 1) - floor(gl_FragCoord.y));
  vec2 selected = vec2(selection - cellOrigin);
  vec2 start = ceil(selected * cellScale - cameraOffset);
  vec2 end = ceil((selected + 1.0) * cellScale - cameraOffset);
  // Center within the sampled pixel interval, including near-native fractional
  // scales where rounding the logical center can land in the previous cell.
  vec2 guide = floor((start + end) * 0.5);
  bool row = selection.y >= 0 && pixel.y == guide.y;
  bool column = selection.x >= 0 && pixel.x == guide.x;
  float opacity = row && column ? guideOpacity.y : row || column ? guideOpacity.x : 0.0;
  vec3 green = vec3(0.001 + 0.819 * y * y * y,
    0.006 + 0.994 * pow(y, 1.5), 0.002 + 0.858 * y * y * y);
  // One-device-pixel guides blend over the unchanged scalar display transfer.
  vec3 linear = mix(green, vec3(${amber.join(', ')}), opacity);
  // RGBA8 canvas storage is display encoded; WebGL performs no extra encoding.
  return mix(1.055 * pow(max(linear, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055,
    12.92 * linear, lessThanEqual(linear, vec3(0.0031308)));
}`;

export const fragmentShader = `#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D weights;
uniform vec2 cameraOffset;
uniform ivec2 cellOrigin;
uniform vec2 cellScale;
uniform int viewHeight;
uniform ivec2 prefix;
uniform int mode;
uniform float slope;
uniform vec2 anchors;
uniform float scale;
uniform float span;
uniform float correction;
${selectionShader}
out vec4 color;

float logistic(float v) { return 1.0 / (1.0 + exp(-v)); }
void main() {
  ivec2 cell = sampledCell();
  if (cell.y > prefix.y || (cell.y == prefix.y && cell.x >= prefix.x)) {
    color = vec4(0.18, 0.24, 0.30, 1.0); // pending, visibly distinct from a zero weight
    return;
  }
  float w = texelFetch(weights, cell, 0).r;
  if (isnan(w) || isinf(w)) {
    color = vec4(0.70, 0.20, 0.55, 1.0); // received, nonfinite
    return;
  }
  float intensity = 0.5;
  if (mode == 0) {
    // Provisional centered logistic; clamp before multiplying to avoid overflow.
    intensity = logistic(clamp(w, -80.0 / slope, 80.0 / slope) * slope);
  } else if (mode == 1) {
    float u;
    if (w < anchors.x) u = 0.0;
    else if (w > anchors.y) u = 1.0;
    else {
      // Same-sign subtraction preserves close values; opposite signs scale
      // first to avoid overflow at the finite float32 extremes.
      float delta = (w < 0.0) != (anchors.x < 0.0)
        ? w / scale - anchors.x / scale : (w - anchors.x) / scale;
      u = clamp((delta + correction) / span, 0.0, 1.0);
    }
    float endpoint = logistic(-0.5 * slope);
    intensity = (logistic(slope * (u - 0.5)) - endpoint) / (1.0 - 2.0 * endpoint);
  }
  color = vec4(semanticColor(clamp(intensity, 0.0, 1.0), cell), 1.0);
}`;

export const distributionFragmentShader = `#version 300 es
precision highp float;
precision highp int;
uniform highp usampler2D weights;
uniform vec2 cameraOffset;
uniform ivec2 cellOrigin;
uniform vec2 cellScale;
uniform int viewHeight;
uniform ivec2 prefix;
uniform float densityDenominator;
${selectionShader}
out vec4 color;
void main() {
  ivec2 cell = sampledCell();
  if (cell.y > prefix.y || (cell.y == prefix.y && cell.x >= prefix.x)) {
    color = vec4(0.18, 0.24, 0.30, 1.0);
    return;
  }
  uint count = texelFetch(weights, cell, 0).r;
  float intensity = densityDenominator > 0.0 ? log(1.0 + float(count)) / densityDenominator : 0.0;
  color = vec4(semanticColor(clamp(intensity, 0.0, 1.0), cell), 1.0);
}`;
