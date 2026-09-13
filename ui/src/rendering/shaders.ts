export const vertexShader = `#version 300 es
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

export const fragmentShader = `#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D weights;
uniform ivec2 bandOffset;
uniform int viewHeight;
uniform ivec2 prefix;
uniform int mode;
uniform float slope;
uniform vec2 anchors;
uniform float scale;
uniform float span;
uniform float correction;
out vec4 color;

float logistic(float v) { return 1.0 / (1.0 + exp(-v)); }
void main() {
  ivec2 cell = ivec2(int(gl_FragCoord.x), viewHeight - 1 - int(gl_FragCoord.y)) + bandOffset;
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
  color = vec4(vec3(intensity), 1.0);
}`;

export const distributionFragmentShader = `#version 300 es
precision highp float;
precision highp int;
uniform highp usampler2D weights;
uniform ivec2 bandOffset;
uniform int viewHeight;
uniform ivec2 prefix;
uniform float densityDenominator;
out vec4 color;
void main() {
  ivec2 cell = ivec2(int(gl_FragCoord.x), viewHeight - 1 - int(gl_FragCoord.y)) + bandOffset;
  if (cell.y > prefix.y || (cell.y == prefix.y && cell.x >= prefix.x)) {
    color = vec4(0.18, 0.24, 0.30, 1.0);
    return;
  }
  uint count = texelFetch(weights, cell, 0).r;
  float intensity = densityDenominator > 0.0 ? log(1.0 + float(count)) / densityDenominator : 0.0;
  color = vec4(vec3(clamp(intensity, 0.0, 1.0)), 1.0);
}`;
