import './style.css';

const app = document.querySelector<HTMLDivElement>('#app');

if (!app) {
  throw new Error('Missing #app root element');
}

const canvas = document.createElement('canvas');
canvas.className = 'gpu-canvas';
canvas.setAttribute('aria-label', 'LLM tensor visualization canvas');

const gl = canvas.getContext('webgl2', {
  alpha: false,
  antialias: false,
  depth: false,
  stencil: false,
  preserveDrawingBuffer: false,
  powerPreference: 'high-performance',
});

app.innerHTML = `
  <main class="shell">
    <header class="hero">
      <p class="eyebrow">HuggingFaceTB/SmolLM2-135M · Base</p>
      <h1>LLM Model Explorer</h1>
      <p class="intro">A GPU-first browser explorer for tensors, weights, activations and matrix operations.</p>
    </header>
    <section class="status" aria-live="polite">
      <span class="status-dot"></span>
      <span>${gl ? 'WebGL2 ready' : 'WebGL2 unavailable'}</span>
    </section>
    <section class="viewport" id="viewport"></section>
  </main>
`;

const viewport = document.querySelector<HTMLElement>('#viewport');
if (!viewport) {
  throw new Error('Missing viewport element');
}

viewport.append(canvas);

function resizeCanvas(): void {
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(viewport.clientWidth * ratio));
  const height = Math.max(1, Math.floor(viewport.clientHeight * ratio));

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  if (gl) {
    gl.viewport(0, 0, width, height);
    gl.clearColor(0.055, 0.067, 0.082, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
  }
}

const observer = new ResizeObserver(resizeCanvas);
observer.observe(viewport);
resizeCanvas();
