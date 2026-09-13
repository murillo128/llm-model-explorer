import { TensorViewport } from './tensor-viewport';

const host = document.querySelector<HTMLElement>('#surface')!;
const progress = document.querySelector<HTMLOutputElement>('#progress')!;
try {
  const viewport = new TensorViewport(host, { shape: [512, 1536], rank: 2, numel: 512 * 1536, logical_dtype: 'float32' });
  const renderer = viewport.renderer;
  function receive() {
    const start = renderer.populatedPrefix;
    const length = Math.min(64 * 1536, renderer.geometry.count - start);
    const chunk = Float32Array.from({ length }, (_, i) => {
      const index = start + i;
      return Math.sin((index % 1536) / 31) * Math.cos(Math.floor(index / 1536) / 17);
    });
    renderer.upload(chunk);
    viewport.refresh();
    progress.textContent = `${renderer.populatedPrefix.toLocaleString()} / ${renderer.geometry.count.toLocaleString()} weights received · DPR ${window.devicePixelRatio}`;
  }
  function report(action: () => void) {
    try { action(); } catch (error) { progress.textContent = String(error); }
  }
  document.querySelector('#upload')!.addEventListener('click', () => report(receive));
  document.querySelector('#statistics')!.addEventListener('click', () => report(() => {
    renderer.setTransfer({ statistics: { minimum: -1, maximum: 1, percentiles: { p01: -0.95, p99: 0.95 } } });
    viewport.refresh();
  }));
  document.querySelector('#retry')!.addEventListener('click', () => report(() => { renderer.reconstruct(); viewport.refresh(); }));
  window.addEventListener('pagehide', () => viewport.dispose(), { once: true });
  receive();
} catch (error) {
  progress.textContent = String(error);
}
