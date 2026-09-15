import type { MatrixViewport } from './matrix-viewport';

/** Local shortcuts run after selection and focused modal/popover dismissal. */
export class MatrixCameraNavigation {
  private hovered = false;

  constructor(private readonly viewport: MatrixViewport) {
    viewport.host.addEventListener('pointerover', this.over);
    viewport.host.addEventListener('pointerout', this.out);
    viewport.host.addEventListener('contextmenu', this.context);
    window.addEventListener('keydown', this.key);
  }

  private isSurface(target: EventTarget | null) {
    return [this.viewport.matrix.renderer, this.viewport.rows, this.viewport.columns]
      .some((renderer) => renderer?.state === 'ready' && renderer.canvas === target);
  }

  private over = (event: PointerEvent) => { this.hovered = this.isSurface(event.target); };
  private out = (event: PointerEvent) => { this.hovered = this.isSurface(event.relatedTarget); };
  private context = (event: MouseEvent) => {
    if (!this.isSurface(event.target)) return;
    event.preventDefault();
    event.stopPropagation();
    this.viewport.matrix.zoomBack();
  };
  private key = (event: KeyboardEvent) => {
    if (event.key !== 'Escape' || event.defaultPrevented) return;
    const focus = document.activeElement;
    const focused = focus !== null && this.viewport.host.contains(focus);
    const neutralFocus = focus === document.body || focus === document.documentElement || !focus;
    if (!focused && !(neutralFocus && this.hovered)) return;
    if (!this.viewport.matrix.zoomBack()) return;
    event.preventDefault();
    event.stopImmediatePropagation();
  };

  dispose() {
    this.viewport.host.removeEventListener('pointerover', this.over);
    this.viewport.host.removeEventListener('pointerout', this.out);
    this.viewport.host.removeEventListener('contextmenu', this.context);
    window.removeEventListener('keydown', this.key);
  }
}
