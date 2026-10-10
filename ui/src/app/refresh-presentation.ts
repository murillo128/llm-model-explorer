import type { Lifetime } from './lifetime';

import type { ArchitectureBookmark } from '../architecture-explorer/navigation-bookmark';
export type { ArchitectureBookmark } from '../architecture-explorer/navigation-bookmark';

export interface RefreshKey { backend: string; modelId: string; attempt: number }
export class RefreshPresentation {
  private capture: (() => ArchitectureBookmark) | undefined;
  private pending: { key: RefreshKey; bookmark: ArchitectureBookmark } | undefined;
  register(view: Lifetime, capture: () => ArchitectureBookmark) {
    this.capture = capture;
    const clear = () => { if (this.capture === capture) this.capture = undefined; };
    const detach = view.onDispose(clear);
    return () => { detach(); clear(); };
  }
  save(key: RefreshKey) {
    try {
      const bookmark = this.capture?.() ?? (this.pending?.key.backend === key.backend && this.pending.key.modelId === key.modelId ? this.pending.bookmark : undefined);
      this.pending = bookmark && JSON.stringify(bookmark).length <= 65_536
        ? { key, bookmark: structuredClone(bookmark) } : undefined;
    } catch { this.pending = undefined; }
  }
  consume(key: RefreshKey): ArchitectureBookmark | undefined {
    const pending = this.pending;
    if (!pending || pending.key.backend !== key.backend || pending.key.modelId !== key.modelId || pending.key.attempt !== key.attempt) return;
    this.pending = undefined;
    return pending.bookmark;
  }
  clear() { this.pending = undefined; this.capture = undefined; }
}
