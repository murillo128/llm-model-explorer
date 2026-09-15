export interface CameraState {
  readonly scale: number;
  readonly x: number;
  readonly y: number;
}

/** Small per-source camera history. A gesture records only its starting camera. */
export class CameraHistory {
  private readonly states: CameraState[] = [];
  private gesture: object | undefined;

  constructor(private readonly limit = 32) {}

  record(before: CameraState, after: CameraState, gesture?: object) {
    if (before.scale === after.scale && before.x === after.x && before.y === after.y) return;
    if (!gesture || gesture !== this.gesture) {
      this.states.push(before);
      if (this.states.length > this.limit) this.states.shift();
    }
    this.gesture = gesture;
  }

  pop() {
    this.gesture = undefined;
    return this.states.pop();
  }

  clear() {
    this.states.length = 0;
    this.gesture = undefined;
  }
}
