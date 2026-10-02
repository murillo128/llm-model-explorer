/** Consecutive wheel/trackpad events share a gesture independently of rendering. */
export class WheelGesture {
  private current: { token: object; time: number } | null = null;

  at(time: number): object {
    if (!this.current || time - this.current.time > 180) this.current = { token: {}, time };
    this.current.time = time;
    return this.current.token;
  }

  reset() { this.current = null; }
}
