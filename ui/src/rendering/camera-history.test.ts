import { describe, expect, it } from 'vitest';
import { CameraHistory } from './camera-history';

const camera = (scale: number, x = 0, y = 0) => ({ scale, x, y });

describe('local camera history', () => {
  it('restores origins and scale across distinct region/range commits', () => {
    const history = new CameraHistory();
    history.record(camera(2, 4, 8), camera(6, 12, 10));
    history.record(camera(6, 12, 10), camera(12, 30, 15));
    expect(history.pop()).toEqual(camera(6, 12, 10));
    expect(history.pop()).toEqual(camera(2, 4, 8));
    expect(history.pop()).toBeUndefined();
  });

  it('coalesces one gesture and starts a fresh entry after zoom-back', () => {
    const history = new CameraHistory(), wheel = {}, pinch = {};
    history.record(camera(1), camera(2), wheel);
    history.record(camera(2), camera(3), wheel);
    history.record(camera(3), camera(4), pinch);
    history.record(camera(4), camera(5), pinch);
    expect(history.pop()).toEqual(camera(3));
    history.record(camera(3), camera(6), pinch);
    expect(history.pop()).toEqual(camera(3));
    expect(history.pop()).toEqual(camera(1));
    expect(history.pop()).toBeUndefined();
  });

  it('ignores unchanged cameras, bounds retained state and clears on fit/source reset', () => {
    const history = new CameraHistory();
    history.record(camera(1), camera(1));
    expect(history.pop()).toBeUndefined();
    for (let i = 1; i <= 40; i++) history.record(camera(i), camera(i + 1));
    const scales = Array.from({ length: 32 }, () => history.pop()!.scale);
    expect(scales).toEqual(Array.from({ length: 32 }, (_, i) => 40 - i));
    expect(history.pop()).toBeUndefined();
    history.record(camera(2), camera(3));
    history.clear();
    expect(history.pop()).toBeUndefined();
    expect(new CameraHistory().pop()).toBeUndefined();
  });
});
