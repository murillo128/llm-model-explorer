import { describe, expect, it } from 'vitest';
import { CameraHistory, type CameraState } from './camera-history';
import { WheelGesture } from './wheel-gesture';

// Independently authored camera transitions; rendering/focal math owns other tests.
const cameras: CameraState[] = [
  { scale: 2, x: 4, y: 8 }, { scale: 3, x: 6, y: 10 },
  { scale: 4, x: 9, y: 12 }, { scale: 5, x: 11, y: 14 },
];

describe('runtime wheel/trackpad temporal history', () => {
  it.each([
    { name: '179 ms gaps coalesce across 358 ms', times: [0, 179, 358], back: [cameras[0], undefined, undefined] },
    { name: '180 ms gaps coalesce across 360 ms', times: [0, 180, 360], back: [cameras[0], undefined, undefined] },
    { name: '181 ms gap splits after the second event', times: [0, 40, 221], back: [cameras[2], cameras[0], undefined] },
  ])('$name', ({ times, back }) => {
    const gesture = new WheelGesture(), history = new CameraHistory();
    times.forEach((time, index) => history.record(cameras[index]!, cameras[index + 1]!, gesture.at(time)));
    expect([history.pop(), history.pop(), history.pop()]).toEqual(back);
  });

  it('wheel then ctrl-wheel bursts have separate starting cameras after a 220 ms gap', () => {
    // ctrl-wheel uses the same temporal policy; the gap, not the modifier, splits it.
    const gesture = new WheelGesture(), history = new CameraHistory();
    history.record(cameras[0]!, cameras[1]!, gesture.at(0));
    history.record(cameras[1]!, cameras[2]!, gesture.at(40));
    history.record(cameras[2]!, cameras[3]!, gesture.at(80));
    history.record(cameras[3]!, { scale: 6, x: 13, y: 16 }, gesture.at(300));
    history.record({ scale: 6, x: 13, y: 16 }, { scale: 7, x: 15, y: 18 }, gesture.at(340));
    expect([history.pop(), history.pop(), history.pop()]).toEqual([cameras[3], cameras[0], undefined]);
  });

  it('interruptions start a fresh gesture and fit/source reset clears temporal and camera history', () => {
    const gesture = new WheelGesture(), history = new CameraHistory();
    history.record(cameras[0]!, cameras[1]!, gesture.at(0));
    expect(history.pop()).toEqual(cameras[0]);
    gesture.reset();
    history.record(cameras[0]!, cameras[2]!, gesture.at(40));
    // A direct region commit interrupts the wheel gesture even inside 180 ms.
    gesture.reset();
    history.record(cameras[2]!, cameras[3]!);
    history.record(cameras[3]!, { scale: 6, x: 13, y: 16 }, gesture.at(80));
    expect([history.pop(), history.pop(), history.pop()]).toEqual([cameras[3], cameras[2], cameras[0]]);
    const beforeReset = gesture.at(100);
    history.record(cameras[0]!, cameras[1]!, beforeReset);
    history.clear(); gesture.reset();
    expect(history.pop()).toBeUndefined();
    expect(gesture.at(140)).not.toBe(beforeReset);
    expect(new CameraHistory().pop()).toBeUndefined();
  });
});
