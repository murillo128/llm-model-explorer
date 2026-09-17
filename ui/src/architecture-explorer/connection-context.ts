import { createContext } from 'react';
import type { Endpoint } from './projection';
import type { Point } from './graph';

export type EmphasisTarget = { edgeId: string } | { edgeIds: string[] } | { port: Endpoint };
export const ConnectionContext = createContext<{
  emphasized: Set<string>; ports: Set<string>; zoom: number;
  lineHit: (id: string, point: Point) => string[];
  hover: (target: EmphasisTarget | null) => void; focus: (target: EmphasisTarget | null) => void;
  selectPort?: (port: Endpoint) => void;
  pin: (id: string, trigger: HTMLElement) => void;
}>({ emphasized: new Set(), ports: new Set(), zoom: 1, lineHit: (id) => [id], hover: () => {}, focus: () => {}, pin: () => {} });

export const routePath = (points: Point[]) => points.map((p, i) => `${i ? 'L' : 'M'}${p.x},${p.y}`).join(' ');
