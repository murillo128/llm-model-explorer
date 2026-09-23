// Optional browser preferences must never prevent scientific inspection.
const memory = new Map<string, unknown>();
export function readInventoryPreference(key: string): unknown {
  let saved: string | null;
  try { saved = localStorage.getItem(key); } catch { return memory.get(key) ?? null; }
  if (saved === null) { memory.delete(key); return null; }
  try { return JSON.parse(saved); } catch { return null; }
}

export function writeInventoryPreference(key: string, value: unknown) {
  memory.set(key, value);
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* Keep the preference in memory. */ }
}
