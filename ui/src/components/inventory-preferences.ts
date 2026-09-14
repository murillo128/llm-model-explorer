// Optional browser preferences must never prevent scientific inspection.
export function readInventoryPreference(key: string): unknown {
  try { return JSON.parse(localStorage.getItem(key) ?? 'null'); } catch { return null; }
}

export function writeInventoryPreference(key: string, value: unknown) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* Keep the preference in memory. */ }
}
