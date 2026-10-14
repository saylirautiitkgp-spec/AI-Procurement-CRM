// Single shared state + a tiny change notifier. Views re-render when `emit()` is called.

import { safeStore } from './ui.js';

export const S = {
  status: null, overview: null, fields: null,
  view: 'companies',            // companies | insights | runs
  sector: null,                 // sector id or null (= all)
  q: '', country: '', enrichedOnly: false,
  sort: 'name', dir: 1,
  group: false, layout: 'cards', page: 1,
  open: null,                   // company id shown in the drawer
  collapsed: new Set(),
};

const listeners = new Set();
export const subscribe = (fn) => { listeners.add(fn); return () => listeners.delete(fn); };
export const emit = () => listeners.forEach((fn) => fn());

export function restorePrefs() {
  S.layout = safeStore('apt.layout') || S.layout;
  S.group = safeStore('apt.group') === '1';
  S.sort = safeStore('apt.sort') || S.sort;
}
export function savePrefs() {
  safeStore('apt.layout', S.layout); safeStore('apt.group', S.group ? '1' : '0'); safeStore('apt.sort', S.sort);
}

// ---- derived data
export const sectorNames = () => (S.overview ? S.overview.sectors.map((s) => s.name) : []);
const norm = (s) => String(s || '').toLowerCase();

export function matchesNonSector(c) {
  if (S.country && c.country !== S.country) return false;
  if (S.enrichedOnly && !(c.has_profile || c.insights)) return false;
  if (S.q) {
    const hay = `${norm(c.name)} ${norm(c.country)} ${norm(c.sector)} ${norm(c.type)}`;
    if (!S.q.split(/\s+/).every((t) => hay.includes(norm(t)))) return false;
  }
  return true;
}

export function filtered() {
  if (!S.overview) return [];
  const rows = S.overview.companies.filter((c) => matchesNonSector(c) && (S.sector == null || c.sector_id === S.sector));
  const k = S.sort;
  const val = {
    name: (c) => norm(c.name), sector: (c) => `${norm(c.sector)}|${norm(c.name)}`, country: (c) => `${norm(c.country)}|${norm(c.name)}`,
    enriched: (c) => -(c.profile_fields + c.insights * 2), priority: (c) => norm(c.priority),
  }[k] || ((c) => norm(c.name));
  return rows.sort((a, b) => (val(a) > val(b) ? 1 : val(a) < val(b) ? -1 : 0) * S.dir);
}

export function sectorCounts() {
  const counts = new Map();
  if (!S.overview) return counts;
  for (const c of S.overview.companies) if (matchesNonSector(c)) counts.set(c.sector_id, (counts.get(c.sector_id) || 0) + 1);
  return counts;
}
