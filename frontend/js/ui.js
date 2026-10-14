// Small DOM toolkit. Every piece of data reaches the page through textContent / setAttribute — never innerHTML.

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k.startsWith('aria-') && typeof v === 'boolean') { el.setAttribute(k, String(v)); continue; }   // aria needs "true"/"false"
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'style' && typeof v === 'object') {
      for (const [sk, sv] of Object.entries(v)) sk.startsWith('--') ? el.style.setProperty(sk, sv) : (el.style[sk] = sv);
    }
    else if (k === 'dataset') Object.assign(el.dataset, v);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === 'value') el.value = v;
    else if (k === 'checked') el.checked = !!v;
    else if (k === 'disabled') el.disabled = !!v;
    else el.setAttribute(k, v === true ? '' : String(v));
  }
  append(el, kids);
  return el;
}

function append(el, kids) {
  for (const kid of kids.flat(Infinity)) {
    if (kid == null || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
}

export function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }
export function mount(el, ...kids) { clear(el); append(el, kids); return el; }

// ---- icons (own simple 24x24 stroke set)
const ICONS = {
  search: '<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/>',
  building: '<rect x="4" y="3" width="12" height="18" rx="1.5"/><path d="M16 9h4v12h-4M8 7h4M8 11h4M8 15h4"/>',
  grid: '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
  list: '<path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/>',
  layers: '<path d="M12 3l9 5-9 5-9-5 9-5zM3 13l9 5 9-5"/>',
  spark: '<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3zM19 16l.8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8L19 16z"/>',
  shield: '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6l8-3z"/><path d="M9 12l2 2 4-4"/>',
  download: '<path d="M12 4v11M7 11l5 5 5-5M4 20h16"/>',
  check: '<path d="M5 12l4 4 10-10"/>',
  x: '<path d="M6 6l12 12M18 6L6 18"/>',
  chevron: '<path d="M9 6l6 6-6 6"/>',
  external: '<path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  moon: '<path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z"/>',
  key: '<circle cx="8" cy="15" r="4"/><path d="M11 12l9-9M16 7l3 3"/>',
  refresh: '<path d="M20 11a8 8 0 0 0-14-4M4 5v4h4M4 13a8 8 0 0 0 14 4M20 19v-4h-4"/>',
  play: '<path d="M7 4l13 8-13 8V4z"/>',
  doc: '<path d="M7 3h7l5 5v13H7V3zM14 3v5h5"/>',
  alert: '<path d="M12 3l10 18H2L12 3zM12 10v4M12 17h.01"/>',
  lock: '<rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5h.01"/>',
  bolt: '<path d="M13 2L4 14h7l-1 8 9-12h-7l1-8z"/>',
  users: '<circle cx="9" cy="8" r="3.5"/><path d="M2.5 20c.6-3.6 3-5.5 6.5-5.5s5.9 1.9 6.5 5.5M16 4.5a3.5 3.5 0 0 1 0 7M18 14.8c2 .6 3.2 2.3 3.5 5.2"/>',
  truck: '<path d="M3 6h11v10H3zM14 9h4l3 3v4h-7"/><circle cx="7" cy="18" r="2"/><circle cx="17" cy="18" r="2"/>',
  target: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
  cart: '<path d="M3 4h2l2.5 11h10L20 7H6"/><circle cx="9" cy="19" r="1.5"/><circle cx="17" cy="19" r="1.5"/>',
  globe: '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c3 3 3 15 0 18M12 3c-3 3-3 15 0 18"/>',
  pin: '<path d="M12 21s7-6.2 7-11a7 7 0 0 0-14 0c0 4.8 7 11 7 11z"/><circle cx="12" cy="10" r="2.5"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  db: '<ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>',
};

export function icon(name, cls = '') {
  const s = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  s.setAttribute('viewBox', '0 0 24 24');
  s.setAttribute('class', `ico ${cls}`.trim());
  s.setAttribute('aria-hidden', 'true');
  s.innerHTML = ICONS[name] || ICONS.info; // static, trusted strings only
  return s;
}

// ---- formatting
export const fmtNum = (n) => (n == null ? '—' : Number(n).toLocaleString());
export function timeAgo(ts) {
  if (!ts) return '';
  const d = new Date(ts.endsWith('Z') || ts.includes('+') ? ts : ts + 'Z');
  const s = Math.max(0, (Date.now() - d.getTime()) / 1000);
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return d.toLocaleDateString();
}
export function safeUrl(u) {
  try { const x = new URL(u); return x.protocol === 'http:' || x.protocol === 'https:' ? x.href : null; } catch { return null; }
}
export function link(url, text) {
  const href = safeUrl(url);
  return href ? h('a', { href, target: '_blank', rel: 'noopener noreferrer' }, text || href) : h('span', {}, text || String(url || ''));
}
export function show(v) {
  if (v == null || v === '') return h('span', { class: 'muted' }, '—');
  if (Array.isArray(v)) return v.join(', ');
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}

// ---- sector colours: stable per sector name
const SECTOR_COLORS = 8;
export function sectorColor(name, ids) {
  const i = Math.max(0, ids.indexOf(name));
  return `var(--sector-${i % SECTOR_COLORS})`;
}

// ---- toasts
export function toast(message, kind = 'ok', ms = 4200) {
  const box = $('#toasts');
  const t = h('div', { class: `toast ${kind}`, role: 'status' }, icon(kind === 'bad' ? 'alert' : 'check'), message);
  box.append(t);
  setTimeout(() => t.remove(), ms);
}

export function debounce(fn, ms) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

export function download(name, data) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
  const a = h('a', { href: url, download: name });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1500);
}

export function safeStore(key, value) {
  try { if (value === undefined) return localStorage.getItem(key); localStorage.setItem(key, value); } catch { /* storage may be blocked */ }
  return null;
}
