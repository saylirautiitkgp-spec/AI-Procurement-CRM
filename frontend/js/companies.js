// Sector sidebar, the Companies view (cards / table / grouped by sector) and the Insights view.

import { h, mount, icon, fmtNum, sectorColor } from './ui.js';
import { S, emit, filtered, sectorCounts, sectorNames, savePrefs } from './state.js';
import { openDrawer } from './drawer.js';

const PAGE = 48;
const colorOf = (c) => sectorColor(c.sector, sectorNames());

// ------------------------------------------------------------------------------------------------ sidebar
export function createSidebar() {
  const list = h('div', { class: 'sector-list', role: 'list' });
  const el = h('aside', { class: 'sidebar', 'aria-label': 'Sectors' },
    h('div', { class: 'side-title' }, h('span', {}, 'Sectors'), h('span', { id: 'side-total' }, '')),
    list,
    h('div', { class: 'side-card' },
      h('h4', {}, 'How to use'),
      h('p', {}, 'Pick a sector to focus the list, search by name or country, then open a company and choose exactly which information you want researched.')));
  function update() {
    if (!S.overview) return;
    const counts = sectorCounts();
    const total = [...counts.values()].reduce((a, b) => a + b, 0);
    const max = Math.max(1, ...counts.values());
    el.querySelector('#side-total').textContent = fmtNum(total);
    const names = sectorNames();
    const rows = [h('button', { class: 'sector', 'aria-pressed': S.sector == null, style: { '--c': 'var(--ink-3)' }, onclick: () => { S.sector = null; S.page = 1; emit(); } },
      h('i', { class: 'dot' }), h('span', { class: 'nm' }, 'All sectors'), h('span', { class: 'ct' }, fmtNum(total)))];
    for (const s of S.overview.sectors) {
      const n = counts.get(s.id) || 0;
      rows.push(h('button', {
        class: 'sector', 'aria-pressed': S.sector === s.id, title: `${s.name}: ${n} companies`,
        style: { '--c': sectorColor(s.name, names), '--w': `${Math.round((n / max) * 100)}%` },
        onclick: () => { S.sector = S.sector === s.id ? null : s.id; S.page = 1; emit(); },
      }, h('i', { class: 'dot' }), h('span', { class: 'nm' }, s.name), h('span', { class: 'ct' }, fmtNum(n)), h('i', { class: 'bar' })));
    }
    mount(list, rows);
  }
  return { el, update };
}

// ------------------------------------------------------------------------------------------------ cards / table
function companyCard(c) {
  const color = colorOf(c);
  const pct = Math.round((100 * c.profile_fields) / (c.profile_fields_total || 1));
  return h('button', { class: 'card', style: { '--c': color }, onclick: () => openDrawer(c.id), 'aria-label': `Open ${c.name}` },
    h('div', { class: 'top' }, h('h3', {}, c.name), h('span', { class: 'chip', style: { '--c': color } }, c.sector || 'No sector')),
    h('div', { class: 'meta' },
      c.country && h('span', {}, icon('pin', 'sm'), c.country),
      c.type && h('span', {}, icon('building', 'sm'), c.type),
      c.priority && h('span', {}, icon('target', 'sm'), c.priority)),
    h('div', { class: 'meter' },
      h('div', { class: 'lbl' }, h('span', {}, 'Profile completeness'), h('span', {}, `${c.profile_fields}/${c.profile_fields_total}`)),
      h('div', { class: 'track' }, h('div', { class: 'fill', style: { '--w': `${pct}%`, '--c': color } }))),
    h('div', { class: 'foot' },
      c.has_profile ? h('span', { class: 'chip ok' }, icon('check', 'sm'), 'Profile') : h('span', { class: 'chip plain' }, 'No profile yet'),
      c.insights ? h('span', { class: 'chip blue' }, icon('shield', 'sm'), `${c.insights} sourced`) : null,
      c.website ? h('span', { class: 'chip plain' }, icon('globe', 'sm'), 'Website') : null));
}

function companyTable(rows) {
  const cols = [['name', 'Company'], ['sector', 'Sector'], ['country', 'Country'], [null, 'Type'], ['priority', 'Priority'], ['enriched', 'Profile'], [null, 'Sourced']];
  const head = h('tr', {}, cols.map(([key, label]) => h('th', {
    scope: 'col', 'aria-sort': key && S.sort === key ? (S.dir > 0 ? 'ascending' : 'descending') : 'none',
    onclick: key ? () => { if (S.sort === key) S.dir *= -1; else { S.sort = key; S.dir = 1; } savePrefs(); emit(); } : null,
  }, label, key && S.sort === key ? (S.dir > 0 ? ' ▲' : ' ▼') : '')));
  const body = rows.map((c) => h('tr', { class: 'row', tabindex: 0, onclick: () => openDrawer(c.id), onkeydown: (e) => e.key === 'Enter' && openDrawer(c.id) },
    h('td', {}, h('b', {}, c.name)),
    h('td', {}, h('span', { class: 'chip', style: { '--c': colorOf(c) } }, c.sector || '—')),
    h('td', {}, c.country || '—'), h('td', {}, c.type || '—'), h('td', {}, c.priority || '—'),
    h('td', {}, h('div', { class: 'meter', style: { minWidth: '120px' } },
      h('div', { class: 'track' }, h('div', { class: 'fill', style: { '--w': `${Math.round((100 * c.profile_fields) / (c.profile_fields_total || 1))}%`, '--c': colorOf(c) } })))),
    h('td', {}, c.insights ? h('span', { class: 'chip blue' }, c.insights) : h('span', { class: 'muted' }, '—'))));
  return h('table', { class: 'tbl' }, h('thead', {}, head), h('tbody', {}, body));
}

function groupedView(rows) {
  const by = new Map();
  for (const c of rows) { const k = c.sector || 'No sector'; if (!by.has(k)) by.set(k, []); by.get(k).push(c); }
  const names = sectorNames();
  const groups = [...by.entries()].sort((a, b) => b[1].length - a[1].length || a[0].localeCompare(b[0]));
  return groups.map(([name, items]) => {
    const enriched = items.filter((c) => c.has_profile || c.insights).length;
    const d = h('details', { class: 'group', open: !S.collapsed.has(name) ? '' : null, style: { '--c': sectorColor(name, names) },
      ontoggle: (e) => (e.target.open ? S.collapsed.delete(name) : S.collapsed.add(name)) },
      h('summary', {}, icon('chevron', 'caret'), h('i', { class: 'dot' }), h('span', {}, name),
        h('span', { class: 'sub' }, h('span', {}, `${enriched}/${items.length} researched`), h('span', { class: 'chip', style: { '--c': sectorColor(name, names) } }, items.length))),
      h('div', { class: 'inner' }, S.layout === 'table' ? companyTable(items) : h('div', { class: 'grid' }, items.map(companyCard))));
    return d;
  });
}

// ------------------------------------------------------------------------------------------------ Companies view
export function createCompaniesView() {
  let countriesBuilt = false;
  const search = h('input', { class: 'input with-ico', type: 'search', id: 'search', placeholder: 'Search companies, countries, sectors…',
    'aria-label': 'Search companies', autocomplete: 'off', oninput: (e) => { S.q = e.target.value.trim(); S.page = 1; emit(); } });
  const sort = h('select', { class: 'select', 'aria-label': 'Sort by', onchange: (e) => { S.sort = e.target.value; S.dir = 1; savePrefs(); emit(); } },
    [['name', 'Sort: Name'], ['sector', 'Sort: Sector'], ['country', 'Sort: Country'], ['enriched', 'Sort: Most researched'], ['priority', 'Sort: Priority']]
      .map(([v, l]) => h('option', { value: v }, l)));
  const dirBtn = h('button', { class: 'btn sm', title: 'Reverse the order', 'aria-label': 'Reverse sort order', onclick: () => { S.dir *= -1; emit(); } }, '⇅');
  const country = h('select', { class: 'select', 'aria-label': 'Country', onchange: (e) => { S.country = e.target.value; S.page = 1; emit(); } }, h('option', { value: '' }, 'All countries'));
  const enriched = h('input', { type: 'checkbox', onchange: (e) => { S.enrichedOnly = e.target.checked; S.page = 1; emit(); } });
  const group = h('button', { 'aria-pressed': 'false', onclick: () => { S.group = !S.group; savePrefs(); emit(); } }, icon('layers', 'sm'), 'Group by sector');
  const cardsBtn = h('button', { 'aria-pressed': 'true', onclick: () => { S.layout = 'cards'; savePrefs(); emit(); } }, icon('grid', 'sm'), 'Cards');
  const tableBtn = h('button', { 'aria-pressed': 'false', onclick: () => { S.layout = 'table'; savePrefs(); emit(); } }, icon('list', 'sm'), 'Table');
  const reset = h('button', { class: 'btn sm ghost', onclick: () => { Object.assign(S, { q: '', country: '', enrichedOnly: false, sector: null, page: 1 }); search.value = ''; emit(); } }, 'Reset filters');
  const count = h('span', { class: 'count-note' }, '');
  const results = h('div', { id: 'results' });
  const title = h('h1', {}, 'Companies');
  const sub = h('p', {}, '');
  const el = h('section', { 'aria-label': 'Companies' },
    h('div', { class: 'view-head' }, h('div', {}, title, sub)),
    h('div', { class: 'toolbar', role: 'search' },
      h('div', { class: 'field search' }, icon('search'), search, h('kbd', {}, '/')),
      sort, dirBtn, country,
      h('label', { class: 'toggle' }, enriched, 'Researched only'),
      h('div', { class: 'seg', role: 'group', 'aria-label': 'Layout' }, cardsBtn, tableBtn),
      h('div', { class: 'seg' }, group), reset, count),
    results);

  function update() {
    if (!S.overview) return;
    if (!countriesBuilt) {
      const cs = [...new Set(S.overview.companies.map((c) => c.country).filter(Boolean))].sort();
      cs.forEach((c) => country.append(h('option', { value: c }, c)));
      countriesBuilt = true;
    }
    sort.value = S.sort; country.value = S.country; enriched.checked = S.enrichedOnly;
    group.setAttribute('aria-pressed', S.group); cardsBtn.setAttribute('aria-pressed', S.layout === 'cards'); tableBtn.setAttribute('aria-pressed', S.layout === 'table');
    if (document.activeElement !== search && search.value !== S.q) search.value = S.q;
    const rows = filtered();
    const sec = S.sector != null ? S.overview.sectors.find((s) => s.id === S.sector) : null;
    title.textContent = sec ? sec.name : 'All companies';
    sub.textContent = sec ? `${fmtNum(sec.companies)} companies in this sector · ${fmtNum(sec.enriched)} already researched` :
      `${fmtNum(S.overview.totals.companies)} companies across ${S.overview.totals.sectors} sectors`;
    count.textContent = `${fmtNum(rows.length)} shown`;
    if (!rows.length) {
      mount(results, h('div', { class: 'empty' }, icon('search', 'lg'), h('h3', {}, 'No companies match'), h('p', {}, 'Try a different search or clear a filter.'),
        h('button', { class: 'btn', onclick: () => reset.click() }, 'Reset filters')));
      return;
    }
    if (S.group) { mount(results, groupedView(rows)); return; }
    const shown = rows.slice(0, S.page * PAGE);
    const more = rows.length > shown.length
      ? h('div', { style: { textAlign: 'center', marginTop: '18px' } },
        h('button', { class: 'btn', onclick: () => { S.page += 1; emit(); } }, `Show ${Math.min(PAGE, rows.length - shown.length)} more`)) : null;
    mount(results, S.layout === 'table' ? companyTable(shown) : h('div', { class: 'grid' }, shown.map(companyCard)), more);
  }
  return { el, update, focusSearch: () => search.focus() };
}

// ------------------------------------------------------------------------------------------------ Insights view
function donut(parts, total) {
  const R = 54, C = 2 * Math.PI * R;
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', '0 0 140 140'); svg.setAttribute('width', '170'); svg.setAttribute('height', '170'); svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', parts.map((p) => `${p.label} ${p.value}`).join(', '));
  let offset = 0;
  const track = document.createElementNS(ns, 'circle');
  Object.entries({ cx: 70, cy: 70, r: R, fill: 'none', 'stroke-width': 16, stroke: 'var(--surface-3)' }).forEach(([k, v]) => track.setAttribute(k, v));
  svg.append(track);
  for (const p of parts) {
    const len = total ? (p.value / total) * C : 0;
    const c = document.createElementNS(ns, 'circle');
    Object.entries({ cx: 70, cy: 70, r: R, fill: 'none', 'stroke-width': 16, stroke: p.color, 'stroke-dasharray': `${len} ${C - len}`,
      'stroke-dashoffset': -offset, transform: 'rotate(-90 70 70)', 'stroke-linecap': 'butt' }).forEach(([k, v]) => c.setAttribute(k, v));
    svg.append(c); offset += len;
  }
  const t = document.createElementNS(ns, 'text');
  Object.entries({ x: 70, y: 76, 'text-anchor': 'middle', 'font-size': 24, 'font-weight': 700, fill: 'var(--ink)' }).forEach(([k, v]) => t.setAttribute(k, v));
  t.textContent = total ? `${Math.round((parts[0].value / total) * 100)}%` : '—';
  svg.append(t);
  return svg;
}

export function createInsightsView() {
  const el = h('section', { 'aria-label': 'Insights' });
  function update() {
    if (!S.overview) return;
    const o = S.overview, t = o.totals, names = sectorNames();
    const max = Math.max(1, ...o.sectors.map((s) => s.companies));
    const countries = new Map();
    o.companies.forEach((c) => c.country && countries.set(c.country, (countries.get(c.country) || 0) + 1));
    const topCountries = [...countries.entries()].sort((a, b) => b[1] - a[1]).slice(0, 8);
    const cmax = Math.max(1, ...topCountries.map((c) => c[1]));
    const researched = o.companies.filter((c) => c.has_profile || c.insights).length;
    mount(el,
      h('div', { class: 'view-head' }, h('div', {}, h('h1', {}, 'Portfolio insights'),
        h('p', {}, 'Where your companies sit, and how much of the picture is already researched and sourced.'))),
      h('div', { class: 'kpis' },
        [['Companies', t.companies, 'in the CRM'], ['Sectors', t.sectors, 'in the sector list'], ['Countries', t.countries, 'represented'],
          ['With a profile', t.with_profile, `${t.companies ? Math.round((100 * t.with_profile) / t.companies) : 0}% of companies`],
          ['With sourced evidence', t.with_insights, 'have a provenance row']].map(([l, v, hint]) =>
          h('div', { class: 'kpi' }, h('div', { class: 'v' }, fmtNum(v)), h('div', { class: 'l' }, l), h('div', { class: 'h' }, hint)))),
      h('div', { class: 'two' },
        h('div', { class: 'panel' }, h('h3', {}, 'Companies per sector'), h('p', { class: 'sub' }, 'Select a bar to open that sector in the company list.'),
          h('div', { class: 'hbars' }, o.sectors.map((s) => h('button', { class: 'hbar', style: { '--c': sectorColor(s.name, names), '--w': `${Math.round((100 * s.companies) / max)}%` },
            onclick: () => { S.sector = s.id; S.view = 'companies'; emit(); window.dispatchEvent(new Event('nav')); } },
            h('span', { class: 'nm' }, s.name), h('span', { class: 'tr' }, h('i', { class: 'fl', style: { display: 'block' } })), h('span', { class: 'n' }, fmtNum(s.companies)))))),
        h('div', { class: 'panel donut-wrap' }, h('div', { style: { width: '100%' } }, h('h3', {}, 'Research coverage'), h('p', { class: 'sub' }, 'Companies with a profile or sourced evidence.')),
          donut([{ label: 'Researched', value: researched, color: 'var(--blue)' }, { label: 'Not yet', value: t.companies - researched, color: 'var(--line-strong)' }], t.companies),
          h('div', { class: 'legend' },
            h('div', {}, h('i', { style: { background: 'var(--blue)' } }), 'Researched', h('b', {}, fmtNum(researched))),
            h('div', {}, h('i', { style: { background: 'var(--line-strong)' } }), 'Not yet researched', h('b', {}, fmtNum(t.companies - researched)))))),
      h('div', { class: 'panel', style: { marginTop: '14px' } }, h('h3', {}, 'Top countries'), h('p', { class: 'sub' }, 'By number of companies.'),
        h('div', { class: 'hbars' }, topCountries.map(([name, n]) => h('button', { class: 'hbar', style: { '--c': 'var(--blue)', '--w': `${Math.round((100 * n) / cmax)}%` },
          onclick: () => { S.country = name; S.view = 'companies'; emit(); window.dispatchEvent(new Event('nav')); } },
          h('span', { class: 'nm' }, name), h('span', { class: 'tr' }, h('i', { class: 'fl', style: { display: 'block' } })), h('span', { class: 'n' }, fmtNum(n)))))));
  }
  return { el, update };
}
