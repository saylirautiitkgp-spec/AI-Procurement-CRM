// Company drawer: overview, information coverage, sourced evidence, products / contacts. Opens the "What do you need?" wizard.

import { h, mount, icon, fmtNum, link, show, sectorColor, toast, timeAgo } from './ui.js';
import { api } from './api.js';
import { S, sectorNames } from './state.js';
import { openWizard } from './wizard.js';

let built = null;
let current = null;
let tab = 'overview';
const TABS = [['overview', 'Overview'], ['coverage', 'Information coverage'], ['evidence', 'Sourced evidence'], ['details', 'Products & contacts']];

function build() {
  const scrim = h('div', { class: 'scrim', onclick: closeDrawer });
  const drawer = h('aside', { class: 'drawer', role: 'dialog', 'aria-modal': 'true', 'aria-label': 'Company details' });
  document.body.append(scrim, drawer);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && drawer.classList.contains('on') && !document.querySelector('.modal.on')) closeDrawer(); });
  window.addEventListener('crm-changed', (e) => { if (current && e.detail && e.detail.id === current.company.company_id) openDrawer(current.company.company_id, true); });
  built = { scrim, drawer };
}

export function closeDrawer() {
  if (!built) return;
  built.drawer.classList.remove('on'); built.scrim.classList.remove('on');
  S.open = null; current = null;
}

export async function openDrawer(id, keepTab = false) {
  if (!built) build();
  if (!keepTab) tab = 'overview';
  S.open = id;
  const { drawer, scrim } = built;
  mount(drawer, h('div', { class: 'drawer-body' }, h('div', { class: 'skeleton', style: { height: '60px', marginBottom: '14px' } }),
    h('div', { class: 'skeleton', style: { height: '200px' } })));
  drawer.classList.add('on'); scrim.classList.add('on');
  try {
    current = await api.company(id);
    render();
  } catch (e) {
    mount(drawer, h('div', { class: 'drawer-body' }, h('div', { class: 'empty' }, icon('alert', 'lg'), h('h3', {}, 'Could not load this company'), h('p', {}, e.message),
      h('button', { class: 'btn', onclick: closeDrawer }, 'Close'))));
  }
}

function fact(label, value) { return h('div', {}, h('div', { class: 'k' }, label), h('div', { class: 'val' }, value)); }

function overviewTab(d) {
  const c = d.company, p = d.profile;
  return [
    h('div', { class: 'kv' },
      fact('Sector', c.sector || '—'), fact('Country', show(c.country)), fact('Company type', show(c.company_type)),
      fact('Priority', show(c.priority)), fact('Plants', show(c.number_of_plants)), fact('Record status', show(c.enrichment_status)),
      fact('Website', c.website ? link(c.website) : show(null)), fact('LinkedIn', c.linkedin_url ? link(c.linkedin_url) : show(null))),
    h('div', { class: 'sect' }, h('h4', {}, 'Researched profile'),
      p ? h('div', { class: 'kv' }, Object.entries(p).filter(([k]) => !['profile_id', 'company_id', 'created_at', 'updated_at'].includes(k))
        .map(([k, v]) => fact(k.replace(/_/g, ' '), show(v)))) :
        h('div', { class: 'note' }, icon('info'), 'No researched profile yet. Choose "Get information" to ask for exactly what you need.')),
    d.lead ? h('div', { class: 'sect' }, h('h4', {}, 'Lead status'), h('div', { class: 'kv' }, Object.entries(d.lead).map(([k, v]) => fact(k.replace(/_/g, ' '), show(v))))) : null,
  ];
}

function coverageTab(d) {
  const groups = { core: 'Core company record', extended: 'Extended profile', insight: 'Procurement insights' };
  return [
    h('p', { class: 'muted' }, 'Green = already researched with a source. Blue = a value is stored without provenance. Grey = nothing yet. Click any field to request it.'),
    ...Object.entries(groups).map(([layer, title]) => h('div', { class: 'sect' }, h('h4', {}, title),
      h('div', { class: 'cov' }, d.coverage.filter((f) => f.layer === layer).map((f) => h('button', { onclick: () => openWizard(d, { fields: [f.key] }) },
        h('span', { class: 't' }, f.label, f.sourced ? h('span', { class: 'chip ok' }, icon('shield', 'sm'), 'Sourced') :
          f.filled ? h('span', { class: 'chip blue' }, 'Stored') : h('span', { class: 'chip plain' }, 'Missing')),
        h('span', { class: 'v' }, f.value != null ? String(Array.isArray(f.value) ? f.value.join(', ') : f.value) : (f.sourced ? 'Recorded as sourced evidence' : 'Not available yet'))))))),
  ];
}

function evidenceTab(d) {
  if (!d.insights.length) return h('div', { class: 'empty' }, icon('shield', 'lg'), h('h3', {}, 'No sourced evidence yet'),
    h('p', {}, 'Every researched fact is stored together with its source URL and the exact supporting quote. None exist for this company so far.'));
  return d.insights.map((i) => {
    const quote = String(i.insight || '').split(' (source: ')[0];
    return h('div', { class: 'insight' },
      h('div', { style: { display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' } }, h('b', {}, i.insight_type),
        i.confidence_score != null ? h('span', { class: 'chip ok' }, `confidence ${Number(i.confidence_score).toFixed(2)}`) : null,
        h('span', { class: 'muted' }, i.generated_by || ''), i.created_at ? h('span', { class: 'muted' }, timeAgo(i.created_at)) : null),
      h('div', { style: { marginTop: '4px' } }, i.insight_title),
      h('blockquote', {}, quote), i.source_url ? link(i.source_url) : null);
  });
}

function detailsTab(d) {
  const chips = (arr) => arr.length ? h('div', { class: 'chips' }, arr.map((x) => h('span', { class: 'chip plain' }, x))) : h('p', { class: 'muted' }, 'None recorded.');
  return [
    h('div', { class: 'sect', style: { marginTop: 0 } }, h('h4', {}, `Products (${d.products.length})`), chips(d.products)),
    h('div', { class: 'sect' }, h('h4', {}, `Components purchased (${d.components.length})`), chips(d.components)),
    h('div', { class: 'sect' }, h('h4', {}, `Contacts (${d.contacts.count}) · ${d.contacts.verified} with a source`),
      d.contacts.count ? h('div', { class: 'chips' }, d.contacts.items.map((c) => h('span', { class: `chip ${c.has_source ? 'ok' : 'plain'}`, title: c.has_source ? 'has a source' : 'unverified: no source recorded' },
        [c.name, c.designation].filter(Boolean).join(' · ')))) : h('p', { class: 'muted' }, 'No contacts recorded.'),
      h('p', { class: 'muted', style: { marginTop: '8px' } }, 'Contacts without a source are legacy CRM data and are shown as unverified.')),
  ];
}

function render() {
  const d = current, c = d.company;
  const color = sectorColor(c.sector, sectorNames());
  const body = { overview: overviewTab, coverage: coverageTab, evidence: evidenceTab, details: detailsTab }[tab](d);
  const missing = d.coverage.filter((f) => !f.filled).length;
  mount(built.drawer,
    h('div', { class: 'drawer-head' },
      h('div', { class: 'row' },
        h('div', { style: { flex: 1, minWidth: 0 } },
          h('div', { class: 'chips', style: { marginBottom: '8px' } }, h('span', { class: 'chip', style: { '--c': color } }, c.sector || 'No sector'),
            c.country ? h('span', { class: 'chip plain' }, icon('pin', 'sm'), c.country) : null, h('span', { class: 'chip plain' }, `ID ${c.company_id}`)),
          h('h2', {}, c.company_name)),
        h('button', { class: 'btn sm', onclick: closeDrawer, 'aria-label': 'Close' }, icon('x'))),
      h('div', { class: 'tabs', role: 'tablist' }, TABS.map(([k, label]) => h('button', { role: 'tab', 'aria-selected': tab === k, onclick: () => { tab = k; render(); } },
        label, k === 'evidence' && d.insights.length ? ` (${d.insights.length})` : '')))),
    h('div', { class: 'drawer-body' }, body),
    h('div', { class: 'drawer-foot' },
      h('button', { class: 'btn primary', onclick: () => openWizard(d, {}) }, icon('spark'), 'Get information'),
      h('span', { class: 'muted' }, `${missing} of ${d.coverage.length} fields still missing`),
      h('span', { class: 'spacer', style: { flex: 1 } }),
      h('button', { class: 'btn', onclick: closeDrawer }, 'Close')));
  built.drawer.querySelector('.drawer-body').scrollTop = 0;
}
