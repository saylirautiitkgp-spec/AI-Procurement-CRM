// Research runs history: reopen the timeline or the review of any earlier run.

import { h, mount, icon, timeAgo } from './ui.js';
import { api } from './api.js';
import { openRun } from './wizard.js';

const KIND = { completed: 'ok', partial: 'warn', failed: 'bad', running: 'blue', queued: 'plain', awaiting_approval: 'blue' };

export function createRunsView() {
  const body = h('div', {});
  const el = h('section', { 'aria-label': 'Research runs' },
    h('div', { class: 'view-head' }, h('div', {}, h('h1', {}, 'Research runs'), h('p', {}, 'Every research request, newest first. Open one to see what the agents did and what they found.')),
      h('button', { class: 'btn', onclick: () => load() }, icon('refresh', 'sm'), 'Refresh')),
    body);
  async function load() {
    mount(body, h('div', { class: 'skeleton', style: { height: '160px' } }));
    try {
      const { runs } = await api.runs();
      if (!runs.length) { mount(body, h('div', { class: 'empty' }, icon('spark', 'lg'), h('h3', {}, 'No research runs yet'), h('p', {}, 'Open a company and choose "Get information" to start one.'))); return; }
      mount(body, h('table', { class: 'tbl' },
        h('thead', {}, h('tr', {}, ['Company', 'Status', 'Started', 'Run id', ''].map((t) => h('th', { style: { cursor: 'default' } }, t)))),
        h('tbody', {}, runs.map((r) => h('tr', { class: 'row', tabindex: 0, onclick: () => openRun(r.run_id, r.company_name), onkeydown: (e) => e.key === 'Enter' && openRun(r.run_id, r.company_name) },
          h('td', {}, h('b', {}, r.company_name || '—')),
          h('td', {}, h('span', { class: `chip ${KIND[r.status] || 'plain'}` }, r.status)),
          h('td', {}, timeAgo(r.created_at)), h('td', {}, h('code', {}, r.run_id)),
          h('td', {}, icon('chevron')))))));
    } catch (e) {
      mount(body, h('div', { class: 'empty' }, icon('alert', 'lg'), h('h3', {}, 'Research is not available'), h('p', {}, e.message)));
    }
  }
  return { el, update: () => {}, onShow: load };
}
