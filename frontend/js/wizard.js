// "What information do you need?" — choose fields, see the plan and budget BEFORE anything runs, watch the research,
// review every proposed value with its source and exact quote, then (only if the server allows it) approve and apply.

import { h, mount, icon, fmtNum, link, show, toast, debounce, download, timeAgo } from './ui.js';
import { api } from './api.js';
import { S } from './state.js';

let built = null;
let W = null;            // wizard state
const FINAL = new Set(['completed', 'partial', 'failed']);
const ACTION = { fill_empty: ['Fill empty field', 'ok'], add_new: ['Add new', 'blue'], update_conflict: ['Would overwrite', 'bad'], no_change: ['Already the same', 'plain'], report_only: ['Report only', 'plain'] };
const GROUP_ICON = { company: 'building', procurement: 'cart', contact: 'users', supplier: 'truck', lead: 'target' };

// ------------------------------------------------------------------------------------------------ shell
function build() {
  const scrim = h('div', { class: 'scrim', style: { zIndex: 75 }, onclick: () => closeWizard() });
  const modal = h('div', { class: 'modal', role: 'dialog', 'aria-modal': 'true', 'aria-label': 'Get information' });
  document.body.append(scrim, modal);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && modal.classList.contains('on')) closeWizard(); });
  built = { scrim, modal };
}

export function closeWizard() {
  if (!built) return;
  if (W && W.timer) clearTimeout(W.timer);
  if (W) W.closed = true;
  built.modal.classList.remove('on'); built.scrim.classList.remove('on');
}

export async function openWizard(detail, opts = {}) {
  if (!built) build();
  if (!S.fields) { try { S.fields = await api.fields(); } catch (e) { toast(e.message, 'bad'); return; } }
  if (W && W.timer) clearTimeout(W.timer);
  W = { detail, name: detail.company.company_name, id: detail.company.company_id, step: 1, selected: new Set(opts.fields || []), refresh: false,
    est: null, estSeq: 0, run: null, cs: null, report: null, picked: new Set(), closed: false, busy: false };
  built.modal.classList.add('on'); built.scrim.classList.add('on');
  render(); scheduleEstimate();
}

export async function openRun(runId, name) {
  if (!built) build();
  if (W && W.timer) clearTimeout(W.timer);
  W = { detail: null, name: name || 'Company', id: null, step: 2, selected: new Set(), refresh: false, est: null, estSeq: 0,
    run: { id: runId, last: 0, events: [], tracker: null, agents: [], mode: S.status && S.status.research_mode }, cs: null, report: null, picked: new Set(), closed: false, busy: false };
  built.modal.classList.add('on'); built.scrim.classList.add('on');
  render(); poll();
}

function frame(stepLabel, bodyKids, footKids) {
  const steps = ['Choose & plan', 'Research', 'Review'];
  mount(built.modal,
    h('div', { class: 'modal-head' },
      h('div', {}, h('h2', {}, 'Get information'), h('div', { class: 'muted' }, W.name)),
      h('div', { class: 'stepper' }, steps.map((s, i) => h('span', { class: W.step === i + 1 ? 'on' : W.step > i + 1 ? 'done' : '' }, h('b', {}, W.step > i + 1 ? '✓' : i + 1), s))),
      h('button', { class: 'btn sm', onclick: () => closeWizard(), 'aria-label': 'Close' }, icon('x'))),
    h('div', { class: 'modal-body' }, bodyKids),
    h('div', { class: 'modal-foot' }, footKids));
}

function render() { ({ 1: stepChoose, 2: stepResearch, 3: stepReview }[W.step])(); }

// ------------------------------------------------------------------------------------------------ step 1: choose + plan
function coverageOf(key) { return W.detail && W.detail.coverage.find((f) => f.key === key); }

function stateOf(f) {
  const cov = coverageOf(f.key);
  if (cov && cov.sourced) return ['Already researched with a source', true];
  if (cov && cov.filled) return ['Already stored in the CRM', true];
  if (f.key === 'procurement_contacts' && W.detail) return [`${W.detail.contacts.count} on file, ${W.detail.contacts.verified} with a source`, false];
  if (f.key === 'suppliers' && W.detail) return ['Needs known products or components', false];
  return ['Not available yet', false];
}

function stepChoose() {
  const groups = S.fields.groups;
  const sel = W.selected;
  const presetOn = (p) => p.fields.length === sel.size && p.fields.every((k) => sel.has(k));
  const body = [
    h('p', { class: 'muted', style: { marginBottom: '14px' } }, 'Tell the platform exactly which information you want. Only the agents needed for those fields will run, and you will see the cost before anything starts.'),
    h('div', { class: 'presets' }, S.fields.presets.map((p) => h('button', { class: `preset ${presetOn(p) ? 'on' : ''}`, onclick: () => { W.selected = new Set(p.fields); render(); scheduleEstimate(); } },
      h('b', {}, icon('spark', 'sm'), p.title), h('span', {}, p.blurb)))),
    h('div', { class: 'ask-grid' },
      h('div', {}, groups.map((g) => {
        const allOn = g.fields.every((f) => sel.has(f.key));
        return h('div', { class: 'fgroup' },
          h('header', {}, icon(GROUP_ICON[g.agent] || 'info', 'lg'), h('div', { class: 'ttl' }, h('b', {}, g.title), h('small', {}, g.blurb)),
            h('span', { class: `chip ${g.uses_model ? 'blue' : 'plain'}` }, g.uses_model ? 'uses the AI model' : 'computed in code'),
            h('button', { class: 'btn sm', onclick: () => { g.fields.forEach((f) => (allOn ? sel.delete(f.key) : sel.add(f.key))); render(); scheduleEstimate(); } }, allOn ? 'Clear' : 'Select all')),
          h('div', { class: 'fopts' }, g.fields.map((f) => {
            const [label, stored] = stateOf(f);
            const on = sel.has(f.key);
            return h('label', { class: `fopt ${on ? 'on' : ''} ${stored && !W.refresh && g.agent === 'company' ? 'stored' : ''}` },
              h('input', { type: 'checkbox', checked: on, onchange: (e) => { e.target.checked ? sel.add(f.key) : sel.delete(f.key); render(); scheduleEstimate(); } }),
              h('span', { class: 'nm' }, f.label, h('span', { class: 'st' }, label)));
          })));
      }),
      h('label', { class: 'toggle' }, h('input', { type: 'checkbox', checked: W.refresh, onchange: (e) => { W.refresh = e.target.checked; render(); scheduleEstimate(); } }),
        'Re-verify values that are already stored (otherwise they are skipped)')),
      planPanel()),
  ];
  const can = W.est && W.est.can_run && sel.size > 0;
  frame('', body, [
    h('button', { class: 'btn', onclick: () => closeWizard() }, 'Cancel'),
    h('span', { class: 'muted' }, `${sel.size} field${sel.size === 1 ? '' : 's'} selected`),
    h('span', { style: { flex: 1 } }),
    h('button', { class: 'btn primary', disabled: !can || W.busy, onclick: startRun }, icon('play', 'sm'), W.est && W.est.mode === 'simulation' ? 'Run simulation' : 'Run research')]);
}

function planPanel() {
  const e = W.est;
  const kids = [h('h3', {}, 'Plan & budget')];
  if (!W.selected.size) { kids.push(h('p', { class: 'muted' }, 'Pick at least one field to see which agents run and what it costs.')); return h('aside', { class: 'plan' }, kids); }
  if (!e) { kids.push(h('div', { style: { display: 'flex', gap: '10px', alignItems: 'center' } }, h('div', { class: 'spinner' }), 'Calculating…')); return h('aside', { class: 'plan' }, kids); }
  if (e.error) { kids.push(h('div', { class: 'note bad' }, icon('alert'), e.error)); return h('aside', { class: 'plan' }, kids); }
  if (e.agent_titles.length) {
    kids.push(h('ol', { class: 'steps', 'aria-label': 'Agents that will run, in order' }, e.agent_titles.map((a, i) =>
      h('li', {}, h('b', {}, i + 1), a.title, h('span', { class: `chip ${a.uses_model ? 'blue' : 'plain'}` }, a.uses_model ? 'AI model' : 'code')))));
  }
  if (e.skipped_already_stored.length) kids.push(h('div', { class: 'note' }, icon('info'), `${e.skipped_already_stored.length} field(s) already stored are skipped: ${e.skipped_already_stored.join(', ')}.`));
  const p = e.plan;
  if (p) {
    const live = e.mode === 'live';
    const rows = [['AI model requests', live ? (p.requests_max === 0 ? 'none' : `up to ${p.requests_max}${p.retries ? ` (incl. ${p.retries} retry for a 503)` : ''}`) : 'simulated'],
      ['Web search credits', live ? `up to ${p.tavily_credits} (cached searches cost 0)` : '0'],
      ['Estimated tokens', live && p.tokens_in ? `≈ ${fmtNum(p.tokens_in)} in / ${fmtNum(p.tokens_out)} out (upper bound)` : '—'],
      ['Database reads', live ? `up to ${p.selects_max}` : 'read-only']];
    kids.push(h('div', {}, rows.map(([k, v]) => h('div', { class: 'stat' }, h('span', { class: 'muted' }, k), h('b', {}, v)))));
  }
  if (e.quota) {
    const pct = Math.min(100, Math.round((100 * e.quota.used_24h) / Math.max(1, e.quota.planning_cap)));
    kids.push(h('div', {}, h('div', { class: 'stat', style: { border: 0 } }, h('span', { class: 'muted' }, 'Daily request allowance (last 24 h)'), h('b', {}, `${e.quota.used_24h} / ${e.quota.planning_cap}`)),
      h('div', { class: 'gauge' }, h('i', { style: { '--w': `${pct}%` } })), h('div', { class: 'muted', style: { fontSize: '12px', marginTop: '4px' } }, `${e.quota.remaining_under_cap} left under the safety cap (console limit ${e.quota.limit_per_day}/day)`)));
  }
  e.notes.forEach((n) => kids.push(h('div', { class: `note ${e.mode === 'simulation' ? 'sim' : e.mode === 'live' ? 'live' : ''}` }, icon(e.mode === 'live' ? 'bolt' : 'info'), n)));
  if (e.blocked) {
    if (e.block_line) kids.push(h('div', { class: 'note bad' }, icon('lock'), e.block_line));
    e.reasons.forEach((r) => kids.push(h('div', { class: 'note bad' }, icon('alert'), r)));
  }
  return h('aside', { class: 'plan' }, kids);
}

const scheduleEstimate = debounce(async () => {
  if (!W || W.step !== 1) return;
  const seq = ++W.estSeq;
  if (!W.selected.size) { W.est = null; planRefresh(); return; }
  W.est = null; planRefresh();
  try {
    const est = await api.estimate({ company_id: W.id, fields: [...W.selected], refresh: W.refresh });
    if (seq === W.estSeq && W.step === 1) { W.est = est; render(); }
  } catch (e) {
    if (seq === W.estSeq) { W.est = { error: e.message, agent_titles: [], skipped_already_stored: [], notes: [], reasons: [] }; render(); }
  }
}, 220);

function planRefresh() {
  const old = built.modal.querySelector('.plan');
  if (old) old.replaceWith(planPanel());
}

// ------------------------------------------------------------------------------------------------ step 2: research in progress
async function startRun() {
  W.busy = true; render();
  try {
    const r = await api.startRun({ company_id: W.id, fields: [...W.selected], refresh: W.refresh });
    W.run = { id: r.run_id, last: 0, events: [], tracker: null, agents: r.agents, mode: r.mode };
    W.step = 2; W.busy = false; render(); poll();
  } catch (e) {
    W.busy = false; render();
    toast(e.message, 'bad', 7000);
  }
}

function describe(ev) {
  const d = ev.data || {};
  const a = ev.agent ? ` · ${ev.agent}` : '';
  switch (ev.type) {
    case 'run_started': return ['bolt', '', 'Run started', d.company || ''];
    case 'db_lookup': return ['db', '', 'CRM record found', d.status ? `match: ${d.status}` : ''];
    case 'crm_sections_loaded': return ['db', '', 'Read CRM tables', (d.sections || []).join(', ')];
    case 'route': return ['target', '', `Next agent: ${d.next}`, d.reason || ''];
    case 'tool_call': return d.tool === 'web_search' ? ['search', 'web', 'Searching the web' + a, d.args && d.args.query] : ['globe', 'web', 'Reading a page' + a, d.args && d.args.url];
    case 'tool_result': return d.ok ? ['check', 'web', `Got ${d.docs_added || 0} document(s)${d.cached ? ' from cache' : ''}${d.credits ? ` · ${d.credits} credit` : ''}`, (d.urls || [])[0] || ''] :
      ['alert', 'warn', 'A tool call failed', (d.error && (d.error.code || d.error.message)) || ''];
    case 'prefetch': return ['layers', 'web', `Evidence packet ${d.status === 'packet' ? 'ready' : 'not usable'}${a}`, d.status === 'packet' ? `${(d.pages || []).length} page(s) of ${d.domain}` : d.reason || ''];
    case 'llm_call': return ['spark', 'llm', `Model answered${a}`, `${fmtNum(d.input_tokens)} tokens in · ${fmtNum(d.output_tokens)} out · ${d.latency_s}s`];
    case 'llm_error': return ['alert', 'bad', `Model error${a}`, `${d.kind}: ${d.message || ''}`];
    case 'submit': return ['shield', 'llm', `Findings checked${a}`, `${d.accepted} accepted · ${d.rejected} rejected by the evidence validator`];
    case 'agent_finished': return ['check', '', `Agent finished${a}`, `${d.status}${d.end_reason && d.end_reason !== 'submitted' ? ` (${d.end_reason})` : ''}`];
    case 'agent_limit': case 'stop': return ['alert', 'warn', 'Stopped by a safety limit', d.reason || ''];
    case 'pinned_identity': return ['shield', '', d.verified ? 'Identity re-verified' : 'Identity not confirmed', d.reason || ''];
    case 'empty_response': return ['alert', 'warn', 'Model returned nothing — not retried', ''];
    case 'run_finished': return ['check', '', 'Run finished', d.status || ''];
    case 'run_failed': return ['alert', 'bad', 'Run failed', d.reason || ''];
    default: return ['info', '', ev.type.replace(/_/g, ' '), ''];
  }
}

async function poll() {
  const run = W && W.run;
  if (!run || W.closed) return;
  try {
    const [t, ev] = await Promise.all([api.run(run.id), api.events(run.id, run.last)]);
    run.tracker = t.tracker || {}; run.status = t.status;
    ev.events.forEach((e) => run.events.push(e));
    run.last = ev.last_seq;
    if (W.step === 2) renderResearch();
    if (FINAL.has(t.status)) { await finishRun(); return; }
  } catch (e) {
    run.errors = (run.errors || 0) + 1;
    if (run.errors > 4) { toast(e.message, 'bad'); return; }
  }
  W.timer = setTimeout(poll, 1000);
}

function agentChips() {
  const run = W.run, tr = run.tracker || {};
  const names = (run.agents && run.agents.length ? run.agents : Object.keys(tr.agents || {}));
  return names.map((n) => {
    const done = tr.agents && tr.agents[n];
    const running = tr.current_agent === n;
    const bad = done && done.end_reason && done.end_reason !== 'submitted' && done.end_reason !== 'unknown';
    return h('span', { class: `chip ${bad ? 'warn' : done ? 'ok' : running ? 'blue' : 'plain'}` },
      running ? h('span', { class: 'spinner', style: { width: '12px', height: '12px', borderWidth: '2px' } }) : done ? icon(bad ? 'alert' : 'check', 'sm') : null, n);
  });
}

function stepResearch() { frame('', researchBody(), researchFoot()); }
function renderResearch() {
  const body = built.modal.querySelector('.modal-body');
  if (!body) return;
  const keep = body.scrollTop, atEnd = body.scrollHeight - body.scrollTop - body.clientHeight < 80;
  mount(body, researchBody());
  body.scrollTop = atEnd ? body.scrollHeight : keep;
  mount(built.modal.querySelector('.modal-foot'), researchFoot());
}

function researchBody() {
  const run = W.run, tr = run.tracker || {}, u = tr.usage || {};
  const sim = (run.mode || (S.status && S.status.research_mode)) === 'simulation';
  return [
    sim ? h('div', { class: 'note sim', style: { marginBottom: '14px' } }, icon('info'), 'SIMULATION — synthetic pages and a scripted model. No quota is used and nothing can be written.') : null,
    h('div', { class: 'counters' }, [['Model requests', u.llm_calls], ['Tool calls', u.tool_calls], ['Search credits', u.search_credits], ['Sources found', (tr.sources || {}).discovered], ['Sources validated', (tr.sources || {}).validated]]
      .map(([l, v]) => h('div', {}, h('b', {}, fmtNum(v || 0)), h('span', {}, l)))),
    h('div', { class: 'flow', style: { marginBottom: '16px' } }, agentChips()),
    h('div', { class: 'timeline', 'aria-live': 'polite' }, run.events.length ? run.events.map((e) => {
      const [ic, kind, title, desc] = describe(e);
      return h('div', { class: `tl ${kind}` }, h('div', { class: 'bullet' }, icon(ic, 'sm')), h('div', {}, h('div', { class: 't' }, title), desc ? h('div', { class: 'd' }, desc) : null));
    }) : h('div', { style: { display: 'flex', gap: '10px', alignItems: 'center' } }, h('div', { class: 'spinner' }), 'Starting…')),
  ];
}

function researchFoot() {
  const st = W.run && W.run.status;
  return [h('div', { style: { display: 'flex', gap: '10px', alignItems: 'center' } }, st && FINAL.has(st) ? icon('check') : h('div', { class: 'spinner' }),
    h('b', {}, st && FINAL.has(st) ? `Run ${st}` : `Running… ${st || 'queued'}`)), h('span', { style: { flex: 1 } }),
    h('button', { class: 'btn', onclick: () => closeWizard() }, 'Hide (it keeps running)')];
}

async function finishRun() {
  const id = W.run.id;
  try { W.cs = await api.changeset(id); } catch { W.cs = null; }
  try { W.report = await api.report(id); } catch { W.report = null; }
  W.picked = new Set((W.cs ? W.cs.items : []).filter(eligible).filter((i) => (i.confidence || 0) >= 0.6).map((i) => i.id));
  W.step = 3; render();
}

// ------------------------------------------------------------------------------------------------ step 3: review
const eligible = (i) => i.entity === 'company' && ['company', 'procurement'].includes(i.agent) && ['fill_empty', 'add_new'].includes(i.action);

function valueText(v) {
  if (v == null || v === '') return '—';
  if (Array.isArray(v)) return v.join(', ');
  if (typeof v === 'object') return Object.values(v).filter((x) => x != null).join(' — ');
  return String(v);
}

function itemCard(i) {
  const [alabel, akind] = ACTION[i.action] || [i.action, 'plain'];
  const ok = eligible(i);
  const conf = Math.round((i.confidence || 0) * 100);
  return h('div', { class: 'result' },
    h('header', {},
      ok ? h('input', { type: 'checkbox', checked: W.picked.has(i.id), 'aria-label': `Select ${i.label}`, onchange: (e) => { e.target.checked ? W.picked.add(i.id) : W.picked.delete(i.id); renderReviewFoot(); }, style: { width: '18px', height: '18px', accentColor: 'var(--blue)' } }) : null,
      h('div', { class: 'ttl' }, h('b', {}, i.label), h('small', {}, i.target || 'report only')),
      h('span', { class: `chip ${akind}` }, alabel),
      i.confidence != null ? h('span', { class: 'confbar', title: i.confidence_basis || '' }, h('i', {}, h('b', { style: { '--w': `${conf}%`, '--c': conf >= 60 ? 'var(--ok)' : 'var(--warn)' } })), `${conf}%`) : null),
    h('div', { class: 'body' },
      h('div', { class: 'diff' }, h('div', { class: 'old' }, h('small', {}, 'Stored now'), valueText(i.old_value)), h('div', { class: 'arr' }, icon('chevron')),
        h('div', { class: 'new' }, h('small', {}, 'Proposed'), valueText(i.proposed_value))),
      (i.evidence || []).map((e) => h('div', { class: 'ev' }, h('q', {}, e.snippet), link(e.url))),
      h('div', { class: 'muted', style: { fontSize: '12.5px' } }, ok ? 'Eligible for the guarded database update after your approval.' :
        'Review only — contacts, suppliers, lead scores and overwrites are never written automatically.')));
}

function stepReview() {
  const cs = W.cs, rep = W.report, run = W.run, tr = run.tracker || {}, u = tr.usage || {};
  const sim = (run.mode || (S.status && S.status.research_mode)) === 'simulation';
  const body = [
    sim ? h('div', { class: 'note sim', style: { marginBottom: '14px' } }, icon('info'), 'SIMULATION — these results come from synthetic pages. They illustrate the workflow; they are not research about the real company.') : null,
    h('div', { class: 'counters' }, [['Status', run.status], ['Model requests', u.llm_calls || 0], ['Search credits', u.search_credits || 0], ['Database reads', u.db_selects ?? '—'], ['Proposed updates', cs ? cs.items.length : 0], ['Not found', cs ? cs.unresolved.length : 0]]
      .map(([l, v]) => h('div', {}, h('b', {}, typeof v === 'number' ? fmtNum(v) : v), h('span', {}, l)))),
    (tr.status_reasons || []).length ? h('div', { class: 'note', style: { marginBottom: '14px' } }, icon('info'), `Why this is "${run.status}": ${tr.status_reasons.join('; ')}`) : null,
    ...(!cs ? [h('div', { class: 'empty' }, icon('alert', 'lg'), h('h3', {}, 'No change set was produced'), h('p', {}, 'The run ended before any finding could be validated. Nothing was written. See the timeline for the reason.'),
      h('button', { class: 'btn', onclick: () => { W.step = 2; render(); } }, 'Show the timeline'))] : [
      h('h3', { style: { margin: '6px 0 12px' } }, `Proposed updates (${cs.items.length})`),
      cs.items.length ? cs.items.map(itemCard) : h('p', { class: 'muted' }, 'Nothing could be confirmed with a source and an exact quote.'),
      cs.unresolved.length ? h('div', { class: 'sect' }, h('h4', {}, `Not found or not supported (${cs.unresolved.length})`),
        h('div', { class: 'chips' }, cs.unresolved.map((x) => h('span', { class: 'chip warn', title: [x.reason, ...(x.rejection_reasons || [])].join(' · ') }, x.label || x.field)))) : null,
      cs.skipped.length ? h('div', { class: 'sect' }, h('h4', {}, 'Skipped — already stored'), h('div', { class: 'chips' }, cs.skipped.map((x) => h('span', { class: 'chip plain' }, x.label || x.field)))) : null,
      rep && rep.gaps && (rep.gaps.rejected_claims || []).length ? h('div', { class: 'sect' }, h('h4', {}, 'Rejected by the evidence validator'),
        h('div', { class: 'chips' }, rep.gaps.rejected_claims.map((x) => h('span', { class: 'chip bad', title: (x.reasons || []).join(', ') }, `${x.field || x.kind}: ${(x.reasons || [])[0] || 'rejected'}`)))) : null,
      applyBox(cs, sim)]),
  ];
  frame('', body, reviewFoot());
}

function reviewFoot() {
  const cs = W.cs;
  return [h('button', { class: 'btn', onclick: () => { W.step = 1; W.run = null; render(); scheduleEstimate(); } }, 'New request'),
    cs ? h('button', { class: 'btn', onclick: () => download(`change-set-${W.run.id}.json`, { company: W.name, run_id: W.run.id, change_set: W.cs, report: W.report }) }, icon('download', 'sm'), 'Download JSON') : null,
    h('span', { style: { flex: 1 } }), h('button', { class: 'btn primary', onclick: () => closeWizard() }, 'Done')];
}
function renderReviewFoot() {
  const btn = built.modal.querySelector('#apply-btn');
  if (btn) { btn.textContent = `Apply ${W.picked.size} selected to the database`; btn.disabled = !applyReady(); }
}

function applyReady() {
  const a = built.modal.querySelector('#appr'), t = built.modal.querySelector('#typed'), c = built.modal.querySelector('#ack');
  return !!(W.picked.size && a && a.value.trim().length >= 2 && c && c.checked && t && t.value.trim().toLowerCase() === String(W.name).trim().toLowerCase());
}

function applyBox(cs, sim) {
  const st = S.status || {};
  if (sim) return h('div', { class: 'apply-box sim' }, h('b', {}, 'Simulation results are never written to the database.'),
    h('span', {}, 'Start the server with --research live to run real research.'));
  if (!st.writes_enabled) return h('div', { class: 'apply-box sim' }, h('b', {}, 'Database writes are switched off on this server.'),
    h('span', {}, 'You can review and download the change set. To allow approved updates, restart with --research live --enable-writes (a one-time access token is then required).'));
  const res = h('div', { id: 'apply-result' });
  const upd = debounce(renderReviewFoot, 50);
  return h('div', { class: 'apply-box' },
    h('div', {}, h('b', {}, 'Approve and write to the CRM'), h('div', { class: 'muted' }, 'Only the ticked items are written. Additive and validated only: empty fields are filled and new sourced insights are added — nothing is overwritten or deleted, and no contacts or suppliers are created.')),
    h('div', { style: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(240px,1fr))', gap: '10px' } },
      h('input', { class: 'input', id: 'appr', placeholder: 'Your name (recorded in the audit log)', oninput: upd }),
      h('input', { class: 'input', id: 'typed', placeholder: `Type the company name to confirm: ${W.name}`, oninput: upd })),
    h('label', { class: 'toggle' }, h('input', { type: 'checkbox', id: 'ack', onchange: upd }), 'I have read the sources and quotes above and approve these changes.'),
    h('div', {}, h('button', { class: 'btn danger', id: 'apply-btn', disabled: true, onclick: doApply }, `Apply ${W.picked.size} selected to the database`)),
    res);
}

async function doApply() {
  if (!applyReady() || W.busy) return;
  W.busy = true;
  const btn = built.modal.querySelector('#apply-btn'); btn.disabled = true; btn.textContent = 'Writing…';
  const out = built.modal.querySelector('#apply-result');
  try {
    const r = await api.apply(W.run.id, { approver: built.modal.querySelector('#appr').value.trim(), confirm: true, item_ids: [...W.picked] });
    const rb = r.read_back || {};
    mount(out,
      h('div', { class: `note ${r.writes ? '' : 'bad'}`, style: { background: r.writes ? 'var(--ok-soft)' : undefined } }, icon(r.writes ? 'check' : 'alert'),
        r.blocked ? `Stopped: ${r.blocked}` : `${r.writes} database row${r.writes === 1 ? '' : 's'} written for ${W.name}. Database reads used: ${r.write_selects_used}/${r.select_cap}.`),
      (r.skipped_items || []).length ? h('div', { class: 'chips' }, r.skipped_items.map((s) => h('span', { class: 'chip warn', title: s.reason }, `${s.field || s.id}: ${s.reason}`))) : null,
      (r.proposals || []).map((p) => h('div', { class: 'muted' }, `${p.summary} — ${p.applied ? 'applied' : (p.not_applied || p.status)}`)),
      rb.company_profile ? h('div', { class: 'muted' }, `Read back from the database: profile #${rb.company_profile.profile_id || '—'}, ${(rb.ai_insights_by_agentic || []).length} sourced insight row(s).`) : null);
    if (r.writes) { toast(`${r.writes} row(s) written`, 'ok'); window.dispatchEvent(new CustomEvent('crm-changed', { detail: { id: W.id } })); window.dispatchEvent(new Event('crm-reload')); }
  } catch (e) {
    mount(out, h('div', { class: 'note bad' }, icon('alert'), e.message));
  } finally { W.busy = false; btn.textContent = `Apply ${W.picked.size} selected to the database`; btn.disabled = !applyReady(); }
}
