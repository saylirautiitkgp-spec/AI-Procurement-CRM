// Boot, shell (top bar, navigation, sidebar), theme, token settings, keyboard shortcuts.

import { h, $, mount, icon, toast, safeStore } from './ui.js';
import { api, getToken, setToken } from './api.js';
import { S, subscribe, emit, restorePrefs } from './state.js';
import { createSidebar, createCompaniesView, createInsightsView } from './companies.js';
import { createRunsView } from './runs.js';

function wordmark() {
  // Placeholder only: drop the official file at frontend/assets/motherson-logo.svg|png and it is used automatically.
  const ns = 'http://www.w3.org/2000/svg';
  const s = document.createElementNS(ns, 'svg');
  s.setAttribute('viewBox', '0 0 214 36'); s.setAttribute('role', 'img'); s.setAttribute('aria-label', 'Motherson');
  s.innerHTML = '<text x="0" y="26" font-family="Segoe UI, Arial, sans-serif" font-weight="800" font-size="25" letter-spacing="3.2" fill="#0b2a5b">MOTHERSON</text><rect x="1" y="31" width="46" height="3.4" rx="1.7" fill="#d6263b"/>';
  return s;
}

function logo(src) {
  // The server reports which official logo file exists in frontend/assets (status.logo); none -> the placeholder wordmark.
  const plate = h('div', { class: 'logo-plate', title: 'Motherson' });
  if (src) {
    const img = h('img', { alt: 'Motherson', decoding: 'async', src });
    img.addEventListener('error', () => { img.remove(); plate.append(wordmark()); });
    plate.append(img);
  } else plate.append(wordmark());
  return plate;
}

// ---- theme
function applyTheme(t) { document.documentElement.setAttribute('data-theme', t); safeStore('apt.theme', t); }
function initialTheme() {
  return safeStore('apt.theme') || (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
}

// ---- shell
const views = {};
let sidebar, content, navButtons = {};

function modePill(st) {
  const m = st.research_mode;
  const label = m === 'live' ? `LIVE research · ${st.model || ''}` : m === 'simulation' ? 'SIMULATION' : 'Research off';
  return h('span', { class: `pill ${m === 'live' ? 'live' : m === 'simulation' ? 'sim' : 'off'}`, title: m === 'simulation' ? 'Scripted model and synthetic pages: no quota used, nothing written' : m === 'live' ? 'Real Gemini, Tavily and Supabase calls within the shown budgets' : 'Browsing only' },
    h('i'), label);
}

function shell() {
  const st = S.status;
  const themeBtn = h('button', { class: 'icon-btn', 'aria-label': 'Toggle light / dark theme', title: 'Light / dark', onclick: () => {
    const t = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark'; applyTheme(t); themeBtn.replaceChildren(icon(t === 'dark' ? 'sun' : 'moon'));
  } }, icon(document.documentElement.getAttribute('data-theme') === 'dark' ? 'sun' : 'moon'));
  const nav = h('nav', { class: 'nav', 'aria-label': 'Main' }, [['companies', 'Companies', 'building'], ['insights', 'Insights', 'layers'], ['runs', 'Research runs', 'spark']]
    .map(([k, label, ic]) => (navButtons[k] = h('button', { onclick: () => go(k) }, icon(ic), h('span', {}, label)))));
  const top = h('header', { class: 'topbar' },
    h('div', { class: 'brand' }, logo(st.logo), h('div', { class: 'brand-text' }, h('b', {}, 'AI CRM Procurement Platform'), h('span', {}, 'Supplier intelligence · Motherson'))),
    nav, h('div', { class: 'spacer' }),
    h('span', { class: 'pill data', title: 'Where the company list comes from' }, h('i'), st.data_source === 'supabase' ? 'Live CRM data' : 'Demo data'),
    modePill(st),
    h('button', { class: 'icon-btn', title: 'Reload CRM data', 'aria-label': 'Reload CRM data', onclick: reload }, icon('refresh')),
    h('button', { class: 'icon-btn', title: 'Access token', 'aria-label': 'Access token', onclick: tokenDialog }, icon('key')),
    themeBtn);
  sidebar = createSidebar();
  content = h('main', { class: 'content', id: 'main' });
  views.companies = createCompaniesView(); views.insights = createInsightsView(); views.runs = createRunsView();
  mount($('#app'), top, h('div', { class: 'layout' }, sidebar.el, content));
}

function go(view) {
  S.view = view;
  Object.entries(navButtons).forEach(([k, b]) => b.setAttribute('aria-current', k === view ? 'page' : 'false'));
  sidebar.el.style.display = view === 'runs' ? 'none' : '';
  $('.layout').style.gridTemplateColumns = view === 'runs' ? '1fr' : '';
  mount(content, views[view].el);
  views[view].update(); if (views[view].onShow) views[view].onShow();
  window.scrollTo({ top: 0 });
}

async function loadData() {
  S.overview = await api.overview();
  emit();
}

async function reload() {
  try { await api.refresh(); await loadData(); toast('CRM data reloaded', 'ok'); } catch (e) { toast(e.message, 'bad'); }
}

function tokenDialog() {
  const t = window.prompt('Access token (only needed when the server was started with --enable-writes or AGENTIC_API_TOKEN):', getToken() || '');
  if (t === null) return;
  setToken(t.trim()); toast(t.trim() ? 'Token saved for this browser session' : 'Token cleared', 'ok');
  loadData().catch((e) => toast(e.message, 'bad'));
}

function loadingScreen() {
  mount($('#app'), h('div', { style: { display: 'grid', placeItems: 'center', height: '100vh', gap: '14px', alignContent: 'center' } }, logo(), h('div', { class: 'spinner' }), h('div', { class: 'muted' }, 'Loading the platform…')));
}

function fatal(e) {
  mount($('#app'), h('div', { style: { display: 'grid', placeItems: 'center', minHeight: '100vh', padding: '24px' } },
    h('div', { class: 'empty', style: { maxWidth: '560px' } }, icon('alert', 'lg'), h('h3', {}, 'The platform could not start'), h('p', {}, e.message),
      e.status === 401 || e.status === 403 ? h('button', { class: 'btn primary', onclick: tokenDialog }, 'Enter access token') : h('button', { class: 'btn', onclick: () => location.reload() }, 'Try again'))));
}

async function boot() {
  applyTheme(initialTheme());
  restorePrefs();
  loadingScreen();
  try {
    S.status = await api.status();
    shell();
    const [overview, fields] = await Promise.all([api.overview(), api.fields().catch(() => null)]);
    S.overview = overview; S.fields = fields;
    subscribe(() => { sidebar.update(); views[S.view].update(); });
    go('companies');
    emit();
  } catch (e) { fatal(e); }
}

// ---- global events
window.addEventListener('nav', () => go('companies'));
window.addEventListener('crm-reload', () => loadData().catch(() => {}));
window.addEventListener('auth-needed', () => { if (!window.__authPrompted) { window.__authPrompted = true; toast('This server needs an access token — click the key icon.', 'warn', 7000); setTimeout(() => (window.__authPrompted = false), 8000); } });
document.addEventListener('keydown', (e) => {
  const typing = /input|textarea|select/i.test(document.activeElement && document.activeElement.tagName);
  if (e.key === '/' && !typing && views.companies) { e.preventDefault(); if (S.view !== 'companies') go('companies'); views.companies.focusSearch(); }
});

boot();
