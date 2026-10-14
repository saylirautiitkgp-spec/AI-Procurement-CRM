// Thin fetch wrapper. The access token (only needed when the server was started with writes enabled) lives in sessionStorage.

let token = null;
try { token = sessionStorage.getItem('apt'); } catch { /* blocked storage */ }

export const getToken = () => token;
export function setToken(t) {
  token = t || null;
  try { t ? sessionStorage.setItem('apt', t) : sessionStorage.removeItem('apt'); } catch { /* ignore */ }
}

async function req(method, path, body) {
  const headers = {};
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  if (token) headers.Authorization = `Bearer ${token}`;
  let res;
  try {
    res = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  } catch {
    const e = new Error('Cannot reach the platform server. Is it still running?');
    e.status = 0;
    throw e;
  }
  let data = null;
  try { data = await res.json(); } catch { /* empty body */ }
  if (!res.ok) {
    const d = data && data.detail;
    const msg = typeof d === 'string' ? d : (d && (d.error || d.message)) || res.statusText || 'Request failed';
    const e = new Error(msg);
    e.status = res.status; e.detail = d;
    if (res.status === 401 || res.status === 403) window.dispatchEvent(new CustomEvent('auth-needed', { detail: res.status }));
    throw e;
  }
  return data;
}

export const api = {
  status: () => req('GET', '/platform/status'),
  overview: () => req('GET', '/platform/overview'),
  company: (id) => req('GET', `/platform/companies/${id}`),
  fields: () => req('GET', '/platform/fields'),
  refresh: () => req('POST', '/platform/refresh', {}),
  estimate: (body) => req('POST', '/platform/estimate', body),
  startRun: (body) => req('POST', '/platform/runs', body),
  apply: (id, body) => req('POST', `/platform/runs/${id}/apply`, body),
  runs: () => req('GET', '/agentic/runs?limit=40'),
  run: (id) => req('GET', `/agentic/runs/${id}`),
  events: (id, after) => req('GET', `/agentic/runs/${id}/events?after=${after}&limit=300`),
  changeset: (id) => req('GET', `/agentic/runs/${id}/changeset`),
  report: (id) => req('GET', `/agentic/runs/${id}/report`),
};
