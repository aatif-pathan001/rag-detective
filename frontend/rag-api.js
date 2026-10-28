// RAG Detective — thin client for the Python backend (rag_detective/).
// The whole RAG pipeline (extraction, chunking, embedding, Chroma, BM25, query rewriting,
// dense/sparse retrieval, weighted RRF and synthesis) runs in Python. This module only
// talks to the backend's streaming endpoints and hands the events to the UI.

// pdf.js is used purely to *preview* uploaded PDF pages; text extraction happens in Python.
export const PDFJS = {
  lib: 'https://cdn.jsdelivr.net/npm/pdfjs-dist@4.4.168/build/pdf.min.mjs',
  worker: 'https://cdn.jsdelivr.net/npm/pdfjs-dist@4.4.168/build/pdf.worker.min.mjs',
};

// Backend location. Override with window.RAG_API_BASE (?api= is only allowed on localhost).
function resolveBase() {
  const isLocal = location.hostname === 'localhost' || location.hostname === '127.0.0.1';
  if (isLocal) {
    const fromQuery = new URLSearchParams(location.search).get('api');
    if (fromQuery && /^https?:\/\//i.test(fromQuery)) return fromQuery;
  }
  if (window.RAG_API_BASE) return window.RAG_API_BASE;
  // Served by the FastAPI app (/ui/...) or deployed on a production origin -> same origin.
  if (location.protocol.startsWith('http') && (!isLocal || location.pathname.startsWith('/ui/'))) {
    return location.origin;
  }
  return 'http://localhost:8000';
}
export const API_ROOT = resolveBase().replace(/\/+$/, '');
export const API_BASE = API_ROOT + '/api/v1';

// Per-tab session ID so concurrent portfolio visitors get isolated Chroma & BM25 indexes.
function resolveSessionId() {
  const key = 'rag_detective_session_id';
  try {
    let sid = sessionStorage.getItem(key);
    if (!sid) {
      sid = (globalThis.crypto && crypto.randomUUID)
        ? crypto.randomUUID().replace(/-/g, '')
        : Array.from({ length: 24 }, () => Math.floor(Math.random() * 16).toString(16)).join('');
      sessionStorage.setItem(key, sid);
    }
    return sid;
  } catch (e) {
    return Array.from({ length: 24 }, () => Math.floor(Math.random() * 16).toString(16)).join('');
  }
}
export const SESSION_ID = resolveSessionId();

export function fileKind(name) {
  const ext = (String(name).split('.').pop() || '').toLowerCase();
  return ext === 'pdf' ? 'pdf' : ext === 'docx' ? 'docx' : (ext === 'md' || ext === 'markdown') ? 'md' : 'txt';
}

async function request(path, init = {}) {
  const headers = { ...(init.headers || {}), 'X-Session-Id': SESSION_ID };
  try {
    return await fetch(API_BASE + path, { ...init, headers });
  } catch (e) {
    if (e && e.name === 'AbortError') throw e;
    throw new Error(`backend unreachable at ${API_ROOT} — start it with: python -m rag_detective.cli serve`);
  }
}

async function failure(res) {
  let msg = `${res.status} ${res.statusText}`;
  try {
    const j = await res.json();
    if (j && j.detail) msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail);
  } catch (e) { /* not JSON */ }
  return new Error(msg);
}

// Parses a newline-delimited JSON (NDJSON) streaming response into events.
async function* ndjson(res) {
  if (!res.ok) throw await failure(res);
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (line) yield JSON.parse(line);
    }
  }
  buf += dec.decode();
  if (buf.trim()) yield JSON.parse(buf);
}

export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024; // 10 MB

// POST /api/v1/cases — ingest a document; yields one event per pipeline stage.
export async function* openCase(file, { signal } = {}) {
  if (file && file.size > MAX_UPLOAD_BYTES) {
    throw new Error('File exceeds the maximum allowed upload size of 10 MB.');
  }
  const fd = new FormData();
  fd.append('file', file, file.name);
  fd.append('reset', 'true');
  yield* ndjson(await request('/cases', { method: 'POST', body: fd, signal }));
}

// POST /api/v1/cases/reset — clear the active collection and create a new empty vector store.
export async function resetCase() {
  try {
    await request('/cases/reset', { method: 'POST' });
  } catch (e) { /* ignore if backend offline */ }
}

// POST /api/v1/cases/investigate — rewrite → retrieve → fuse → synthesize; yields stage events.
export async function* investigate(body, { signal } = {}) {
  yield* ndjson(await request('/cases/investigate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  }));
}

// GET /api/v1/stats — backend configuration (models, RRF/BM25 parameters, corpus size).
export async function stats() {
  const res = await request('/stats');
  if (!res.ok) throw await failure(res);
  return res.json();
}

export const SAMPLE_NAME = 'Employee_Handbook_2026.md';
export const SAMPLE_MD = `# Employee Handbook 2026

Policies in this handbook are effective from 1 January 2026 and apply to all full-time and part-time employees of Northwind Labs. Where local law grants stronger rights, local law takes precedence.

## 1. Leave Policy

Full-time employees accrue 1.5 days of paid leave per month, up to 18 days per calendar year. Part-time employees accrue leave pro rata based on contracted hours. Leave requests should be submitted in the HR portal at least two weeks in advance for absences longer than three days.

Sick leave is separate from paid leave. Employees receive 10 days of paid sick leave per year; a medical certificate is required for absences longer than three consecutive working days.

### 1.1 Carry-forward

Up to 6 unused paid leave days roll into the first quarter of the next year. Carried-forward days that are not used by 31 March expire and cannot be paid out, except where required by law.

### 1.2 Parental leave

Primary caregivers receive 20 weeks of fully paid parental leave and secondary caregivers receive 6 weeks. Parental leave can be taken in up to three blocks within the first 18 months after birth or adoption.

## 2. Remote Work

Northwind Labs operates a hybrid schedule: employees are expected in the office three days a week, with team-anchor days set by each department. Fully remote arrangements require written approval from a director and are reviewed every six months.

| Team | Office days |
| Engineering | 2 |
| Sales | 3 |
| Operations | 4 |

Employees working remotely must use the company VPN, work from a private space when handling customer data, and remain reachable during core hours of 10:00 to 16:00 local time.

## 3. Expense Claims

Submit receipts within 30 days of purchase via the finance portal. Claims older than 90 days will not be reimbursed. Meals while travelling are covered up to 60 EUR per day; alcohol is not reimbursable.

Travel must be booked through the approved travel tool. Economy class is standard for flights under six hours; business class requires VP approval.

## 4. IT and Security

Laptops must use full-disk encryption and lock automatically after five minutes of inactivity. Multi-factor authentication is mandatory for email, source control and the HR portal.

Report lost or stolen devices to the security team within one hour. Never store customer data on personal devices or in personal cloud storage.

## 5. Health Benefits

Medical cover starts on day one and includes dental and vision care for employees and their dependants. Every employee also receives an annual wellbeing budget of 500 EUR for gym memberships, therapy or ergonomic equipment.

## 6. Code of Conduct

Treat colleagues, customers and partners with respect. Harassment, discrimination and retaliation are not tolerated. Concerns can be reported to People Operations or anonymously through the ethics hotline; every report is investigated within ten working days.
`;
