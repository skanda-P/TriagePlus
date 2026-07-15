export async function postJSON<T = unknown>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export async function getJSON<T = unknown>(url: string): Promise<T> {
  const res = await fetch(url, { method: 'GET', headers: { 'Accept': 'application/json' } });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export function buildWsUrl(sessionId: string): string {
  if (import.meta.env.VITE_WS_BASE_URL) {
    return `${import.meta.env.VITE_WS_BASE_URL}/api/v1/ws/chat/${sessionId}`;
  }
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/api/v1/ws/chat/${sessionId}`;
}

export function buildApiUrl(path: string): string {
  const base = import.meta.env.VITE_API_BASE_URL ?? '';
  return `${base}${path}`;
}
