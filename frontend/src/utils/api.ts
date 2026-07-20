export function buildWsUrl(sessionId: string): string {
  if (import.meta.env.VITE_WS_BASE_URL) {
    return `${import.meta.env.VITE_WS_BASE_URL}/api/v1/ws/chat/${sessionId}`;
  }
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/api/v1/ws/chat/${sessionId}`;
}

export function buildDiagnosticsWsUrl(token: string): string {
  if (import.meta.env.VITE_WS_BASE_URL) {
    return `${import.meta.env.VITE_WS_BASE_URL}/api/v1/ws/diagnostics?token=${encodeURIComponent(token)}`;
  }
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/api/v1/ws/diagnostics?token=${encodeURIComponent(token)}`;
}

export function buildApiUrl(path: string): string {
  const base = import.meta.env.VITE_API_BASE_URL ?? '';
  return `${base}${path}`;
}

/**
 * Wrapper around fetch that:
 * - Adds Authorization header if token is provided
 * - Supports AbortController for cancellation
 * - Throws on non-OK responses
 */
export async function apiFetch<T = unknown>(
  path: string,
  options: RequestInit & { token?: string; signal?: AbortSignal } = {}
): Promise<T> {
  const { token, signal, headers, ...rest } = options;
  const url = buildApiUrl(path);
  
  const requestHeaders: HeadersInit = {
    'Content-Type': 'application/json',
    Accept: 'application/json',
    ...headers,
  };
  
  if (token) {
    (requestHeaders as Record<string, string>)['Authorization'] = `Bearer ${token}`;
  }
  
  const res = await fetch(url, {
    ...rest,
    headers: requestHeaders,
    signal,
  });
  
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }
  
  if (res.status === 204) {
    return undefined as T;
  }
  
  return res.json() as Promise<T>;
}