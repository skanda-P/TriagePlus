import { useState } from 'react';

export function useSession(): string {
  const [sessionId] = useState<string>(() => {
    const existing = sessionStorage.getItem('triageplus_session_id');
    if (existing) return existing;
    const newId = crypto.randomUUID();
    sessionStorage.setItem('triageplus_session_id', newId);
    return newId;
  });
  return sessionId;
}
