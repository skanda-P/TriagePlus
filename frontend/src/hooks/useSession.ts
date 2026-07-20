import { useState, useEffect } from 'react';

export function useSession(): string {
  const [sessionId, setSessionId] = useState<string>(() => {
    const existing = sessionStorage.getItem('triageplus_session_id');
    if (existing) return existing;
    const newId = crypto.randomUUID();
    sessionStorage.setItem('triageplus_session_id', newId);
    return newId;
  });

  // Re-read sessionStorage on focus (in case it was changed by another tab or "New Session")
  useEffect(() => {
    const handleFocus = () => {
      const current = sessionStorage.getItem('triageplus_session_id');
      if (current && current !== sessionId) {
        setSessionId(current);
      }
    };
    window.addEventListener('focus', handleFocus);
    return () => window.removeEventListener('focus', handleFocus);
  }, [sessionId]);

  return sessionId;
}