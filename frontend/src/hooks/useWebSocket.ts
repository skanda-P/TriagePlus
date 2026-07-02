import { useCallback, useEffect, useRef, useState } from 'react';
import { useChatStore } from '../stores/chatStore';
import { buildWsUrl } from '../utils/api';

export type WsStatus = 'connecting' | 'open' | 'reconnecting' | 'closed';

export function useWebSocket(sessionId: string) {
  const [status, setStatus] = useState<WsStatus>('connecting');
  const wsRef      = useRef<WebSocket | null>(null);
  const attemptRef = useRef(0);
  const mountedRef = useRef(true);

  const addMessage     = useChatStore((s) => s.addMessage);
  const appendMessageChunk = useChatStore((s) => s.appendMessageChunk);
  const setFsmState    = useChatStore((s) => s.setFsmState);
  const setSessionMeta = useChatStore((s) => s.setSessionMeta);
  const setIsTyping    = useChatStore((s) => s.setIsTyping);

  const connect = useCallback(() => {
    if (!mountedRef.current) return;
    setStatus('connecting');
    const ws = new WebSocket(buildWsUrl(sessionId));
    wsRef.current = ws;

    ws.onopen = () => {
      if (!mountedRef.current) { ws.close(); return; }
      setStatus('open');
      attemptRef.current = 0;
    };

    ws.onmessage = (evt) => {
      if (!mountedRef.current) return;
      try {
        const data = JSON.parse(evt.data as string);
        if (data.type === 'ping') { ws.send(JSON.stringify({ type: 'pong' })); return; }
        setIsTyping(false);
        if (data.state) setFsmState({ current: data.state });
        if (data.meta) {
          setSessionMeta({
            specialty:       data.meta.specialty,
            confidence:      data.meta.confidence,
            confidenceLabel: data.meta.confidence_label,
            urgency:         data.meta.urgency,
            triageLevel:     data.meta.triage_level,
            triageColor:     data.meta.triage_color,
          });
        }
        if (data.type === 'emergency') {
          addMessage({ role: 'emergency', content: data.content });
        } else if (data.type === 'error') {
          addMessage({ role: 'error', content: data.content });
        } else if (data.type === 'message' && data.content) {
          addMessage({ role: 'assistant', content: data.content });
        } else if (data.type === 'typing') {
          setIsTyping(data.content);
        } else if (data.type === 'stream_start') {
          setIsTyping(false); // Hide typing indicator when stream starts
        } else if (data.type === 'stream_chunk') {
          setIsTyping(false);
          appendMessageChunk('assistant', data.content);
        }
      } catch { /* non-JSON frame */ }
    };

    ws.onclose = () => {
      if (!mountedRef.current) return;
      setStatus('reconnecting');
      const delay = Math.min(1000 * Math.pow(2, attemptRef.current++), 30_000);
      setTimeout(connect, delay);
    };

    ws.onerror = () => ws.close();
  }, [sessionId, addMessage, setFsmState, setSessionMeta, setIsTyping]);

  useEffect(() => {
    mountedRef.current = true;
    connect();
    return () => { mountedRef.current = false; wsRef.current?.close(); };
  }, [connect]);

  const send = useCallback((msg: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  }, []);

  return { status, send };
}
