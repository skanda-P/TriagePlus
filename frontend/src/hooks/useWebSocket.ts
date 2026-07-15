import { useCallback, useEffect, useRef, useState } from 'react';
import { useChatStore } from '../stores/chatStore';
import { buildWsUrl } from '../utils/api';

export type WsStatus = 'connecting' | 'open' | 'reconnecting' | 'closed';

export function useWebSocket(sessionId: string) {
  const [status, setStatus] = useState<WsStatus>('connecting');
  const [isInitialConnect, setIsInitialConnect] = useState(true);
  const wsRef = useRef<WebSocket | null>(null);
  const attemptRef = useRef(0);
  const mountedRef = useRef(true);
  const sessionIdRef = useRef(sessionId);

  // Keep sessionId ref updated
  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);

  const addMessage = useChatStore((s) => s.addMessage);
  const appendMessageChunk = useChatStore((s) => s.appendMessageChunk);
  const setFsmState = useChatStore((s) => s.setFsmState);
  const setSessionMeta = useChatStore((s) => s.setSessionMeta);
  const setIsTyping = useChatStore((s) => s.setIsTyping);
  const replaceMessages = useChatStore((s) => s.replaceMessages);

  const connect = useCallback(() => {
    if (!mountedRef.current) return;

    const currentSessionId = sessionIdRef.current;
    if (!currentSessionId) {
      console.warn('[WebSocket] No sessionId available for WebSocket connection');
      return;
    }

    setStatus('connecting');
    const wsUrl = buildWsUrl(currentSessionId);
    console.log('[WebSocket] Connecting to:', wsUrl);

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('[WebSocket] Connected');
      if (!mountedRef.current) {
        ws.close();
        return;
      }
      setStatus('open');
      setIsInitialConnect(false);
      attemptRef.current = 0;
    };

    ws.onmessage = (evt) => {
      if (!mountedRef.current) return;
      try {
        const data = JSON.parse(evt.data as string);

        // Handle ping/pong
        if (data.type === 'ping') {
          ws.send(JSON.stringify({ type: 'pong' }));
          return;
        }

        setIsTyping(false);

        if (data.state) setFsmState({ current: data.state });

        if (data.meta) {
          setSessionMeta({
            specialty: data.meta.specialty,
            confidence: data.meta.confidence,
            confidenceLabel: data.meta.confidence_label,
            urgency: data.meta.urgency,
            triageLevel: data.meta.triage_level,
            triageColor: data.meta.triage_color,
          });
        }

        if (data.type === 'sync_history') {
          replaceMessages(Array.isArray(data.history) ? data.history : []);
        } else if (data.type === 'emergency') {
          addMessage({ role: 'emergency', content: data.content });
        } else if (data.type === 'error') {
          addMessage({ role: 'error', content: data.content });
        } else if (data.type === 'message' && data.content) {
          addMessage({ role: 'assistant', content: data.content, chips: data.chips });
        } else if (data.type === 'typing') {
          setIsTyping(data.content);
        } else if (data.type === 'stream_start') {
          setIsTyping(false);
        } else if (data.type === 'stream_chunk') {
          setIsTyping(false);
          appendMessageChunk('assistant', data.content);
        }
      } catch (err) {
        console.error('Error processing websocket message:', err);
      }
    };

    ws.onclose = (event) => {
      console.log('[WebSocket] Disconnected:', event.code, event.reason);
      if (!mountedRef.current || wsRef.current !== ws) return;
      setStatus('reconnecting');
      const delay = Math.min(1000 * Math.pow(2, attemptRef.current++), 30_000);
      setTimeout(connect, delay);
    };

    ws.onerror = (error) => {
      console.error('[WebSocket] Error:', error);
      ws.close();
    };
  }, [addMessage, setFsmState, setSessionMeta, setIsTyping, replaceMessages]);

  useEffect(() => {
    mountedRef.current = true;
    connect();
    return () => {
      mountedRef.current = false;
      wsRef.current?.close();
    };
  }, [connect]);

  const send = useCallback((msg: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    } else {
      console.warn('[WebSocket] Cannot send message, socket not open. State:', wsRef.current?.readyState);
    }
  }, []);

  return { status, send, isInitialConnect };
}