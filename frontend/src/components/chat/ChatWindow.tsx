import React, { useEffect, useRef, useState } from 'react';
import { Send, Wifi, WifiOff, Loader2 } from 'lucide-react';
import { useChatStore } from '../../stores/chatStore';
import { useWebSocket, type WsStatus } from '../../hooks/useWebSocket';
import { MessageBubble } from './MessageBubble';
import { TypingDots } from '../shared/Spinner';
import { TriageBadge } from '../shared/Badge';
import { StethoscopeToggle } from '../shared/StethoscopeToggle';

function WsStatusPill({ status }: { status: WsStatus }) {
  const map: Record<WsStatus, { label: string; cls: string; icon: React.ReactNode }> = {
    open:         { label: 'Connected',    cls: 'bg-mint-wash text-canopy-green border-leaf-bright',  icon: <Wifi className="w-3 h-3" /> },
    connecting:   { label: 'Connecting…',  cls: 'bg-cream text-graphite border-ash',                 icon: <Loader2 className="w-3 h-3 animate-spin" /> },
    reconnecting: { label: 'Reconnecting', cls: 'bg-peach-wash text-coral-pulse border-coral-pulse', icon: <Loader2 className="w-3 h-3 animate-spin" /> },
    closed:       { label: 'Disconnected', cls: 'bg-red-50 text-red-700 border-red-200',             icon: <WifiOff className="w-3 h-3" /> },
  };
  const { label, cls, icon } = map[status];
  return <span className={`tag-chip border flex items-center gap-1 ${cls}`}>{icon} {label}</span>;
}

function DisclaimerBanner() {
  return (
    <div className="px-4 py-3 bg-cream border-t border-frost-gray">
      <p className="text-xs text-slate-muted text-center leading-relaxed">
        ⚠️ This information is general in nature and does not constitute a medical diagnosis. Always consult a qualified medical professional.
      </p>
    </div>
  );
}

export function ChatWindow({ sessionId }: { sessionId: string }) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef       = useRef<HTMLInputElement>(null);

  const messages    = useChatStore((s) => s.messages);
  const isTyping    = useChatStore((s) => s.isTyping);
  const sessionMeta = useChatStore((s) => s.sessionMeta);
  const fsmState    = useChatStore((s) => s.fsmState);
  const setIsTyping = useChatStore((s) => s.setIsTyping);

  const { status, send } = useWebSocket(sessionId);

  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, isTyping]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const content = input.trim();
    if (!content || status !== 'open') return;
    useChatStore.getState().addMessage({ role: 'patient', content });
    setInput('');
    setIsTyping(true);
    send({ type: 'message', content });
    inputRef.current?.focus();
  };

  const hasAiResponse = messages.some((m) => m.role === 'assistant');
  const isEmergency   = fsmState.current === 'EMERGENCY';

  return (
    <div className="flex flex-col h-full bg-white dark:bg-slate-900 rounded-cards overflow-hidden border border-frost-gray dark:border-gray-800 transition-colors duration-300" style={{ boxShadow: 'var(--shadow-subtle)' }}>
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-frost-gray dark:border-gray-800 bg-lavender-mist dark:bg-slate-800 transition-colors duration-300">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-full bg-canopy-green flex items-center justify-center text-white font-bold text-sm select-none">M</div>
          <div>
            <p className="font-semibold text-charcoal dark:text-white text-sm leading-none transition-colors">TriagePlus AI</p>
            <p className="text-xs text-slate-muted dark:text-ash mt-0.5">Triage assistant</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <StethoscopeToggle />
          <WsStatusPill status={status} />
        </div>
      </div>

      {/* Triage result card */}
      {sessionMeta.specialty && (
        <div className="mx-4 mt-4 card-pastel" style={{ backgroundColor: 'var(--color-mint-wash)' }}>
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <p className="text-xs text-graphite font-medium uppercase tracking-wider mb-1">Recommended</p>
              <p className="text-2xl font-bold text-canopy-green">{sessionMeta.specialty}</p>
              <div className="flex gap-4 mt-1">
                <p className="text-sm text-graphite">Confidence: <strong>{sessionMeta.confidenceLabel ?? `${Math.round(sessionMeta.confidence ?? 0)}%`}</strong></p>
                {sessionMeta.urgency !== undefined && (
                  <p className="text-sm text-graphite">Urgency: <strong>{sessionMeta.urgency}/5</strong></p>
                )}
              </div>
            </div>
            {sessionMeta.triageLevel && <TriageBadge level={sessionMeta.triageLevel} />}
          </div>
        </div>
      )}

      {/* Messages */}
      <div id="chat-messages" className="flex-1 overflow-y-auto chat-scroll px-4 py-4 flex flex-col gap-3" role="log" aria-live="polite">
        {messages.length === 0 && <div className="flex-1 flex items-center justify-center text-ash text-sm"><p>Connecting to TriagePlus AI…</p></div>}
        {messages.map((msg) => <MessageBubble key={msg.id} message={msg} />)}
        {isTyping && <div className="flex justify-start"><div className="bubble-ai"><TypingDots /></div></div>}
        <div ref={messagesEndRef} />
      </div>

      {hasAiResponse && <DisclaimerBanner />}

      {/* Input */}
      {!isEmergency && (
        <form id="msg-form" onSubmit={handleSubmit} className="flex items-center gap-2 px-4 py-3 border-t border-frost-gray dark:border-gray-800 bg-white dark:bg-slate-900 transition-colors duration-300">
          <input
            id="msg-input" ref={inputRef} type="text" value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={status === 'open' ? 'Type your message…' : 'Connecting…'}
            disabled={status !== 'open'} autoComplete="off"
            className="flex-1 min-h-11 rounded-full bg-lavender-mist dark:bg-slate-800 px-5 text-base text-charcoal dark:text-white placeholder:text-ash outline-none focus:ring-2 focus:ring-canopy-green/20 dark:focus:ring-sky-signal/50 transition-all disabled:opacity-60"
          />
          <button id="send-btn" type="submit" disabled={!input.trim() || status !== 'open'}
            className="btn-coral w-11 h-11 !p-0 !rounded-full flex-shrink-0 disabled:opacity-50" aria-label="Send">
            <Send className="w-4 h-4" />
          </button>
        </form>
      )}
    </div>
  );
}
