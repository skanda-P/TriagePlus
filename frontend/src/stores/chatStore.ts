import { create } from 'zustand';

export type MessageRole = 'patient' | 'assistant' | 'emergency' | 'error';

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: number;
}

export interface FsmState { current: string; }

export interface SessionMeta {
  specialty?: string;
  confidence?: number;
  confidenceLabel?: string;
  urgency?: number;
  triageLevel?: number;
  triageColor?: string;
}

interface ChatStore {
  messages:        Message[];
  fsmState:        FsmState;
  sessionMeta:     SessionMeta;
  isTyping:        boolean;
  emergencyClosed: boolean;
  addMessage:      (msg: Omit<Message, 'id' | 'timestamp'>) => void;
  setFsmState:     (state: FsmState) => void;
  setSessionMeta:  (meta: Partial<SessionMeta>) => void;
  setIsTyping:     (v: boolean) => void;
  setEmergencyClosed: (v: boolean) => void;
  clearMessages:   () => void;
}

export const useChatStore = create<ChatStore>((set) => ({
  messages:        [],
  fsmState:        { current: 'NAME_ENTRY' },
  sessionMeta:     {},
  isTyping:        false,
  emergencyClosed: false,
  addMessage: (msg) => set((s) => {
    const lastMsg = s.messages[s.messages.length - 1];
    if (lastMsg && lastMsg.role === msg.role && lastMsg.content === msg.content) {
      return {}; // Ignore duplicate consecutive messages
    }
    return { messages: [...s.messages, { ...msg, id: crypto.randomUUID(), timestamp: Date.now() }] };
  }),
  setFsmState:     (fsmState)  => set({ fsmState }),
  setSessionMeta:  (meta)      => set((s) => ({ sessionMeta: { ...s.sessionMeta, ...meta } })),
  setIsTyping:     (isTyping)  => set({ isTyping }),
  setEmergencyClosed: (v)      => set({ emergencyClosed: v }),
  clearMessages:   ()          => set({ messages: [], fsmState: { current: 'NAME_ENTRY' }, sessionMeta: {} }),
}));
