import { create } from 'zustand';

const generateId = () => (typeof crypto !== 'undefined' && crypto.randomUUID) ? crypto.randomUUID() : Math.random().toString(36).substring(2, 15);

export type MessageRole = 'patient' | 'assistant' | 'emergency' | 'error';

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: number;
  chips?: string[];
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
  appendMessageChunk: (role: MessageRole, chunk: string) => void;
  setFsmState:     (state: FsmState) => void;
  setSessionMeta:  (meta: Partial<SessionMeta>) => void;
  setIsTyping:     (v: boolean) => void;
  setEmergencyClosed: (v: boolean) => void;
  clearMessages:   () => void;
  replaceMessages: (items: Array<{ role: string; content: string }>) => void;
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
    return { messages: [...s.messages, { ...msg, id: generateId(), timestamp: Date.now() }] };
  }),
  appendMessageChunk: (role, chunk) => set((s) => {
    const lastMsg = s.messages[s.messages.length - 1];
    if (lastMsg && lastMsg.role === role) {
      const updatedMessages = [...s.messages];
      updatedMessages[updatedMessages.length - 1] = {
        ...lastMsg,
        content: lastMsg.content + chunk
      };
      return { messages: updatedMessages };
    }
    return { messages: [...s.messages, { id: generateId(), role, content: chunk, timestamp: Date.now() }] };
  }),
  setFsmState:     (fsmState)  => set({ fsmState }),
  setSessionMeta:  (meta)      => set((s) => ({ sessionMeta: { ...s.sessionMeta, ...meta } })),
  setIsTyping:     (isTyping)  => set({ isTyping }),
  setEmergencyClosed: (v)      => set({ emergencyClosed: v }),
  clearMessages:   ()          => set({ messages: [], fsmState: { current: 'NAME_ENTRY' }, sessionMeta: {} }),
  replaceMessages: (items)     => set(() => {
    const mapped: Message[] = [];
    for (const item of items) {
      let role: MessageRole | null = null;
      if (item.role === 'assistant') role = 'assistant';
      if (item.role === 'user') role = 'patient';
      if (!role || !item.content) continue;
      mapped.push({
        id: generateId(),
        role,
        content: item.content,
        timestamp: Date.now(),
      });
    }
    return { messages: mapped };
  }),
}));
