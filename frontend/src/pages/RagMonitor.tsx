import React, { useEffect, useState, useRef } from 'react';
import { Link } from 'react-router-dom';
import { Database, Search, ArrowRight, Zap, RefreshCw, ChevronDown, ChevronRight, Activity, Clock, Lock } from 'lucide-react';
import { buildWsUrl } from '../utils/api';

interface DiagnosticEvent {
  node: string;
  state: {
    session_id: string;
    present_symptoms: string[];
    absent_symptoms: string[];
    is_emergency: boolean;
    final_diagnosis: string | null;
    department: string | null;
    confidence: number;
    urgency: number;
    rag_chunks: string[];
    latencies: Record<string, number>;
  };
}

const Collapsible = ({ title, children, defaultOpen = false }: { title: React.ReactNode, children: React.ReactNode, defaultOpen?: boolean }) => {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border border-frost-gray rounded-cards-sm overflow-hidden bg-white mb-3">
      <button 
        onClick={() => setOpen(!open)} 
        className="w-full flex items-center justify-between p-3 bg-cloud-gray hover:bg-slate-100 transition-colors text-left"
      >
        <span className="font-semibold text-sm text-charcoal flex items-center gap-2">{title}</span>
        {open ? <ChevronDown className="w-4 h-4 text-ash" /> : <ChevronRight className="w-4 h-4 text-ash" />}
      </button>
      {open && <div className="p-4 border-t border-frost-gray">{children}</div>}
    </div>
  );
};

export default function RagMonitor() {
  const [events, setEvents] = useState<DiagnosticEvent[]>([]);
  const [status, setStatus] = useState<'connecting' | 'connected' | 'disconnected'>('disconnected');
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [passwordInput, setPasswordInput] = useState('');
  
  const wsRef = useRef<WebSocket | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const connect = (pass?: string) => {
    const token = pass || sessionStorage.getItem('developer_password');
    if (!token) {
      setStatus('disconnected');
      return;
    }

    setStatus('connecting');
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const baseUrl = import.meta.env.VITE_WS_BASE_URL 
      ? `${import.meta.env.VITE_WS_BASE_URL}/api/v1/ws/diagnostics`
      : `${protocol}//${host}/api/v1/ws/diagnostics`;
    const wsUrl = `${baseUrl}?token=${encodeURIComponent(token)}`;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus('connected');
      setIsAuthenticated(true);
      sessionStorage.setItem('developer_password', token);
    };
    ws.onclose = (e) => {
      setStatus('disconnected');
      if (e.code === 1008) {
        setIsAuthenticated(false);
        sessionStorage.removeItem('developer_password');
      }
    };
    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.type === 'diagnostic_update') {
          setEvents((prev) => [...prev, data]);
          setTimeout(() => {
            if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
          }, 100);
        }
      } catch (err) {
        console.error('Failed to parse diagnostic event', err);
      }
    };
  };

  useEffect(() => {
    const savedPass = sessionStorage.getItem('developer_password');
    if (savedPass) {
      connect(savedPass);
    }
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    if (passwordInput) {
      connect(passwordInput);
    }
  };

  if (!isAuthenticated && status !== 'connected') {
    return (
      <div className="min-h-screen flex items-center justify-center bg-lavender-mist dark:bg-slate-900 transition-colors duration-300">
        <div className="card-white dark:bg-slate-800 dark:border-gray-700 max-w-md w-full mx-4">
          <div className="flex flex-col items-center mb-6">
            <div className="w-12 h-12 rounded-lg flex items-center justify-center text-white mb-4" style={{ backgroundColor: 'var(--color-indigo-bloom)' }}>
              <Lock className="w-6 h-6" />
            </div>
            <h1 className="text-xl font-bold text-charcoal dark:text-white">Developer Access</h1>
            <p className="text-sm text-slate-muted text-center mt-2">Enter the developer password to view real-time diagnostics.</p>
          </div>
          <form onSubmit={handleLogin} className="flex flex-col gap-4">
            <input 
              type="password" 
              value={passwordInput}
              onChange={(e) => setPasswordInput(e.target.value)}
              placeholder="Developer Password" 
              className="input-field dark:bg-slate-900 dark:border-gray-700 dark:text-white"
              autoFocus
            />
            {status === 'disconnected' && passwordInput && (
              <p className="text-red-500 text-xs text-center">Connection failed or invalid password.</p>
            )}
            <button type="submit" className="btn-primary w-full flex justify-center items-center gap-2" disabled={status === 'connecting'}>
              {status === 'connecting' ? <RefreshCw className="w-4 h-4 animate-spin" /> : 'Connect'}
            </button>
            <Link to="/" className="text-center text-sm text-slate-muted hover:text-charcoal mt-2">Return Home</Link>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col bg-lavender-mist dark:bg-slate-900 transition-colors duration-300">
      <nav className="sticky top-0 z-50 bg-white dark:bg-slate-800 border-b border-frost-gray dark:border-gray-800 shadow-sm transition-colors duration-300">
        <div className="page-container flex items-center justify-between py-4">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center text-white font-bold text-sm" style={{ backgroundColor: 'var(--color-indigo-bloom)' }}>
              <Activity className="w-5 h-5" />
            </div>
            <span className="font-semibold text-base text-charcoal dark:text-white transition-colors">LangGraph & RAG Monitor</span>
          </div>
          <div className="flex items-center gap-4">
            <span className={`text-xs font-semibold px-2 py-1 rounded ${status === 'connected' ? 'bg-green-100 text-green-700' : status === 'connecting' ? 'bg-yellow-100 text-yellow-700' : 'bg-red-100 text-red-700'}`}>
              {status.toUpperCase()}
            </span>
            {status === 'disconnected' && (
              <button onClick={() => connect()} className="btn-ghost-dark !py-1.5 !px-3 text-xs gap-1"><RefreshCw className="w-3.5 h-3.5" /> Reconnect</button>
            )}
            <Link to="/" className="text-sm font-medium text-indigo-bloom hover:underline">Exit</Link>
          </div>
        </div>
      </nav>

      <main className="page-container py-6 flex-1 flex flex-col max-w-5xl mx-auto">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-xl font-bold text-charcoal dark:text-white transition-colors">Live Inference Feed</h1>
          <button onClick={() => setEvents([])} className="text-xs text-ash hover:text-charcoal dark:hover:text-white transition-colors">Clear Logs</button>
        </div>

        <div ref={scrollRef} className="flex-1 overflow-y-auto flex flex-col gap-6 pb-20">
          {events.length === 0 && (
            <div className="text-center py-20 text-slate-muted flex flex-col items-center">
              <Search className="w-12 h-12 mb-4 opacity-20" />
              <p>Waiting for LangGraph inference events...</p>
              <p className="text-xs mt-2">Open the Patient Chat in another window and send a message.</p>
            </div>
          )}

          {events.map((ev, i) => {
            const s = ev.state || {};
            const chunks = s.rag_chunks || [];
            const lat = s.latencies || {};

            return (
            <div key={i} className="card-white dark:bg-slate-800 dark:border-gray-700 border-l-4 transition-colors duration-300" style={{ borderLeftColor: 'var(--color-indigo-bloom)' }}>
              <div className="flex justify-between items-start mb-4">
                <div>
                  <span className="text-xs font-mono text-white bg-indigo-bloom dark:bg-sky-signal px-2 py-1 rounded">Node: {ev.node}</span>
                  <div className="mt-3">
                    <h4 className="text-xs font-bold text-slate-muted uppercase mb-1">Extracted Symptoms (E_Codes)</h4>
                    <p className="font-mono text-sm">{s.present_symptoms?.length ? s.present_symptoms.join(', ') : 'None'}</p>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-xs text-slate-muted uppercase tracking-wider mb-1">State Prediction</div>
                  <div className="font-bold text-indigo-bloom">{s.final_diagnosis || 'Pending...'}</div>
                  {s.confidence > 0 && <div className="text-xs text-canopy-green font-mono">{s.confidence}% confidence</div>}
                </div>
              </div>

              {chunks.length > 0 && (
                <Collapsible title={<><Database className="w-4 h-4" /> Retrieved RAG Chunks ({chunks.length})</>}>
                  <div className="flex flex-col gap-3">
                    {chunks.map((chunk, j) => (
                      <div key={j} className="bg-cloud-gray p-3 rounded-md text-sm">
                        <p className="text-graphite font-mono text-xs whitespace-pre-wrap">{chunk}</p>
                      </div>
                    ))}
                  </div>
                </Collapsible>
              )}

              {Object.keys(lat).length > 0 && (
                <div className="bg-slate-50 dark:bg-slate-900 border border-frost-gray dark:border-gray-700 rounded-md p-3 mt-3 flex items-center justify-start gap-6 text-xs font-mono text-slate-muted transition-colors">
                  <Clock className="w-4 h-4 text-ash" />
                  {Object.entries(lat).map(([k, v]) => (
                    <span key={k}><strong className="text-charcoal dark:text-white">{k}:</strong> {v}ms</span>
                  ))}
                </div>
              )}
            </div>
          )})}
        </div>
      </main>
    </div>
  );
}
