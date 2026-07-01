import { useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { ArrowLeft, LogIn } from 'lucide-react';
import { useSession } from '../hooks/useSession';
import { useChatStore } from '../stores/chatStore';
import { ChatWindow } from '../components/chat/ChatWindow';

export default function Chat() {
  const navigate = useNavigate();
  const sessionId = useSession();
  const clearMessages = useChatStore((s) => s.clearMessages);

  useEffect(() => {
    if (!sessionStorage.getItem('userName')) navigate('/');
  }, [navigate]);

  const userName = sessionStorage.getItem('userName') ?? 'Patient';

  const handleNewSession = () => {
    clearMessages();
    sessionStorage.removeItem('triageplus_session_id');
    sessionStorage.removeItem('userName');
    navigate('/');
  };

  return (
    <div className="min-h-screen flex flex-col bg-lavender-mist dark:bg-slate-900 transition-colors duration-300">
      <nav className="sticky top-0 z-50 bg-white dark:bg-slate-800 border-b border-frost-gray dark:border-gray-800 transition-colors duration-300" style={{ boxShadow: 'var(--shadow-subtle)' }}>
        <div className="page-container flex items-center justify-between py-3">
          <button onClick={handleNewSession} className="flex items-center gap-2 text-sm font-medium text-graphite hover:text-canopy-green transition-colors" aria-label="New session">
            <ArrowLeft className="w-4 h-4" /> New Session
          </button>
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-full flex items-center justify-center text-white font-bold text-xs" style={{ backgroundColor: 'var(--color-canopy-green)' }}>M</div>
            <span className="font-semibold text-base text-charcoal dark:text-white transition-colors">TriagePlus</span>
          </div>
          <div className="flex items-center gap-3">
            <span className="hidden sm:inline text-sm text-slate-muted dark:text-ash transition-colors">👤 {userName}</span>
            <Link to="/doctor/login" className="btn-nav flex items-center gap-1.5">
              <LogIn className="w-3.5 h-3.5" /> Doctor
            </Link>
          </div>
        </div>
      </nav>
      <main className="flex-1 page-container py-6">
        <div className="max-w-3xl mx-auto h-full" style={{ minHeight: 'calc(100vh - 140px)' }}>
          <ChatWindow sessionId={sessionId} />
        </div>
      </main>
    </div>
  );
}
