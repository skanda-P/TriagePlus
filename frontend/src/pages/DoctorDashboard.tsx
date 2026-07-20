import { useEffect, useState, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Users, Clock, LogOut, RefreshCw } from 'lucide-react';
import { apiFetch } from '../utils/api';

interface QueueEntry {
  id: string;
  position: number;
  est_wait_min: number;
  appointment_date: string;
  appointment?: {
    id: string;
    status: string;
    triage_level: number;
    department: string;
    patient?: { name: string; age: number; gender: string };
  };
}

export default function DoctorDashboard() {
  const navigate = useNavigate();
  const [queue, setQueue] = useState<QueueEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const token = sessionStorage.getItem('doctor_token');
  const abortRef = useRef<AbortController | null>(null);

  const fetchQueue = useCallback(async () => {
    if (!token) {
      navigate('/doctor/login');
      return;
    }
    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    setError('');
    try {
      const data = await apiFetch<QueueEntry[]>('/api/v1/doctor/queue', {
        token,
        signal: controller.signal,
      });
      setQueue(data ?? []);
    } catch (e: any) {
      if (e.name === 'AbortError') return;
      if (e.message?.includes('401') || e.message?.includes('Authentication')) {
        sessionStorage.removeItem('doctor_token');
        navigate('/doctor/login');
        return;
      }
      setError('Could not load queue. Is the backend running?');
    } finally {
      setLoading(false);
    }
  }, [token, navigate]);

  useEffect(() => {
    fetchQueue();
    return () => {
      if (abortRef.current) abortRef.current.abort();
    };
  }, [fetchQueue]);

  return (
    <div className="min-h-screen transition-colors duration-300" style={{ backgroundColor: 'var(--color-lavender-mist)' }}>
      <nav className="sticky top-0 z-50 bg-white dark:bg-slate-800 border-b border-frost-gray dark:border-gray-800 transition-colors duration-300" style={{ boxShadow: 'var(--shadow-subtle)' }}>
        <div className="page-container flex items-center justify-between py-4">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center text-white font-bold text-sm" style={{ backgroundColor: 'var(--color-canopy-green)' }}>M</div>
            <span className="font-semibold text-base text-charcoal dark:text-white transition-colors">TriagePlus · Doctor Portal</span>
          </div>
          <div className="flex items-center gap-3">
            <button onClick={fetchQueue} className="btn-ghost-dark !py-2 !px-4 text-sm gap-2"><RefreshCw className="w-4 h-4" /> Refresh</button>
            <button onClick={() => { sessionStorage.removeItem('doctor_token'); navigate('/doctor/login'); }} className="btn-nav text-sm gap-1.5"><LogOut className="w-3.5 h-3.5" /> Sign Out</button>
          </div>
        </div>
      </nav>
      <main className="page-container py-8">
        <div className="flex items-center gap-3 mb-6">
          <Users className="w-6 h-6" style={{ color: 'var(--color-canopy-green)' }} />
          <h1 className="font-bold text-charcoal dark:text-white transition-colors" style={{ fontSize: '28px', letterSpacing: '-0.019em' }}>Today's Queue</h1>
          <span className="tag-chip bg-mint-wash text-canopy-green border border-leaf-bright ml-2">{queue.length} patients</span>
        </div>
        {loading && <div className="card-white dark:bg-slate-800 text-center py-12 text-ash transition-colors duration-300">Loading queue…</div>}
        {!loading && error && <div className="bubble-emergency">{error}</div>}
        {!loading && !error && queue.length === 0 && <div className="card-white dark:bg-slate-800 text-center py-12 transition-colors duration-300"><p className="text-base text-slate-muted dark:text-ash">No patients in queue for today.</p></div>}
        <div className="flex flex-col gap-4">
          {queue.map((entry) => (
            <div key={entry.id} className="card-white dark:bg-slate-800 dark:border-gray-700 flex items-start gap-4 transition-colors duration-300">
              <div className="w-12 h-12 rounded-full flex items-center justify-center text-white font-bold text-lg flex-shrink-0" style={{ backgroundColor: 'var(--color-canopy-green)' }}>{entry.position}</div>
              <div className="flex-1">
                <div className="flex flex-wrap items-center gap-3 mb-2">
                  <p className="font-semibold text-charcoal dark:text-white text-lg transition-colors">{entry.appointment?.patient?.name ?? 'Unknown'}</p>
                  {entry.appointment?.department && <span className="tag-chip bg-sky-wash text-deep-teal">{entry.appointment.department}</span>}
                  <span className="tag-chip bg-cream text-graphite flex items-center gap-1"><Clock className="w-3 h-3" /> ~{entry.est_wait_min}m</span>
                  {entry.appointment?.triage_level && (
                    <span className={`tag-chip ${entry.appointment.triage_level <= 2 ? 'bg-red-50 text-red-700' : 'bg-mint-wash text-canopy-green'}`}>Triage {entry.appointment.triage_level}</span>
                  )}
                </div>
                {entry.appointment?.triage_level !== undefined && (
                  <div className="rounded-cards-sm p-3 text-sm text-graphite dark:text-slate-300 transition-colors" style={{ backgroundColor: 'var(--color-lavender-mist)' }}>
                    <p className="font-semibold text-indigo-bloom text-xs uppercase tracking-wider mb-1">Patient</p>
                    <p>Age {entry.appointment.patient?.age ?? '—'}, {entry.appointment.patient?.gender ?? '—'} · Status: {entry.appointment.status}</p>
                  </div>
                )}
              </div>
              <div className="flex flex-col gap-2 flex-shrink-0">
                <button className="btn-coral !py-2 !px-4 text-sm" onClick={() => alert('Start consultation — endpoint not yet wired.')}>Start</button>
                <button className="btn-ghost-dark !py-2 !px-4 text-sm" onClick={() => alert('End consultation — endpoint not yet wired.')}>End</button>
              </div>
            </div>
          ))}
        </div>
      </main>
    </div>
  );
}