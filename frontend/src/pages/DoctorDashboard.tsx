import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Users, Clock, LogOut, RefreshCw } from 'lucide-react';

interface QueueEntry { position: number; patient_name: string; specialty: string; ai_brief: string; wait_time_minutes: number; }

export default function DoctorDashboard() {
  const navigate = useNavigate();
  const [queue, setQueue] = useState<QueueEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const token = sessionStorage.getItem('doctor_token');

  useEffect(() => { if (!token) { navigate('/doctor/login'); return; } fetchQueue(); }, [token, navigate]);

  const fetchQueue = async () => {
    setLoading(true); setError('');
    try {
      const res = await fetch('/api/v1/doctors/me/queue', { headers: { Authorization: `Bearer ${token}` } });
      if (res.status === 401) { sessionStorage.removeItem('doctor_token'); navigate('/doctor/login'); return; }
      setQueue(await res.json());
    } catch { setError('Could not load queue. Is the backend running?'); }
    finally { setLoading(false); }
  };

  return (
    <div className="min-h-screen" style={{ backgroundColor: 'var(--color-lavender-mist)' }}>
      <nav className="sticky top-0 z-50 bg-white border-b border-frost-gray" style={{ boxShadow: 'var(--shadow-subtle)' }}>
        <div className="page-container flex items-center justify-between py-4">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center text-white font-bold text-sm" style={{ backgroundColor: 'var(--color-canopy-green)' }}>M</div>
            <span className="font-semibold text-base text-charcoal">TriagePlus · Doctor Portal</span>
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
          <h1 className="font-bold text-charcoal" style={{ fontSize: '28px', letterSpacing: '-0.019em' }}>Today's Queue</h1>
          <span className="tag-chip bg-mint-wash text-canopy-green border border-leaf-bright ml-2">{queue.length} patients</span>
        </div>
        {loading && <div className="card-white text-center py-12 text-ash">Loading queue…</div>}
        {!loading && error && <div className="bubble-emergency">{error}</div>}
        {!loading && !error && queue.length === 0 && <div className="card-white text-center py-12"><p className="text-base text-slate-muted">No patients in queue for today.</p></div>}
        <div className="flex flex-col gap-4">
          {queue.map((entry) => (
            <div key={entry.position} className="card-white flex items-start gap-4">
              <div className="w-12 h-12 rounded-full flex items-center justify-center text-white font-bold text-lg flex-shrink-0" style={{ backgroundColor: 'var(--color-canopy-green)' }}>{entry.position}</div>
              <div className="flex-1">
                <div className="flex flex-wrap items-center gap-3 mb-2">
                  <p className="font-semibold text-charcoal text-lg">{entry.patient_name}</p>
                  <span className="tag-chip bg-sky-wash text-deep-teal">{entry.specialty}</span>
                  <span className="tag-chip bg-cream text-graphite flex items-center gap-1"><Clock className="w-3 h-3" /> ~{entry.wait_time_minutes}m</span>
                </div>
                {entry.ai_brief && (
                  <div className="rounded-cards-sm p-3 text-sm text-graphite" style={{ backgroundColor: 'var(--color-lavender-mist)' }}>
                    <p className="font-semibold text-indigo-bloom text-xs uppercase tracking-wider mb-1">AI Brief</p>
                    <p>{entry.ai_brief}</p>
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
