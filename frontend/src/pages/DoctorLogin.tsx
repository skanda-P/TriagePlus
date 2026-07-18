import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { LogIn, Stethoscope } from 'lucide-react';

import { supabase } from '../utils/supabase';

export default function DoctorLogin() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    try {
      const { data, error: authError } = await supabase.auth.signInWithPassword({
        email,
        password,
      });
      
      if (authError) {
        setError(authError.message);
        return;
      }
      
      if (data.session) {
        sessionStorage.setItem('doctor_token', data.session.access_token);
        navigate('/doctor/dashboard');
      }
    } catch {
      setError('Could not connect to the server.');
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center" style={{ backgroundColor: 'var(--color-lavender-mist)' }}>
      <div className="card-white w-full max-w-md">
        <div className="flex flex-col items-center mb-8">
          <div className="w-14 h-14 rounded-cards-sm flex items-center justify-center text-white mb-4" style={{ backgroundColor: 'var(--color-canopy-green)' }}>
            <Stethoscope className="w-7 h-7" />
          </div>
          <h1 className="font-bold text-charcoal" style={{ fontSize: '28px', letterSpacing: '-0.019em' }}>Doctor Portal</h1>
          <p className="text-sm text-slate-muted mt-1">Sign in with your hospital credentials</p>
        </div>
        <form id="doctor-login-form" onSubmit={handleLogin} className="flex flex-col gap-4">
          <div>
            <label htmlFor="doctor-email" className="block text-sm font-medium text-charcoal mb-1">Email</label>
            <input id="doctor-email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="doctor@hospital.com" required className="w-full min-h-11 rounded-cards-sm border border-frost-gray bg-cloud-gray px-4 text-base text-charcoal outline-none focus:ring-2 transition-all" />
          </div>
          <div>
            <label htmlFor="doctor-password" className="block text-sm font-medium text-charcoal mb-1">Password</label>
            <input id="doctor-password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="••••••••" required className="w-full min-h-11 rounded-cards-sm border border-frost-gray bg-cloud-gray px-4 text-base text-charcoal outline-none focus:ring-2 transition-all" />
          </div>
          {error && <p className="text-sm text-red-700 bg-red-50 border border-red-200 rounded-cards-sm px-4 py-3">{error}</p>}
          <button id="doctor-login-btn" type="submit" className="btn-coral w-full mt-2 gap-2"><LogIn className="w-4 h-4" /> Sign In</button>
        </form>
        <p className="text-center text-xs text-ash mt-6">
          <a href="/" className="underline" style={{ color: 'var(--color-canopy-green)' }}>← Back to patient triage</a>
        </p>
      </div>
    </div>
  );
}
