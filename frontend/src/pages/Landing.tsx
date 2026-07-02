import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowRight, Stethoscope, Clock, Shield, Star } from 'lucide-react';
import { StethoscopeToggle } from '../components/shared/StethoscopeToggle';

const PASTEL_CARDS = [
  { bg: 'var(--color-mint-wash)',  icon: <Stethoscope className="w-7 h-7" />, title: 'Smart Triage', body: 'Describe your symptoms in plain language. Our AI routes you to the right specialist — fast.' },
  { bg: 'var(--color-sky-wash)',   icon: <Clock className="w-7 h-7" />,       title: 'Zero Wait',   body: 'Book a confirmed slot in seconds. No phone calls, no hold music, no paperwork.' },
  { bg: 'var(--color-lilac-wash)', icon: <Shield className="w-7 h-7" />,      title: 'Safe & Private', body: 'Your health data stays protected. We never share your conversation or personal details.' },
  { bg: 'var(--color-peach-wash)', icon: <Star className="w-7 h-7" />,        title: 'AI-Powered',  body: 'Powered by Gemini 2.5 Flash — the same frontier AI used in top healthcare research.' },
];

export default function Landing() {
  const navigate = useNavigate();

  const startChat = () => {
    navigate('/chat');
  };

  return (
    <div className="min-h-screen flex flex-col transition-colors duration-300 bg-sand dark:bg-charcoal">
      {/* Nav */}
      <nav className="sticky top-0 z-50 bg-white dark:bg-gray-900 border-b border-frost-gray dark:border-gray-800 shadow-sm transition-colors duration-300">
        <div className="page-container flex items-center justify-between py-4">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center text-white font-bold text-sm" style={{ backgroundColor: 'var(--color-indigo-bloom)' }}>MG</div>
            <span className="font-semibold text-lg text-charcoal dark:text-white transition-colors duration-300">TriagePlus</span>
          </div>
          <div className="flex items-center gap-3">
            <StethoscopeToggle />
            <a href="/diagnostics" className="text-xs font-mono font-bold text-indigo-bloom bg-indigo-50 px-2 py-1 rounded border border-indigo-200">Dev: RAG Monitor</a>
            <a href="/chat" className="text-sm font-medium" style={{ color: 'var(--color-canopy-green)' }}>Patient Chat</a>
            <a href="/doctor/login" className="btn-nav text-sm">Doctor Portal</a>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="w-full transition-colors duration-300 bg-gradient-to-br from-slate-50 to-slate-200 dark:from-slate-800 dark:to-slate-900" style={{ padding: '80px 0' }}>
        <div className="page-container">
          <div className="flex flex-col lg:flex-row items-center gap-12 lg:gap-16">
            <div className="flex-1 max-w-xl">
              <div className="inline-flex items-center tag-chip mb-6 bg-slate-200 text-deep-teal dark:bg-slate-700 dark:text-sky-signal transition-colors duration-300">
                ✦ AI-Powered Medical Triage
              </div>
              <h1 className="font-bold mb-6 leading-tight text-deep-teal dark:text-white transition-colors duration-300" style={{ fontSize: 'clamp(48px, 6vw, 80px)', letterSpacing: '-0.028em' }}>
                Your Health, <span className="text-indigo-bloom dark:text-sky-signal">Triaged</span> Instantly.
              </h1>
              <p className="text-lg mb-10 text-graphite dark:text-ash transition-colors duration-300" style={{ maxWidth: '460px', lineHeight: '1.6' }}>
                Describe your symptoms to our AI assistant. Get routed to the right specialist and book a confirmed appointment — all in under 2 minutes.
              </p>
              <div className="flex items-center gap-4">
                <button id="start-triage-btn" onClick={startChat} className="btn-coral flex-shrink-0 gap-2">
                  Start Triage <ArrowRight className="w-4 h-4" />
                </button>
              </div>
              <p className="mt-5 text-xs" style={{ color: 'var(--color-ash)' }}>🔒 Your data is private and never shared with third parties.</p>
            </div>

            {/* Mock phone */}
            <div className="flex-shrink-0 w-72 lg:w-80">
              <div className="rounded-cards overflow-hidden bg-deep-teal dark:bg-slate-950 transition-colors duration-300" style={{ boxShadow: 'var(--shadow-subtle)' }}>
                <div className="flex items-center justify-between px-4 py-3 text-xs text-white/70">
                  <span>9:41</span><span className="font-medium text-white">TriagePlus</span><span>●●●</span>
                </div>
                <div className="bg-white dark:bg-slate-900 px-4 pt-4 pb-6 flex flex-col gap-3 transition-colors duration-300">
                  <div className="bubble-ai text-sm">👋 Welcome to TriagePlus! What is your gender?</div>
                  <div className="bubble-user text-sm">Female</div>
                  <div className="bubble-ai text-sm">Got it. What is your phone number?</div>
                  <div className="bubble-user text-sm">I have chest tightness and shortness of breath since morning.</div>
                  <div className="bubble-ai text-sm">Based on your symptoms, I recommend <strong className="text-indigo-bloom dark:text-sky-signal">Cardiology</strong> (93% confidence). 🟢 Level 3 — Soon.</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Feature cards */}
      <section className="page-container py-20">
        <div className="text-center mb-12">
          <p className="tag-chip bg-lavender-mist dark:bg-indigo-900 text-indigo-bloom dark:text-sky-signal mb-4">Why TriagePlus?</p>
          <h2 className="font-bold text-charcoal dark:text-white" style={{ fontSize: '40px', letterSpacing: '-0.019em' }}>
            Healthcare, at the speed of <span className="text-leaf-bright dark:text-emerald-400">AI</span>.
          </h2>
        </div>
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {PASTEL_CARDS.map((card, i) => (
            <div key={i} className="card-pastel flex flex-col gap-4 hover:scale-[1.02] transition-transform cursor-default dark:bg-slate-800" style={{ backgroundColor: card.bg }}>
              <div className="w-12 h-12 rounded-icons flex items-center justify-center bg-canopy-green text-white dark:bg-sky-signal">{card.icon}</div>
              <h3 className="font-bold text-2xl text-canopy-green dark:text-white" style={{ letterSpacing: '-0.019em' }}>{card.title}</h3>
              <p className="text-base text-graphite dark:text-ash leading-relaxed">{card.body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA band */}
      <section className="w-full" style={{ background: 'linear-gradient(135deg, var(--color-indigo-bloom) 0%, var(--color-canopy-green) 100%)', padding: '64px 0' }}>
        <div className="page-container text-center">
          <h2 className="font-bold text-white mb-4" style={{ fontSize: '40px', letterSpacing: '-0.019em' }}>Ready to skip the waiting room?</h2>
          <p className="text-lg mb-8" style={{ color: 'var(--color-mint-wash)' }}>Join thousands of patients triaged in under 2 minutes.</p>
          <div className="flex items-center justify-center gap-4 flex-wrap">
            <button onClick={() => navigate('/chat')} className="btn-coral">Start Your Triage <ArrowRight className="w-4 h-4" /></button>
            <a href="/doctor/login" className="btn-ghost-white">Doctor Portal</a>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 border-t border-frost-gray dark:border-gray-800 bg-cloud-gray dark:bg-slate-900 transition-colors duration-300">
        <div className="page-container flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-sm text-slate-muted dark:text-ash">© 2025 TriagePlus · IIT Dharwad · Hardly Human</p>
          <p className="text-xs text-ash dark:text-slate-muted text-center">⚠️ TriagePlus is an AI tool. It does not replace professional medical advice.</p>
        </div>
      </footer>
    </div>
  );
}
