import React, { Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Spinner } from './components/shared/Spinner';
import { StethoscopeToggle } from './components/shared/StethoscopeToggle';

import Landing from './pages/Landing';
import Chat from './pages/Chat';

const DoctorLogin     = lazy(() => import('./pages/DoctorLogin'));
const DoctorDashboard = lazy(() => import('./pages/DoctorDashboard'));
const RagMonitor      = lazy(() => import('./pages/RagMonitor'));

function App() {
  return (
    <Router>
      <Suspense
        fallback={
          <div className="min-h-screen flex items-center justify-center bg-lavender-mist dark:bg-charcoal transition-colors duration-300">
            <Spinner size={32} className="text-canopy-green dark:text-sky-signal" />
          </div>
        }
      >
        <StethoscopeToggle />
        <Routes>
          <Route path="/"                  element={<Landing />} />
          <Route path="/chat"              element={<Chat />} />
          <Route path="/doctor/login"      element={<DoctorLogin />} />
          <Route path="/doctor/dashboard"  element={<DoctorDashboard />} />
          <Route path="/diagnostics"       element={<RagMonitor />} />
        </Routes>
      </Suspense>
    </Router>
  );
}

export default App;
