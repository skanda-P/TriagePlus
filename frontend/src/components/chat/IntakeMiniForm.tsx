import React, { useState } from 'react';

export interface IntakeData {
  name: string;
  age: string;
  gender: string;
  contact: string;
}

export function IntakeMiniForm({ onSubmit }: { onSubmit: (data: IntakeData) => void }) {
  const [formData, setFormData] = useState<IntakeData>({
    name: '',
    age: '',
    gender: 'M',
    contact: ''
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <div className="w-full max-w-sm mx-auto bg-white dark:bg-slate-800 rounded-lg p-5 border border-frost-gray dark:border-gray-700 mt-4 shadow-sm">
      <h3 className="text-lg font-semibold text-charcoal dark:text-white mb-1">Welcome to TriagePlus</h3>
      <p className="text-xs text-slate-muted dark:text-ash mb-4">Please provide a few details to get started.</p>

      <form onSubmit={handleSubmit} className="space-y-3">
        <div>
          <label htmlFor="intake-name" className="block text-xs font-medium text-graphite dark:text-gray-300 mb-1">Full Name</label>
          <input
            id="intake-name"
            required type="text"
            autoComplete="name"
            className="w-full rounded-md border border-ash dark:border-gray-600 bg-transparent px-3 py-2 text-sm text-charcoal dark:text-white focus:outline-none focus:ring-2 focus:ring-canopy-green/50"
            value={formData.name}
            onChange={(e) => setFormData(p => ({ ...p, name: e.target.value }))}
          />
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label htmlFor="intake-age" className="block text-xs font-medium text-graphite dark:text-gray-300 mb-1">Age</label>
            <input
              id="intake-age"
              required type="number" min="0" max="130"
              autoComplete="age"
              className="w-full rounded-md border border-ash dark:border-gray-600 bg-transparent px-3 py-2 text-sm text-charcoal dark:text-white focus:outline-none focus:ring-2 focus:ring-canopy-green/50"
              value={formData.age}
              onChange={(e) => setFormData(p => ({ ...p, age: e.target.value }))}
            />
          </div>
          <div>
            <label htmlFor="intake-gender" className="block text-xs font-medium text-graphite dark:text-gray-300 mb-1">Gender</label>
            <select
              id="intake-gender"
              className="w-full rounded-md border border-ash dark:border-gray-600 bg-transparent px-3 py-2 text-sm text-charcoal dark:text-white focus:outline-none focus:ring-2 focus:ring-canopy-green/50"
              value={formData.gender}
              onChange={(e) => setFormData(p => ({ ...p, gender: e.target.value }))}
            >
              <option value="Male">Male</option>
              <option value="Female">Female</option>
              <option value="Other">Other</option>
            </select>
          </div>
        </div>

        <div>
          <label htmlFor="intake-contact" className="block text-xs font-medium text-graphite dark:text-gray-300 mb-1">Contact Number</label>
          <input
            id="intake-contact"
            required type="tel" placeholder="+1..."
            autoComplete="tel"
            className="w-full rounded-md border border-ash dark:border-gray-600 bg-transparent px-3 py-2 text-sm text-charcoal dark:text-white focus:outline-none focus:ring-2 focus:ring-canopy-green/50"
            value={formData.contact}
            onChange={(e) => setFormData(p => ({ ...p, contact: e.target.value }))}
          />
        </div>

        <button type="submit" className="w-full py-2 bg-canopy-green hover:bg-leaf-bright text-white rounded-md text-sm font-semibold transition-colors mt-2">
          Start Triage
        </button>
      </form>
    </div>
  );
}