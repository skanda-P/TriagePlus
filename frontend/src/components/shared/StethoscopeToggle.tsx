import React from 'react';
import { useThemeStore } from '../../stores/themeStore';

export function StethoscopeToggle({ className = '' }: { className?: string }) {
  const { isDark, toggleTheme } = useThemeStore();

  return (
    <div className={`group flex flex-col items-center select-none transition-transform hover:scale-105 duration-300 ${className}`}>
      <button
        type="button"
        onClick={toggleTheme}
        aria-label={isDark ? 'Switch to light theme' : 'Switch to dark theme'}
        className="bg-transparent border-0 p-0 cursor-pointer"
      >
        <svg
          width="32"
          height="64"
          viewBox="0 0 64 128"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          className="text-slate-muted dark:text-ash"
        >
          {/* Earpieces */}
          <circle cx="20" cy="10" r="4" fill="currentColor" />
          <circle cx="44" cy="10" r="4" fill="currentColor" />

          {/* Ear tubes */}
          <path d="M20 10 C 20 30, 32 40, 32 50" stroke="currentColor" strokeWidth="4" strokeLinecap="round" />
          <path d="M44 10 C 44 30, 32 40, 32 50" stroke="currentColor" strokeWidth="4" strokeLinecap="round" />

          {/* Main tube */}
          <path d="M32 50 L 32 90" stroke="currentColor" strokeWidth="6" strokeLinecap="round" />

          {/* Diaphragm */}
          <circle
            cx="32"
            cy="104"
            r="16"
            className={`transition-colors duration-300 ${isDark ? 'fill-sky-signal' : 'fill-indigo-bloom'} group-hover:opacity-80`}
          />
          <circle
            cx="32"
            cy="104"
            r="8"
            className="fill-white dark:fill-charcoal transition-colors duration-300 group-hover:scale-110 transform origin-center"
          />
        </svg>
      </button>
      <div className="mt-2 text-[10px] font-bold text-slate-muted dark:text-ash tracking-widest uppercase opacity-0 transition-opacity duration-300 group-hover:opacity-100">
        {isDark ? 'Light' : 'Dark'}
      </div>
    </div>
  );
}