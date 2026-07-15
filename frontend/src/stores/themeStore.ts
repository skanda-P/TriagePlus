import { create } from 'zustand';

interface ThemeStore {
  isDark: boolean;
  toggleTheme: () => void;
}

// Check initial state from localStorage or system preference
const getInitialTheme = () => {
  const saved = localStorage.getItem('theme');
  if (saved) return saved === 'dark';
  return false; // Default to light mode
};

// Apply initial class
const initDark = getInitialTheme();
if (initDark) document.documentElement.classList.add('dark');

export const useThemeStore = create<ThemeStore>((set) => ({
  isDark: initDark,
  toggleTheme: () => set((state) => {
    const newDark = !state.isDark;
    if (newDark) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
    return { isDark: newDark };
  }),
}));
