/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        'canopy-green':   '#0a3922',
        'coral-pulse':    '#ff643b',
        'indigo-bloom':   '#460095',
        'leaf-bright':    '#1dbf73',
        'deep-teal':      '#003642',
        'orchid-tint':    '#f3b3fe',
        'sky-signal':     '#00b4dc',
        'aubergine':      '#47264c',
        'ink-black':      '#000000',
        'paper-white':    '#ffffff',
        'ash':            '#a6a6a6',
        'slate-muted':    '#7a7a7a',
        'graphite':       '#3d3d3d',
        'charcoal':       '#333333',
        'frost-gray':     '#e0e0e0',
        'cloud-gray':     '#f2f2f2',
        'lavender-mist':  '#eee2ff',
        'mint-wash':      '#d2f2e3',
        'sky-wash':       '#ccf0f8',
        'sage-wash':      '#e4f7ee',
        'lilac-wash':     '#fdf0ff',
        'peach-wash':     '#ffede8',
        'cream':          '#faf7e8',
        'sand':           '#f4ece9',
        'triage-red':     '#ef4444',
        'triage-orange':  '#f97316',
        'triage-yellow':  '#eab308',
        'triage-green':   '#22c55e',
      },
      fontFamily: {
        'dm-sans': ['DM Sans', 'Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        sans:      ['DM Sans', 'Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
      },
      borderRadius: {
        'tags':     '9999px',
        'cards':    '24px',
        'icons':    '8px',
        'buttons':  '40px',
        'cards-sm': '16px',
      },
      boxShadow: {
        'subtle': 'rgba(0, 0, 0, 0.05) 0px 1px 1px 0px',
      },
      keyframes: {
        slideUp: {
          '0%':   { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        fadeIn: {
          '0%':   { opacity: '0' },
          '100%': { opacity: '1' },
        },
        blink: {
          '0%, 100%': { opacity: '1' },
          '50%':      { opacity: '0.2' },
        },
      },
      animation: {
        'slide-up': 'slideUp 0.3s ease-out forwards',
        'fade-in':  'fadeIn 0.4s ease-out forwards',
        'blink':    'blink 1.4s ease-in-out infinite',
      },
    },
  },
  plugins: [],
};
