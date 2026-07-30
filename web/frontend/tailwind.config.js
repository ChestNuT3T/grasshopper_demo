/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#10a37f',
          hover: '#0d8f6d',
          light: 'rgba(16, 163, 127, 0.08)',
        },
        success: '#10b981',
        warning: '#f59e0b',
        danger: '#ef4444',
        bg: {
          DEFAULT: '#ffffff',
          dark: '#212121',
        },
        sidebar: {
          DEFAULT: '#f9fafb',
          dark: '#171717',
        },
        card: {
          DEFAULT: '#f3f4f6',
          dark: '#252525',
        },
        bubble: {
          user: '#f3f4f6',
          userDark: '#2a2a2a',
          ai: 'transparent',
          aiDark: 'transparent',
        },
        text: {
          main: '#1f2937',
          mainDark: '#e5e7eb',
          sub: '#6b7280',
          subDark: '#9ca3af',
          muted: '#9ca3af',
          mutedDark: '#6b7280',
        },
        border: {
          DEFAULT: '#e5e7eb',
          dark: '#2d2d2d',
        },
        hover: {
          DEFAULT: '#f3f4f6',
          dark: '#2a2a2a',
        },
      },
      fontFamily: {
        sans: ['-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', 'Noto Sans', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'SF Mono', 'Monaco', 'Consolas', 'monospace'],
      },
      fontSize: {
        'base': ['15px', '1.6'],
        'sm': ['13px', '1.5'],
        'xs': ['12px', '1.4'],
        'lg': ['16px', '1.6'],
        'xl': ['18px', '1.5'],
      },
      animation: {
        'fade-in': 'fadeIn 0.15s ease-out',
        'slide-up': 'slideUp 0.2s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(4px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
}