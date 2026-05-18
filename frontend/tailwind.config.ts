import type { Config } from 'tailwindcss'

const config: Config = {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        border: 'hsl(224 10% 15%)',
        input: 'hsl(224 10% 20%)',
        ring: 'hsl(224 10% 25%)',
        background: 'hsl(224 15% 8%)',
        foreground: 'hsl(224 5% 95%)',
        primary: {
          DEFAULT: 'hsl(224 80% 65%)',
          foreground: 'hsl(224 80% 98%)',
        },
        secondary: {
          DEFAULT: 'hsl(220 15% 18%)',
          foreground: 'hsl(220 15% 75%)',
        },
        destructive: {
          DEFAULT: 'hsl(0 70% 55%)',
          foreground: 'hsl(0 70% 95%)',
        },
        muted: {
          DEFAULT: 'hsl(224 10% 18%)',
          foreground: 'hsl(224 10% 60%)',
        },
        accent: {
          DEFAULT: 'hsl(160 60% 45%)',
          foreground: 'hsl(160 60% 95%)',
        },
        card: {
          DEFAULT: 'hsl(224 12% 12%)',
          foreground: 'hsl(224 5% 90%)',
        },
      },
      borderRadius: {
        lg: '0.75rem',
        md: '0.5rem',
        sm: '0.375rem',
      },
      keyframes: {
        'pulse-slow': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.6' },
        },
      },
      animation: {
        'pulse-slow': 'pulse-slow 3s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}

export default config
