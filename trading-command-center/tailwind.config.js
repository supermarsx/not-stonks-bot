/** @type {import('tailwindcss').Config} */
module.exports = {
	darkMode: ['class'],
	content: [
		'./pages/**/*.{ts,tsx}',
		'./components/**/*.{ts,tsx}',
		'./app/**/*.{ts,tsx}',
		'./src/**/*.{ts,tsx}',
	],
	theme: {
		container: {
			center: true,
			padding: '2rem',
			screens: {
				'2xl': '1400px',
			},
		},
		extend: {
			colors: {
				// Matrix theme colors
				matrix: {
					green: '#00ff00',
					'dark-green': '#003300',
					'medium-green': '#006600',
					black: '#000000',
					'dark-gray': '#111111',
					glow: '#00ff0088',
				},
				border: '#003300',
				input: '#001100',
				ring: '#00ff00',
				background: '#000000',
				foreground: '#00ff00',
				primary: {
					DEFAULT: '#00ff00',
					foreground: '#000000',
				},
				secondary: {
					DEFAULT: '#003300',
					foreground: '#00ff00',
				},
				accent: {
					DEFAULT: '#006600',
					foreground: '#00ff00',
				},
				destructive: {
					DEFAULT: '#ff0000',
					foreground: '#ffffff',
				},
				muted: {
					DEFAULT: '#001100',
					foreground: '#00cc00',
				},
				popover: {
					DEFAULT: '#000000',
					foreground: '#00ff00',
				},
				card: {
					DEFAULT: '#000000',
					foreground: '#00ff00',
				},
			},
			borderRadius: {
				lg: '0px',
				md: '0px',
				sm: '0px',
			},
			fontFamily: {
				mono: ['JetBrains Mono', 'Fira Code', 'Courier New', 'monospace'],
			},
			boxShadow: {
				'matrix-glow': '0 0 10px #00ff00, inset 0 0 10px #00ff0033',
				'matrix-glow-lg': '0 0 20px #00ff00, inset 0 0 20px #00ff0033',
			},
			keyframes: {
				'accordion-down': {
					from: { height: 0 },
					to: { height: 'var(--radix-accordion-content-height)' },
				},
				'accordion-up': {
					from: { height: 'var(--radix-accordion-content-height)' },
					to: { height: 0 },
				},
				pulse: {
					'0%, 100%': { opacity: 1 },
					'50%': { opacity: 0.5 },
				},
				'matrix-rain': {
					'0%': { transform: 'translateY(-100%)' },
					'100%': { transform: 'translateY(100%)' },
				},
				glow: {
					'0%, 100%': { 
						boxShadow: '0 0 5px #00ff00, inset 0 0 5px #00ff0033',
					},
					'50%': { 
						boxShadow: '0 0 20px #00ff00, inset 0 0 15px #00ff0066',
					},
				},
				'data-flow': {
					'0%': { transform: 'translateX(-100%)' },
					'100%': { transform: 'translateX(100%)' },
				},
			},
			animation: {
				'accordion-down': 'accordion-down 0.2s ease-out',
				'accordion-up': 'accordion-up 0.2s ease-out',
				pulse: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
				'matrix-rain': 'matrix-rain 2s linear infinite',
				glow: 'glow 2s ease-in-out infinite',
				'data-flow': 'data-flow 3s linear infinite',
			},
		},
	},
	plugins: [require('tailwindcss-animate')],
}
