# Trading Matrix Command Center

A comprehensive React-based Matrix-themed command center for the Day Trading Orchestrator System. This modern web application provides a professional trading interface with real-time functionality for managing multi-broker trading operations.

## 🚀 Live Demo

**Deployed Application**: https://392yraumsxwg.space.minimax.io

## ✨ Features

### Core Functionality
- **7 Complete Pages**: Dashboard, Orders, Strategies, Brokers, Risk, AI Assistant, Configuration
- **Real-time Data**: WebSocket integration for live portfolio and market updates
- **Multi-Broker Support**: Binance, Interactive Brokers (IBKR), Alpaca Markets
- **Order Management**: Full order entry with MARKET, LIMIT, STOP, STOP_LIMIT types
- **Strategy Monitoring**: Track and control automated trading strategies
- **Risk Analytics**: Comprehensive risk metrics and portfolio analysis
- **AI Assistant**: Integrated GPT-4/Claude AI for market analysis and trading assistance
- **Demo Data**: Complete demo dataset for testing all features without live connections

### Matrix Theme Design
- **Cyberpunk Aesthetic**: Immersive Matrix (#00ff00) green color scheme
- **Glowing Effects**: Animated borders, shadows, and hover states
- **Monospace Fonts**: JetBrains Mono and Fira Code for terminal aesthetics
- **Custom Animations**: Matrix rain effect, data flow animations, glitch effects
- **Terminal UI**: Command-center style interface with glowing panels

## 🛠 Technology Stack

### Frontend Framework
- **React 18.3** - Modern UI library with hooks
- **TypeScript** - Type-safe development
- **Vite 6** - Fast build tool and dev server

### UI & Styling
- **TailwindCSS 3.4** - Utility-first CSS with custom Matrix theme
- **Radix UI** - Accessible headless components
- **Lucide React** - High-quality SVG icons

### State Management & Data
- **Zustand 5** - Lightweight global state management
- **TanStack React Query 5** - Server state and data fetching
- **Axios** - HTTP client for API requests

### Forms & Validation
- **React Hook Form 7** - Performant form management
- **Zod 3** - TypeScript-first schema validation

### Real-time Communication
- **WebSocket** - Live data streaming from backend
- **React Hot Toast** - Beautiful toast notifications

## 📁 Project Structure

```
trading-command-center/
├── src/
│   ├── components/          # Reusable UI components
│   │   ├── DataTable.tsx    # Matrix-themed data table
│   │   ├── GlowingButton.tsx
│   │   ├── MatrixCard.tsx
│   │   ├── PriceDisplay.tsx
│   │   ├── StatCard.tsx
│   │   ├── StatusBadge.tsx
│   │   └── Layout.tsx       # Main layout with navigation
│   ├── pages/               # Page components
│   │   ├── Dashboard.tsx    # Portfolio overview
│   │   ├── Orders.tsx       # Order management
│   │   ├── Strategies.tsx   # Strategy monitoring
│   │   ├── Brokers.tsx      # Broker connections
│   │   ├── Risk.tsx         # Risk analytics
│   │   ├── AIAssistant.tsx  # AI chat interface
│   │   └── Configuration.tsx
│   ├── services/
│   │   ├── api.ts           # API service layer
│   │   └── websocket.ts     # WebSocket service
│   ├── store/
│   │   └── tradingStore.ts  # Zustand state management
│   ├── hooks/
│   │   ├── useApi.ts        # React Query hooks
│   │   └── useWebSocket.ts  # WebSocket hooks
│   ├── types/
│   │   └── index.ts         # TypeScript type definitions
│   ├── utils/
│   │   ├── demoData.ts      # Demo data for testing
│   │   └── cn.ts            # Utility functions
│   ├── App.tsx              # Main app with routing
│   ├── main.tsx             # Application entry point
│   └── index.css            # Global styles & Matrix theme
├── public/                  # Static assets
├── dist/                    # Production build output
└── tailwind.config.js       # Tailwind custom configuration
```

## 🎨 Pages Overview

### 1. Dashboard
- Real-time portfolio overview with 4 key stat cards
- Active positions table with P&L tracking
- Risk metrics panel with 6 key indicators
- Recent trades history
- Live/Disconnected connection indicator

### 2. Orders
- **NEW ORDER** button to open order entry form
- Dynamic form with symbol, broker, type, side, quantity, price fields
- Conditional price field display based on order type
- Active orders table with status badges
- Order cancellation functionality

### 3. Strategies
- Strategy cards with comprehensive metrics
- Total P&L, Daily P&L, Win Rate, Sharpe Ratio, Max Drawdown
- PAUSE/START controls for each strategy
- Performance summary table
- Strategy status indicators

### 4. Brokers
- 3 broker connection cards (Binance, IBKR, Alpaca)
- Connection status indicators with green check/red X
- Balance, positions, orders count
- Last sync timestamp
- SYNC buttons for manual synchronization
- Connection information panel

### 5. Risk
- 4 top-level risk metric cards
- 6 detailed metric sections (Drawdown, Performance, Win Metrics, VaR, Concentration, Risk Status)
- Color-coded metrics (green=good, yellow=warning, red=danger)
- Risk alerts panel
- Portfolio health indicators

### 6. AI Assistant
- Full-featured chat interface
- User messages on right (blue highlight), AI on left (with bot icon)
- 4 quick action buttons for common queries
- AI capabilities panel showing 4 core features
- Message history with timestamps
- Interactive send button

### 7. Configuration
- **4 settings sections**:
  - Trading Settings (broker, time in force, order size)
  - Risk Management (position limits, stop loss, take profit)
  - AI Assistant Settings (provider, model, temperature, auto-trade)
  - Display Settings (theme, refresh interval, notifications)
- Interactive form controls (dropdowns, inputs, checkboxes, sliders)
- SAVE CONFIGURATION and RESET TO DEFAULTS buttons
- System information panel (version, API status, brokers, AI engine)

## 🔧 Development

### Prerequisites
- Node.js 18+ and pnpm

### Installation
```bash
# Clone the repository
cd /workspace/trading-command-center

# Install dependencies
pnpm install

# Start development server
pnpm run dev

# Build for production
pnpm run build

# Preview production build
pnpm run preview
```

### Environment Variables
Create a `.env` file in the root directory:

```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

## 🔌 Backend Integration

The frontend is designed to connect to a FastAPI backend. WebSocket auto-reconnection is built-in.

## 📊 Demo Data

Complete demo dataset available: Portfolio ($125,450.50), 5 positions, 3 orders, 3 brokers, 3 strategies, risk metrics.

## 🧪 Testing

Comprehensive testing performed - all features verified working. See `test-progress.md`.

## 🚀 Production Build

- **HTML**: 0.35 kB (gzipped: 0.25 kB)
- **CSS**: 15.41 kB (gzipped: 3.51 kB)
- **JavaScript**: 407.59 kB (gzipped: 91.34 kB)

Created by MiniMax Agent.
