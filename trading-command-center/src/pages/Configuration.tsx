import { Save, Settings, Shield, Zap } from 'lucide-react';
import { useState } from 'react';
import { MatrixCard } from '../components/MatrixCard';
import { GlowingButton } from '../components/GlowingButton';
import toast from 'react-hot-toast';

export default function Configuration() {
  const [config, setConfig] = useState({
    // Trading Settings
    defaultBroker: 'alpaca',
    defaultTimeInForce: 'GTC',
    confirmOrders: true,
    maxOrderSize: 10000,

    // Risk Settings
    maxPositionSize: 50000,
    maxDailyLoss: 5000,
    maxDrawdown: 10,
    enableStopLoss: true,
    enableTakeProfit: true,

    // AI Settings
    aiProvider: 'openai',
    aiModel: 'gpt-4',
    aiTemperature: 0.7,
    enableAutoTrade: false,

    // Display Settings
    theme: 'matrix',
    refreshInterval: 5000,
    showNotifications: true,
  });

  const handleSave = () => {
    toast.success('Configuration saved successfully!');
  };

  const handleReset = () => {
    toast('Configuration reset to defaults', { icon: '🔄' });
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold matrix-glow-text">SYSTEM CONFIGURATION</h1>
        <p className="mt-1 text-sm text-matrix-green/70">
          Manage trading parameters, risk limits, and system preferences
        </p>
      </div>

      {/* Configuration Sections */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Trading Settings */}
        <MatrixCard title="TRADING SETTINGS" glow>
          <div className="space-y-4">
            <div>
              <label className="mb-2 block text-sm text-matrix-green/70">DEFAULT BROKER</label>
              <select
                value={config.defaultBroker}
                onChange={(e) => setConfig({ ...config, defaultBroker: e.target.value })}
                className="w-full border-2 border-matrix-green bg-matrix-black px-3 py-2 text-matrix-green focus:outline-none focus:ring-2 focus:ring-matrix-green"
              >
                <option value="alpaca">Alpaca Markets</option>
                <option value="ibkr">Interactive Brokers</option>
                <option value="binance">Binance</option>
              </select>
            </div>

            <div>
              <label className="mb-2 block text-sm text-matrix-green/70">
                DEFAULT TIME IN FORCE
              </label>
              <select
                value={config.defaultTimeInForce}
                onChange={(e) => setConfig({ ...config, defaultTimeInForce: e.target.value })}
                className="w-full border-2 border-matrix-green bg-matrix-black px-3 py-2 text-matrix-green focus:outline-none focus:ring-2 focus:ring-matrix-green"
              >
                <option value="GTC">GTC (Good Till Cancelled)</option>
                <option value="DAY">DAY</option>
                <option value="IOC">IOC (Immediate or Cancel)</option>
                <option value="FOK">FOK (Fill or Kill)</option>
              </select>
            </div>

            <div>
              <label className="mb-2 block text-sm text-matrix-green/70">
                MAX ORDER SIZE ($)
              </label>
              <input
                type="number"
                value={config.maxOrderSize}
                onChange={(e) => setConfig({ ...config, maxOrderSize: Number(e.target.value) })}
                className="w-full border-2 border-matrix-green bg-matrix-black px-3 py-2 text-matrix-green placeholder-matrix-green/30 focus:outline-none focus:ring-2 focus:ring-matrix-green"
              />
            </div>

            <div className="flex items-center justify-between">
              <label className="text-sm text-matrix-green/70">CONFIRM ORDERS</label>
              <input
                type="checkbox"
                checked={config.confirmOrders}
                onChange={(e) => setConfig({ ...config, confirmOrders: e.target.checked })}
                className="h-5 w-5 border-2 border-matrix-green bg-matrix-black checked:bg-matrix-green"
              />
            </div>
          </div>
        </MatrixCard>

        {/* Risk Management */}
        <MatrixCard title="RISK MANAGEMENT" glow>
          <div className="space-y-4">
            <div>
              <label className="mb-2 block text-sm text-matrix-green/70">
                MAX POSITION SIZE ($)
              </label>
              <input
                type="number"
                value={config.maxPositionSize}
                onChange={(e) => setConfig({ ...config, maxPositionSize: Number(e.target.value) })}
                className="w-full border-2 border-matrix-green bg-matrix-black px-3 py-2 text-matrix-green placeholder-matrix-green/30 focus:outline-none focus:ring-2 focus:ring-matrix-green"
              />
            </div>

            <div>
              <label className="mb-2 block text-sm text-matrix-green/70">
                MAX DAILY LOSS ($)
              </label>
              <input
                type="number"
                value={config.maxDailyLoss}
                onChange={(e) => setConfig({ ...config, maxDailyLoss: Number(e.target.value) })}
                className="w-full border-2 border-matrix-green bg-matrix-black px-3 py-2 text-matrix-green placeholder-matrix-green/30 focus:outline-none focus:ring-2 focus:ring-matrix-green"
              />
            </div>

            <div>
              <label className="mb-2 block text-sm text-matrix-green/70">
                MAX DRAWDOWN (%)
              </label>
              <input
                type="number"
                value={config.maxDrawdown}
                onChange={(e) => setConfig({ ...config, maxDrawdown: Number(e.target.value) })}
                className="w-full border-2 border-matrix-green bg-matrix-black px-3 py-2 text-matrix-green placeholder-matrix-green/30 focus:outline-none focus:ring-2 focus:ring-matrix-green"
              />
            </div>

            <div className="flex items-center justify-between">
              <label className="text-sm text-matrix-green/70">ENABLE STOP LOSS</label>
              <input
                type="checkbox"
                checked={config.enableStopLoss}
                onChange={(e) => setConfig({ ...config, enableStopLoss: e.target.checked })}
                className="h-5 w-5 border-2 border-matrix-green bg-matrix-black checked:bg-matrix-green"
              />
            </div>

            <div className="flex items-center justify-between">
              <label className="text-sm text-matrix-green/70">ENABLE TAKE PROFIT</label>
              <input
                type="checkbox"
                checked={config.enableTakeProfit}
                onChange={(e) => setConfig({ ...config, enableTakeProfit: e.target.checked })}
                className="h-5 w-5 border-2 border-matrix-green bg-matrix-black checked:bg-matrix-green"
              />
            </div>
          </div>
        </MatrixCard>

        {/* AI Configuration */}
        <MatrixCard title="AI ASSISTANT SETTINGS" glow>
          <div className="space-y-4">
            <div>
              <label className="mb-2 block text-sm text-matrix-green/70">AI PROVIDER</label>
              <select
                value={config.aiProvider}
                onChange={(e) => setConfig({ ...config, aiProvider: e.target.value })}
                className="w-full border-2 border-matrix-green bg-matrix-black px-3 py-2 text-matrix-green focus:outline-none focus:ring-2 focus:ring-matrix-green"
              >
                <option value="openai">OpenAI (GPT-4)</option>
                <option value="anthropic">Anthropic (Claude)</option>
              </select>
            </div>

            <div>
              <label className="mb-2 block text-sm text-matrix-green/70">MODEL</label>
              <select
                value={config.aiModel}
                onChange={(e) => setConfig({ ...config, aiModel: e.target.value })}
                className="w-full border-2 border-matrix-green bg-matrix-black px-3 py-2 text-matrix-green focus:outline-none focus:ring-2 focus:ring-matrix-green"
              >
                {config.aiProvider === 'openai' ? (
                  <>
                    <option value="gpt-4">GPT-4</option>
                    <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
                  </>
                ) : (
                  <>
                    <option value="claude-3-sonnet">Claude 3 Sonnet</option>
                    <option value="claude-3-haiku">Claude 3 Haiku</option>
                  </>
                )}
              </select>
            </div>

            <div>
              <label className="mb-2 block text-sm text-matrix-green/70">
                TEMPERATURE ({config.aiTemperature})
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={config.aiTemperature}
                onChange={(e) => setConfig({ ...config, aiTemperature: Number(e.target.value) })}
                className="w-full"
              />
            </div>

            <div className="flex items-center justify-between">
              <label className="text-sm text-matrix-green/70">ENABLE AUTO-TRADE</label>
              <input
                type="checkbox"
                checked={config.enableAutoTrade}
                onChange={(e) => setConfig({ ...config, enableAutoTrade: e.target.checked })}
                className="h-5 w-5 border-2 border-matrix-green bg-matrix-black checked:bg-matrix-green"
              />
            </div>

            {config.enableAutoTrade && (
              <div className="border-2 border-yellow-500 bg-yellow-500/10 p-3">
                <p className="text-sm text-yellow-500">
                  ⚠️ WARNING: Auto-trade enabled. AI will execute trades automatically based on
                  market analysis.
                </p>
              </div>
            )}
          </div>
        </MatrixCard>

        {/* Display Settings */}
        <MatrixCard title="DISPLAY SETTINGS" glow>
          <div className="space-y-4">
            <div>
              <label className="mb-2 block text-sm text-matrix-green/70">THEME</label>
              <select
                value={config.theme}
                onChange={(e) => setConfig({ ...config, theme: e.target.value })}
                className="w-full border-2 border-matrix-green bg-matrix-black px-3 py-2 text-matrix-green focus:outline-none focus:ring-2 focus:ring-matrix-green"
              >
                <option value="matrix">Matrix (Green)</option>
                <option value="dark">Dark</option>
                <option value="light">Light</option>
              </select>
            </div>

            <div>
              <label className="mb-2 block text-sm text-matrix-green/70">
                REFRESH INTERVAL (ms)
              </label>
              <input
                type="number"
                value={config.refreshInterval}
                onChange={(e) => setConfig({ ...config, refreshInterval: Number(e.target.value) })}
                className="w-full border-2 border-matrix-green bg-matrix-black px-3 py-2 text-matrix-green placeholder-matrix-green/30 focus:outline-none focus:ring-2 focus:ring-matrix-green"
              />
            </div>

            <div className="flex items-center justify-between">
              <label className="text-sm text-matrix-green/70">SHOW NOTIFICATIONS</label>
              <input
                type="checkbox"
                checked={config.showNotifications}
                onChange={(e) => setConfig({ ...config, showNotifications: e.target.checked })}
                className="h-5 w-5 border-2 border-matrix-green bg-matrix-black checked:bg-matrix-green"
              />
            </div>
          </div>
        </MatrixCard>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-4">
        <GlowingButton onClick={handleSave} icon={<Save size={20} />}>
          SAVE CONFIGURATION
        </GlowingButton>
        <GlowingButton variant="secondary" onClick={handleReset}>
          RESET TO DEFAULTS
        </GlowingButton>
      </div>

      {/* System Information */}
      <MatrixCard title="SYSTEM INFORMATION" glow>
        <div className="grid gap-4 md:grid-cols-3">
          <div>
            <p className="mb-1 text-xs text-matrix-green/70">VERSION</p>
            <p className="font-bold">v1.0.0</p>
          </div>
          <div>
            <p className="mb-1 text-xs text-matrix-green/70">API STATUS</p>
            <p className="font-bold text-matrix-green">CONNECTED</p>
          </div>
          <div>
            <p className="mb-1 text-xs text-matrix-green/70">WEBSOCKET</p>
            <p className="font-bold text-matrix-green">ACTIVE</p>
          </div>
          <div>
            <p className="mb-1 text-xs text-matrix-green/70">DATABASE</p>
            <p className="font-bold">SQLite</p>
          </div>
          <div>
            <p className="mb-1 text-xs text-matrix-green/70">BROKERS</p>
            <p className="font-bold">3 Connected</p>
          </div>
          <div>
            <p className="mb-1 text-xs text-matrix-green/70">AI ENGINE</p>
            <p className="font-bold">GPT-4 + Claude</p>
          </div>
        </div>
      </MatrixCard>
    </div>
  );
}
