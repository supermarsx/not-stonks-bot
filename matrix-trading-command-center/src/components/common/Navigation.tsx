import React from 'react';
import { motion } from 'framer-motion';
import { useTradingStore } from '@/stores/tradingStore';
import { MatrixButton } from '@/components/ui/MatrixButton';
import { StatusIndicator } from '@/components/ui/StatusIndicator';
import {
  Activity,
  TrendingUp,
  AlertTriangle,
  Settings,
  MessageSquare,
  BarChart3,
  PieChart,
  Wifi,
  WifiOff,
} from 'lucide-react';

interface NavigationProps {
  onViewChange: (view: string) => void;
}

export const Navigation: React.FC<NavigationProps> = ({ onViewChange }) => {
  const { 
    activeView, 
    wsConnected, 
    apiConnected,
    unreadAlerts,
    brokerConnections
  } = useTradingStore();
  
  const connectedBrokers = brokerConnections.filter(b => b.status === 'connected').length;
  
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
    { id: 'portfolio', label: 'Portfolio', icon: PieChart },
    { id: 'orders', label: 'Orders', icon: TrendingUp },
    { id: 'risk', label: 'Risk', icon: AlertTriangle },
    { id: 'strategies', label: 'Strategies', icon: Activity },
    { id: 'chat', label: 'AI Chat', icon: MessageSquare },
    { id: 'settings', label: 'Settings', icon: Settings },
  ];
  
  return (
    <motion.nav
      className="matrix-card p-4 space-y-4"
      initial={{ x: -300 }}
      animate={{ x: 0 }}
      transition={{ duration: 0.5 }}
    >
      {/* Logo/Title */}
      <div className="text-center pb-4 border-b border-green-800/30">
        <h1 className="text-xl font-bold matrix-text-glow text-green-400">
          MATRIX TRADING
        </h1>
        <p className="text-xs text-green-600 font-mono">Command Center</p>
      </div>
      
      {/* Connection Status */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-sm font-mono text-green-400">WebSocket</span>
          <StatusIndicator 
            status={wsConnected ? 'online' : 'offline'} 
            size="sm"
          />
        </div>
        
        <div className="flex items-center justify-between">
          <span className="text-sm font-mono text-green-400">API</span>
          <StatusIndicator 
            status={apiConnected ? 'online' : 'offline'} 
            size="sm"
          />
        </div>
        
        <div className="flex items-center justify-between">
          <span className="text-sm font-mono text-green-400">Brokers</span>
          <div className="flex items-center gap-2">
            <span className="text-sm text-green-500 font-mono">
              {connectedBrokers}
            </span>
            <StatusIndicator 
              status={connectedBrokers > 0 ? 'online' : 'offline'} 
              size="sm"
            />
          </div>
        </div>
      </div>
      
      {/* Alerts */}
      {unreadAlerts > 0 && (
        <motion.div
          className="bg-red-900/30 border border-red-700 rounded p-3"
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          whileHover={{ scale: 1.02 }}
        >
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-red-400" />
            <span className="text-sm font-mono text-red-400">
              {unreadAlerts} Alert{unreadAlerts !== 1 ? 's' : ''}
            </span>
          </div>
        </motion.div>
      )}
      
      {/* Navigation Items */}
      <div className="space-y-1">
        {navItems.map((item, index) => {
          const Icon = item.icon;
          const isActive = activeView === item.id;
          
          return (
            <motion.button
              key={item.id}
              onClick={() => onViewChange(item.id)}
              className={`w-full flex items-center gap-3 px-3 py-2 rounded font-mono text-sm transition-all duration-200 ${
                isActive 
                  ? 'bg-green-600/30 text-green-400 border border-green-500' 
                  : 'text-green-600 hover:bg-green-900/20 hover:text-green-400'
              }`}
              whileHover={{ x: 4 }}
              whileTap={{ scale: 0.98 }}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <Icon className="w-4 h-4" />
              <span>{item.label}</span>
              {item.id === 'chat' && (
                <div className="ml-auto w-2 h-2 bg-blue-400 rounded-full animate-pulse" />
              )}
            </motion.button>
          );
        })}
      </div>
      
      {/* Footer */}
      <div className="pt-4 border-t border-green-800/30 text-center">
        <p className="text-xs text-green-600 font-mono">
          v2.0.1 | MATRIX
        </p>
      </div>
    </motion.nav>
  );
};