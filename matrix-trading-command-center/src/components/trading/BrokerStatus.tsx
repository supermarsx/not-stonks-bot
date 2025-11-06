import React from 'react';
import { motion } from 'framer-motion';
import { useTradingStore } from '@/stores/tradingStore';
import { MatrixCard } from '@/components/ui/MatrixCard';
import { StatusIndicator } from '@/components/ui/StatusIndicator';
import { 
  Server, 
  Wifi, 
  WifiOff,
  Clock,
  Activity
} from 'lucide-react';

export const BrokerStatus: React.FC = () => {
  const { brokerConnections } = useTradingStore();

  const connectedBrokers = brokerConnections.filter(b => b.status === 'connected');
  const totalLatency = connectedBrokers.reduce((sum, b) => sum + b.latency, 0);
  const avgLatency = connectedBrokers.length > 0 ? totalLatency / connectedBrokers.length : 0;

  return (
    <div className="space-y-4">
      <MatrixCard title="Broker Connections" glow>
        <div className="space-y-3">
          {brokerConnections.length === 0 ? (
            <div className="text-center py-8">
              <Server className="w-12 h-12 mx-auto mb-2 opacity-50 text-green-400" />
              <p className="text-green-600">No brokers configured</p>
              <p className="text-xs text-green-700 mt-1">Add broker connections in Settings</p>
            </div>
          ) : (
            <>
              {/* Summary */}
              <div className="grid grid-cols-2 gap-4 pb-4 border-b border-green-800/30">
                <div>
                  <div className="text-sm text-green-600">Connected</div>
                  <div className="text-xl font-bold text-green-400">
                    {connectedBrokers.length}/{brokerConnections.length}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-green-600">Avg Latency</div>
                  <div className="text-xl font-bold matrix-text-glow">
                    {avgLatency.toFixed(0)}ms
                  </div>
                </div>
              </div>

              {/* Broker List */}
              <div className="space-y-2">
                {brokerConnections.map((broker) => (
                  <motion.div
                    key={broker.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="flex items-center justify-between p-3 rounded border border-green-800/30 bg-green-900/5"
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 bg-gray-800 rounded flex items-center justify-center">
                        <Server className="w-4 h-4 text-green-400" />
                      </div>
                      <div>
                        <div className="font-mono text-sm text-green-400">{broker.name}</div>
                        <div className="text-xs text-green-600">
                          {broker.capabilities.join(', ') || 'Standard trading'}
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-3">
                      {broker.latency > 0 && (
                        <div className="text-xs text-green-600 font-mono">
                          {broker.latency}ms
                        </div>
                      )}
                      <StatusIndicator
                        status={
                          broker.status === 'connected' ? 'online' :
                          broker.status === 'connecting' ? 'warning' :
                          broker.status === 'error' ? 'error' : 'offline'
                        }
                      >
                        {broker.status.toUpperCase()}
                      </StatusIndicator>
                    </div>
                  </motion.div>
                ))}
              </div>
            </>
          )}
        </div>
      </MatrixCard>
    </div>
  );
};