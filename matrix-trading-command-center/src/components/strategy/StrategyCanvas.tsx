import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  DndContext, 
  DragEndEvent, 
  DragOverlay, 
  DragStartEvent,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors
} from '@dnd-kit/core';
import { 
  SortableContext, 
  sortableKeyboardCoordinates,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable';
import { MatrixCard } from '@/components/ui/MatrixCard';
import { MatrixButton } from '@/components/ui/MatrixButton';
import { 
  Plus, 
  Settings, 
  Play, 
  Pause, 
  Trash2,
  ArrowRight,
  ArrowDown,
  Brain,
  TrendingUp,
  TrendingDown,
  Zap,
  Target,
  BarChart3,
  DollarSign,
  Shield,
  Cpu,
  Network
} from 'lucide-react';

interface StrategyNode {
  id: string;
  type: 'signal' | 'filter' | 'entry' | 'exit' | 'risk' | 'broker' | 'portfolio';
  label: string;
  position: { x: number; y: number };
  config: Record<string, any>;
  connections: string[];
  icon: React.ReactNode;
  color: string;
}

interface Connection {
  from: string;
  to: string;
  label?: string;
}

export const StrategyCanvas: React.FC = () => {
  const [nodes, setNodes] = useState<StrategyNode[]>([
    {
      id: 'signal1',
      type: 'signal',
      label: 'Price Signal',
      position: { x: 100, y: 100 },
      config: { source: 'price', timeframe: '1h', indicator: 'RSI' },
      connections: ['filter1'],
      icon: <BarChart3 className="w-4 h-4" />,
      color: 'border-blue-500'
    },
    {
      id: 'filter1',
      type: 'filter',
      label: 'Volume Filter',
      position: { x: 300, y: 100 },
      config: { minVolume: 100000, volumeMA: 20 },
      connections: ['entry1'],
      icon: <Target className="w-4 h-4" />,
      color: 'border-purple-500'
    },
    {
      id: 'entry1',
      type: 'entry',
      label: 'Long Entry',
      position: { x: 500, y: 100 },
      config: { signalType: 'buy', threshold: 0.02, timeframe: '1h' },
      connections: ['risk1'],
      icon: <TrendingUp className="w-4 h-4" />,
      color: 'border-green-500'
    },
    {
      id: 'risk1',
      type: 'risk',
      label: 'Stop Loss',
      position: { x: 500, y: 250 },
      config: { stopLoss: 0.03, trailingStop: true },
      connections: ['exit1'],
      icon: <Shield className="w-4 h-4" />,
      color: 'border-red-500'
    },
    {
      id: 'exit1',
      type: 'exit',
      label: 'Profit Target',
      position: { x: 700, y: 175 },
      config: { takeProfit: 0.06, exitSignal: 'profit' },
      connections: ['broker1'],
      icon: <TrendingDown className="w-4 h-4" />,
      color: 'border-yellow-500'
    },
    {
      id: 'broker1',
      type: 'broker',
      label: 'Alpaca Broker',
      position: { x: 900, y: 175 },
      config: { broker: 'alpaca', allocation: 50, orderType: 'market' },
      connections: [],
      icon: <Network className="w-4 h-4" />,
      color: 'border-cyan-500'
    }
  ]);

  const [connections, setConnections] = useState<Connection[]>([
    { from: 'signal1', to: 'filter1', label: 'Signal' },
    { from: 'filter1', to: 'entry1', label: 'Filter' },
    { from: 'entry1', to: 'risk1', label: 'Position' },
    { from: 'risk1', to: 'exit1', label: 'Risk' },
    { from: 'exit1', to: 'broker1', label: 'Execute' }
  ]);

  const [selectedNode, setSelectedNode] = useState<StrategyNode | null>(null);
  const [isRunning, setIsRunning] = useState(false);

  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );

  const nodeTypes = [
    { type: 'signal', label: 'Signal', icon: <BarChart3 className="w-4 h-4" />, color: 'border-blue-500' },
    { type: 'filter', label: 'Filter', icon: <Target className="w-4 h-4" />, color: 'border-purple-500' },
    { type: 'entry', label: 'Entry', icon: <TrendingUp className="w-4 h-4" />, color: 'border-green-500' },
    { type: 'exit', label: 'Exit', icon: <TrendingDown className="w-4 h-4" />, color: 'border-yellow-500' },
    { type: 'risk', label: 'Risk', icon: <Shield className="w-4 h-4" />, color: 'border-red-500' },
    { type: 'broker', label: 'Broker', icon: <Network className="w-4 h-4" />, color: 'border-cyan-500' },
    { type: 'portfolio', label: 'Portfolio', icon: <DollarSign className="w-4 h-4" />, color: 'border-orange-500' }
  ];

  const handleDragStart = useCallback((event: DragStartEvent) => {
    // Handle drag start
  }, []);

  const handleDragEnd = useCallback((event: DragEndEvent) => {
    // Handle drag end for nodes
  }, []);

  const addNode = (type: StrategyNode['type']) => {
    const newNode: StrategyNode = {
      id: `${type}_${Date.now()}`,
      type,
      label: `${type.charAt(0).toUpperCase() + type.slice(1)} Node`,
      position: { x: 200, y: 200 },
      config: getDefaultConfig(type),
      connections: [],
      icon: nodeTypes.find(n => n.type === type)?.icon || <Settings className="w-4 h-4" />,
      color: nodeTypes.find(n => n.type === type)?.color || 'border-gray-500'
    };
    setNodes([...nodes, newNode]);
  };

  const getDefaultConfig = (type: StrategyNode['type']): Record<string, any> => {
    switch (type) {
      case 'signal':
        return { source: 'price', timeframe: '1h', indicator: 'RSI', period: 14 };
      case 'filter':
        return { minVolume: 100000, volumeMA: 20, volatility: 0.02 };
      case 'entry':
        return { signalType: 'buy', threshold: 0.02, timeframe: '1h', confirmation: true };
      case 'exit':
        return { takeProfit: 0.06, stopLoss: 0.03, exitSignal: 'profit' };
      case 'risk':
        return { stopLoss: 0.03, trailingStop: true, maxPositionSize: 0.1 };
      case 'broker':
        return { broker: 'alpaca', allocation: 100, orderType: 'market', slippage: 0.001 };
      case 'portfolio':
        return { rebalanceFrequency: 'daily', maxDrawdown: 0.1, targetAllocation: {} };
      default:
        return {};
    }
  };

  const removeNode = (nodeId: string) => {
    setNodes(nodes.filter(n => n.id !== nodeId));
    setConnections(connections.filter(c => c.from !== nodeId && c.to !== nodeId));
  };

  const connectNodes = (fromId: string, toId: string) => {
    if (nodes.find(n => n.id === fromId)?.connections.includes(toId)) return;
    
    setConnections([...connections, { from: fromId, to: toId }]);
    setNodes(nodes.map(node => 
      node.id === fromId 
        ? { ...node, connections: [...node.connections, toId] }
        : node
    ));
  };

  const toggleStrategy = () => {
    setIsRunning(!isRunning);
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-green-800/30">
        <div>
          <h1 className="text-2xl font-bold matrix-text-glow text-green-400">
            STRATEGY BUILDER
          </h1>
          <p className="text-green-600 text-sm">Drag-and-drop strategy canvas</p>
        </div>
        
        <div className="flex items-center gap-3">
          <MatrixButton
            onClick={toggleStrategy}
            className={isRunning ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'}
          >
            {isRunning ? (
              <>
                <Pause className="w-4 h-4 mr-2" />
                STOP
              </>
            ) : (
              <>
                <Play className="w-4 h-4 mr-2" />
                RUN
              </>
            )}
          </MatrixButton>
        </div>
      </div>

      <div className="flex-1 flex">
        {/* Component Palette */}
        <div className="w-64 border-r border-green-800/30 p-4 bg-black/20">
          <h3 className="text-sm font-bold text-green-400 mb-4">COMPONENTS</h3>
          
          <div className="space-y-2">
            {nodeTypes.map(({ type, label, icon, color }) => (
              <div
                key={type}
                onClick={() => addNode(type as StrategyNode['type'])}
                className={`matrix-card p-3 border-2 ${color} cursor-pointer hover:bg-green-900/20 transition-colors group`}
              >
                <div className="flex items-center gap-3">
                  <div className="text-green-400 group-hover:matrix-text-glow">
                    {icon}
                  </div>
                  <div>
                    <div className="text-sm font-medium text-green-400">{label}</div>
                    <div className="text-xs text-green-600">{type}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Strategy Actions */}
          <div className="mt-8 pt-4 border-t border-green-800/30">
            <h4 className="text-xs font-bold text-green-400 mb-3">ACTIONS</h4>
            <div className="space-y-2">
              <MatrixButton size="sm" variant="secondary" className="w-full justify-start">
                <Settings className="w-3 h-3 mr-2" />
                Save Strategy
              </MatrixButton>
              <MatrixButton size="sm" variant="secondary" className="w-full justify-start">
                <Play className="w-3 h-3 mr-2" />
                Backtest
              </MatrixButton>
              <MatrixButton size="sm" variant="secondary" className="w-full justify-start">
                <Brain className="w-3 h-3 mr-2" />
                Optimize
              </MatrixButton>
            </div>
          </div>
        </div>

        {/* Main Canvas */}
        <div className="flex-1 relative bg-gray-900/50">
          <DndContext
            sensors={sensors}
            collisionDetection={closestCenter}
            onDragStart={handleDragStart}
            onDragEnd={handleDragEnd}
          >
            {/* Grid Background */}
            <div 
              className="absolute inset-0 opacity-10"
              style={{
                backgroundImage: `
                  linear-gradient(to right, #10b981 1px, transparent 1px),
                  linear-gradient(to bottom, #10b981 1px, transparent 1px)
                `,
                backgroundSize: '20px 20px'
              }}
            />

            {/* Strategy Nodes */}
            <SortableContext items={nodes} strategy={verticalListSortingStrategy}>
              <AnimatePresence>
                {nodes.map((node) => (
                  <motion.div
                    key={node.id}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.8 }}
                    className={`absolute cursor-pointer ${node.color} border-2 bg-black/80 rounded-lg p-4 min-w-48 ${
                      selectedNode?.id === node.id ? 'ring-2 ring-green-500' : ''
                    }`}
                    style={{ left: node.position.x, top: node.position.y }}
                    onClick={() => setSelectedNode(node)}
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <div className="text-green-400">
                        {node.icon}
                      </div>
                      <div className="text-sm font-bold text-green-400">{node.label}</div>
                    </div>
                    
                    <div className="text-xs text-green-600 space-y-1">
                      {Object.entries(node.config).slice(0, 3).map(([key, value]) => (
                        <div key={key}>
                          <span>{key}:</span>
                          <span className="font-mono ml-1">{String(value)}</span>
                        </div>
                      ))}
                    </div>

                    {/* Remove Button */}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        removeNode(node.id);
                      }}
                      className="absolute top-2 right-2 w-5 h-5 bg-red-600 hover:bg-red-700 rounded-full flex items-center justify-center text-white text-xs"
                    >
                      ×
                    </button>

                    {/* Connection Points */}
                    <div className="absolute -right-2 top-1/2 w-4 h-4 bg-green-500 rounded-full border-2 border-black"></div>
                    <div className="absolute -left-2 top-1/2 w-4 h-4 bg-green-500 rounded-full border-2 border-black"></div>
                  </motion.div>
                ))}
              </AnimatePresence>
            </SortableContext>

            {/* Connection Lines */}
            <svg className="absolute inset-0 pointer-events-none">
              <AnimatePresence>
                {connections.map((connection, index) => {
                  const fromNode = nodes.find(n => n.id === connection.from);
                  const toNode = nodes.find(n => n.id === connection.to);
                  
                  if (!fromNode || !toNode) return null;

                  const fromX = fromNode.position.x + 192; // Width of node
                  const fromY = fromNode.position.y + 32; // Middle of node
                  const toX = toNode.position.x;
                  const toY = toNode.position.y + 32;

                  return (
                    <motion.g key={index}>
                      <defs>
                        <marker
                          id={`arrowhead-${index}`}
                          markerWidth="10"
                          markerHeight="7"
                          refX="9"
                          refY="3.5"
                          orient="auto"
                        >
                          <polygon
                            points="0 0, 10 3.5, 0 7"
                            fill="#10b981"
                          />
                        </marker>
                      </defs>
                      <motion.line
                        initial={{ pathLength: 0 }}
                        animate={{ pathLength: 1 }}
                        x1={fromX}
                        y1={fromY}
                        x2={toX}
                        y2={toY}
                        stroke="#10b981"
                        strokeWidth="2"
                        markerEnd={`url(#arrowhead-${index})`}
                      />
                      {connection.label && (
                        <text
                          x={(fromX + toX) / 2}
                          y={(fromY + toY) / 2 - 5}
                          fill="#10b981"
                          textAnchor="middle"
                          className="text-xs font-bold"
                        >
                          {connection.label}
                        </text>
                      )}
                    </motion.g>
                  );
                })}
              </AnimatePresence>
            </svg>

            <DragOverlay>
              {/* Drag overlay content */}
            </DragOverlay>
          </DndContext>
        </div>

        {/* Properties Panel */}
        {selectedNode && (
          <div className="w-80 border-l border-green-800/30 p-4 bg-black/20">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-bold text-green-400">NODE PROPERTIES</h3>
              <button
                onClick={() => setSelectedNode(null)}
                className="text-green-600 hover:text-green-400"
              >
                ×
              </button>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="block text-xs text-green-400 mb-2">Label</label>
                <input
                  type="text"
                  value={selectedNode.label}
                  onChange={(e) => {
                    setNodes(nodes.map(n => 
                      n.id === selectedNode.id 
                        ? { ...n, label: e.target.value }
                        : n
                    ));
                    setSelectedNode({ ...selectedNode, label: e.target.value });
                  }}
                  className="matrix-input w-full px-3 py-2 text-sm"
                />
              </div>

              <div>
                <label className="block text-xs text-green-400 mb-2">Type</label>
                <div className="text-sm text-green-600 capitalize">{selectedNode.type}</div>
              </div>

              <div>
                <h4 className="text-xs text-green-400 mb-2">Configuration</h4>
                <div className="space-y-2">
                  {Object.entries(selectedNode.config).map(([key, value]) => (
                    <div key={key}>
                      <label className="block text-xs text-green-600 mb-1">{key}</label>
                      <input
                        type="text"
                        value={String(value)}
                        onChange={(e) => {
                          const newConfig = { ...selectedNode.config, [key]: e.target.value };
                          const updatedNode = { ...selectedNode, config: newConfig };
                          setNodes(nodes.map(n => 
                            n.id === selectedNode.id ? updatedNode : n
                          ));
                          setSelectedNode(updatedNode);
                        }}
                        className="matrix-input w-full px-2 py-1 text-xs"
                      />
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};