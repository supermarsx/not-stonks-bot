import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MatrixCard } from '@/components/ui/MatrixCard';
import { MatrixButton } from '@/components/ui/MatrixButton';
import { MatrixInput } from '@/components/ui/MatrixInput';
import { 
  Brain, 
  Cpu, 
  MessageSquare, 
  Settings,
  Save,
  Zap,
  Target,
  BarChart3,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  Plus,
  Trash2,
  Copy,
  Download,
  Upload
} from 'lucide-react';

interface LLMProvider {
  id: string;
  name: string;
  type: 'openai' | 'anthropic' | 'google' | 'cohere' | 'custom';
  model: string;
  apiKey: string;
  endpoint?: string;
  parameters: {
    temperature: number;
    maxTokens: number;
    topP: number;
    frequencyPenalty: number;
    presencePenalty: number;
  };
  cost: {
    inputPrice: number; // per 1K tokens
    outputPrice: number; // per 1K tokens
    monthlyLimit?: number;
  };
  features: {
    streaming: boolean;
    functionCalling: boolean;
    vision: boolean;
    fineTuning: boolean;
  };
  status: 'connected' | 'disconnected' | 'error';
  lastUsed?: Date;
}

interface PromptTemplate {
  id: string;
  name: string;
  description: string;
  category: 'signal_generation' | 'risk_assessment' | 'entry_decision' | 'exit_decision' | 'portfolio_optimization' | 'market_analysis';
  prompt: string;
  variables: {
    name: string;
    type: 'string' | 'number' | 'boolean' | 'array' | 'object';
    required: boolean;
    default?: any;
    description: string;
  }[];
  examples: {
    input: Record<string, any>;
    expected: string;
    context: string;
  }[];
  performance: {
    accuracy: number;
    avgResponseTime: number;
    usageCount: number;
    lastUsed?: Date;
  };
  isActive: boolean;
}

interface AICapability {
  id: string;
  name: string;
  description: string;
  type: 'analysis' | 'prediction' | 'optimization' | 'automation' | 'monitoring';
  enabled: boolean;
  priority: number; // 1-10
  llmProvider?: string;
  promptTemplate?: string;
  confidence: number; // 0-1
  lastUpdated?: Date;
  config: Record<string, any>;
}

interface AIConfig {
  id: string;
  name: string;
  description: string;
  version: string;
  llmProviders: LLMProvider[];
  promptTemplates: PromptTemplate[];
  capabilities: AICapability[];
  globalSettings: {
    enabled: boolean;
    primaryProvider: string;
    fallbackProvider: string;
    autoRetry: boolean;
    maxRetries: number;
    timeout: number; // seconds
    costLimit: number; // monthly budget
    usageTracking: boolean;
  };
  decisionEngine: {
    confidenceThreshold: number;
    consensusRequired: boolean;
    minProviders: number;
    votingStrategy: 'majority' | 'unanimous' | 'weighted';
    decisionTimeout: number; // seconds
  };
}

interface AIConfigProps {
  config: AIConfig;
  onChange: (config: AIConfig) => void;
  onSave?: () => void;
}

export const AIConfig: React.FC<AIConfigProps> = ({
  config,
  onChange,
  onSave
}) => {
  const [selectedProvider, setSelectedProvider] = useState<string | null>(null);
  const [selectedTemplate, setSelectedTemplate] = useState<string | null>(null);
  const [selectedCapability, setSelectedCapability] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'providers' | 'prompts' | 'capabilities' | 'decision-engine'>('overview');

  const llmProviders: Record<string, Partial<LLMProvider>> = {
    openai: {
      name: 'OpenAI',
      type: 'openai',
      model: 'gpt-4',
      parameters: { temperature: 0.7, maxTokens: 4096, topP: 1, frequencyPenalty: 0, presencePenalty: 0 },
      cost: { inputPrice: 0.03, outputPrice: 0.06 },
      features: { streaming: true, functionCalling: true, vision: true, fineTuning: false }
    },
    anthropic: {
      name: 'Anthropic',
      type: 'anthropic',
      model: 'claude-3-opus',
      parameters: { temperature: 0.7, maxTokens: 4096, topP: 1, frequencyPenalty: 0, presencePenalty: 0 },
      cost: { inputPrice: 0.015, outputPrice: 0.075 },
      features: { streaming: true, functionCalling: true, vision: true, fineTuning: false }
    },
    google: {
      name: 'Google AI',
      type: 'google',
      model: 'gemini-pro',
      parameters: { temperature: 0.7, maxTokens: 4096, topP: 1, frequencyPenalty: 0, presencePenalty: 0 },
      cost: { inputPrice: 0.0005, outputPrice: 0.0015 },
      features: { streaming: true, functionCalling: true, vision: true, fineTuning: false }
    }
  };

  const capabilityTypes = {
    analysis: {
      name: 'Analysis',
      icon: <BarChart3 className="w-4 h-4" />,
      description: 'Market and data analysis'
    },
    prediction: {
      name: 'Prediction',
      icon: <TrendingUp className="w-4 h-4" />,
      description: 'Price and trend prediction'
    },
    optimization: {
      name: 'Optimization',
      icon: <Target className="w-4 h-4" />,
      description: 'Portfolio and strategy optimization'
    },
    automation: {
      name: 'Automation',
      icon: <Zap className="w-4 h-4" />,
      description: 'Automated decision making'
    },
    monitoring: {
      name: 'Monitoring',
      icon: <AlertCircle className="w-4 h-4" />,
      description: 'System monitoring and alerts'
    }
  };

  const promptCategories = {
    signal_generation: {
      name: 'Signal Generation',
      description: 'Generate trading signals from market data'
    },
    risk_assessment: {
      name: 'Risk Assessment',
      description: 'Assess risk levels and portfolio exposure'
    },
    entry_decision: {
      name: 'Entry Decision',
      description: 'Decide when to enter positions'
    },
    exit_decision: {
      name: 'Exit Decision',
      description: 'Decide when to exit positions'
    },
    portfolio_optimization: {
      name: 'Portfolio Optimization',
      description: 'Optimize portfolio allocation'
    },
    market_analysis: {
      name: 'Market Analysis',
      description: 'Analyze market conditions and trends'
    }
  };

  const createLLMProvider = (type: string): LLMProvider => {
    const providerTemplate = llmProviders[type];
    return {
      id: `provider_${Date.now()}`,
      name: providerTemplate.name || 'New Provider',
      type: providerTemplate.type || 'openai',
      model: providerTemplate.model || 'gpt-4',
      apiKey: '',
      endpoint: providerTemplate.endpoint,
      parameters: providerTemplate.parameters || { temperature: 0.7, maxTokens: 4096, topP: 1, frequencyPenalty: 0, presencePenalty: 0 },
      cost: providerTemplate.cost || { inputPrice: 0.03, outputPrice: 0.06 },
      features: providerTemplate.features || { streaming: true, functionCalling: true, vision: false, fineTuning: false },
      status: 'disconnected'
    };
  };

  const createPromptTemplate = (category: keyof typeof promptCategories): PromptTemplate => {
    const categoryInfo = promptCategories[category];
    return {
      id: `prompt_${Date.now()}`,
      name: `${categoryInfo.name} Template`,
      description: categoryInfo.description,
      category,
      prompt: getDefaultPrompt(category),
      variables: getDefaultVariables(category),
      examples: [],
      performance: { accuracy: 0, avgResponseTime: 0, usageCount: 0 },
      isActive: true
    };
  };

  const createCapability = (type: keyof typeof capabilityTypes): AICapability => {
    const typeInfo = capabilityTypes[type];
    return {
      id: `capability_${Date.now()}`,
      name: `${typeInfo.name} Capability`,
      description: typeInfo.description,
      type,
      enabled: true,
      priority: 5,
      confidence: 0.5,
      config: getDefaultCapabilityConfig(type)
    };
  };

  const getDefaultPrompt = (category: keyof typeof promptCategories): string => {
    const prompts = {
      signal_generation: `You are a trading signal generator. Analyze the provided market data and generate a trading signal.

Market Data: {{market_data}}
Technical Indicators: {{indicators}}
Current Price: {{price}}
Volume: {{volume}}

Provide your analysis and signal in JSON format:
{
  "signal": "BUY" | "SELL" | "HOLD",
  "confidence": 0.0-1.0,
  "reasoning": "explanation",
  "targets": ["price1", "price2"],
  "stop_loss": "price",
  "risk_level": "LOW" | "MEDIUM" | "HIGH"
}`,
      
      risk_assessment: `You are a risk assessment AI. Evaluate the risk level of the current portfolio and positions.

Portfolio Data: {{portfolio_data}}
Positions: {{positions}}
Market Conditions: {{market_conditions}}

Analyze and provide risk assessment:
{
  "risk_score": 0.0-1.0,
  "risk_level": "LOW" | "MEDIUM" | "HIGH",
  "concerns": ["concern1", "concern2"],
  "recommendations": ["rec1", "rec2"],
  "max_loss_potential": "amount"
}`,
      
      entry_decision: `You are an entry decision AI. Determine if this is a good time to enter a position.

Asset: {{asset}}
Current Analysis: {{analysis}}
Market Conditions: {{conditions}}
Risk Parameters: {{risk_params}}

Provide entry decision:
{
  "decision": "ENTER" | "WAIT" | "AVOID",
  "confidence": 0.0-1.0,
  "entry_price": "price",
  "position_size": "percentage",
  "reasoning": "explanation"
}`,
      
      exit_decision: `You are an exit decision AI. Determine if this is a good time to exit a position.

Position: {{position}}
Current P&L: {{pnl}}
Market Changes: {{market_changes}}
Time Factors: {{time_factors}}

Provide exit decision:
{
  "decision": "EXIT" | "HOLD" | "ADJUST",
  "exit_price": "price",
  "reasoning": "explanation",
  "alternative_actions": ["action1", "action2"]
}`,
      
      portfolio_optimization: `You are a portfolio optimization AI. Suggest optimal allocation changes.

Current Portfolio: {{current_portfolio}}
Target Allocation: {{target_allocation}}
Market Data: {{market_data}}
Risk Profile: {{risk_profile}}

Provide optimization suggestions:
{
  "recommended_changes": [
    {"asset": "symbol", "action": "BUY" | "SELL", "percentage": "value"},
  ],
  "expected_improvement": "description",
  "risk_impact": "description"
}`,
      
      market_analysis: `You are a market analysis AI. Provide comprehensive market analysis.

Market Data: {{market_data}}
News Sentiment: {{sentiment}}
Economic Indicators: {{indicators}}
Technical Analysis: {{technical}}

Provide analysis:
{
  "overall_sentiment": "BULLISH" | "BEARISH" | "NEUTRAL",
  "key_factors": ["factor1", "factor2"],
  "outlook": "short_term_outlook",
  "recommendations": ["rec1", "rec2"]
}`
    };
    
    return prompts[category] || '';
  };

  const getDefaultVariables = (category: keyof typeof promptCategories) => {
    const variables = {
      signal_generation: [
        { name: 'market_data', type: 'object', required: true, description: 'Current market data' },
        { name: 'indicators', type: 'object', required: true, description: 'Technical indicators' },
        { name: 'price', type: 'number', required: true, description: 'Current price' },
        { name: 'volume', type: 'number', required: true, description: 'Trading volume' }
      ],
      risk_assessment: [
        { name: 'portfolio_data', type: 'object', required: true, description: 'Portfolio information' },
        { name: 'positions', type: 'array', required: true, description: 'Current positions' },
        { name: 'market_conditions', type: 'object', required: true, description: 'Market conditions' }
      ],
      entry_decision: [
        { name: 'asset', type: 'string', required: true, description: 'Asset to trade' },
        { name: 'analysis', type: 'object', required: true, description: 'Current analysis' },
        { name: 'conditions', type: 'object', required: true, description: 'Market conditions' },
        { name: 'risk_params', type: 'object', required: false, description: 'Risk parameters' }
      ],
      exit_decision: [
        { name: 'position', type: 'object', required: true, description: 'Position to evaluate' },
        { name: 'pnl', type: 'number', required: true, description: 'Current P&L' },
        { name: 'market_changes', type: 'object', required: true, description: 'Recent market changes' },
        { name: 'time_factors', type: 'object', required: false, description: 'Time-related factors' }
      ],
      portfolio_optimization: [
        { name: 'current_portfolio', type: 'object', required: true, description: 'Current holdings' },
        { name: 'target_allocation', type: 'object', required: true, description: 'Target allocation' },
        { name: 'market_data', type: 'object', required: true, description: 'Market data' },
        { name: 'risk_profile', type: 'object', required: true, description: 'Risk profile' }
      ],
      market_analysis: [
        { name: 'market_data', type: 'object', required: true, description: 'Market data' },
        { name: 'sentiment', type: 'object', required: false, description: 'News sentiment' },
        { name: 'indicators', type: 'object', required: false, description: 'Economic indicators' },
        { name: 'technical', type: 'object', required: false, description: 'Technical analysis' }
      ]
    };
    
    return variables[category] || [];
  };

  const getDefaultCapabilityConfig = (type: keyof typeof capabilityTypes) => {
    const configs = {
      analysis: { frequency: 'realtime', depth: 'detailed', dataSources: [] },
      prediction: { horizon: '1d', confidence_level: 0.8, model_type: 'ensemble' },
      optimization: { objective: 'sharpe', constraints: {}, max_iterations: 100 },
      automation: { auto_execute: false, confirm_threshold: 0.8, max_slippage: 0.01 },
      monitoring: { alert_frequency: 'immediate', thresholds: {}, notification_channels: [] }
    };
    
    return configs[type] || {};
  };

  const addLLMProvider = (type: string) => {
    const newProvider = createLLMProvider(type);
    onChange({
      ...config,
      llmProviders: [...config.llmProviders, newProvider]
    });
    setSelectedProvider(newProvider.id);
  };

  const addPromptTemplate = (category: keyof typeof promptCategories) => {
    const newTemplate = createPromptTemplate(category);
    onChange({
      ...config,
      promptTemplates: [...config.promptTemplates, newTemplate]
    });
    setSelectedTemplate(newTemplate.id);
  };

  const addCapability = (type: keyof typeof capabilityTypes) => {
    const newCapability = createCapability(type);
    onChange({
      ...config,
      capabilities: [...config.capabilities, newCapability]
    });
    setSelectedCapability(newCapability.id);
  };

  const updateLLMProvider = (providerId: string, updates: Partial<LLMProvider>) => {
    onChange({
      ...config,
      llmProviders: config.llmProviders.map(provider =>
        provider.id === providerId ? { ...provider, ...updates } : provider
      )
    });
  };

  const updatePromptTemplate = (templateId: string, updates: Partial<PromptTemplate>) => {
    onChange({
      ...config,
      promptTemplates: config.promptTemplates.map(template =>
        template.id === templateId ? { ...template, ...updates } : template
      )
    });
  };

  const updateCapability = (capabilityId: string, updates: Partial<AICapability>) => {
    onChange({
      ...config,
      capabilities: config.capabilities.map(capability =>
        capability.id === capabilityId ? { ...capability, ...updates } : capability
      )
    });
  };

  const removeLLMProvider = (providerId: string) => {
    onChange({
      ...config,
      llmProviders: config.llmProviders.filter(provider => provider.id !== providerId)
    });
    if (selectedProvider === providerId) {
      setSelectedProvider(null);
    }
  };

  const removePromptTemplate = (templateId: string) => {
    onChange({
      ...config,
      promptTemplates: config.promptTemplates.filter(template => template.id !== templateId)
    });
    if (selectedTemplate === templateId) {
      setSelectedTemplate(null);
    }
  };

  const removeCapability = (capabilityId: string) => {
    onChange({
      ...config,
      capabilities: config.capabilities.filter(capability => capability.id !== capabilityId)
    });
    if (selectedCapability === capabilityId) {
      setSelectedCapability(null);
    }
  };

  const renderLLMProviderConfig = (provider: LLMProvider) => {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-green-400 mb-1">Provider Name</label>
            <MatrixInput
              value={provider.name}
              onChange={(e) => updateLLMProvider(provider.id, { name: e.target.value })}
              className="text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-green-400 mb-1">Model</label>
            <select
              value={provider.model}
              onChange={(e) => updateLLMProvider(provider.id, { model: e.target.value })}
              className="matrix-input w-full px-3 py-2 text-sm"
            >
              <option value="gpt-4">GPT-4</option>
              <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
              <option value="claude-3-opus">Claude 3 Opus</option>
              <option value="claude-3-sonnet">Claude 3 Sonnet</option>
              <option value="gemini-pro">Gemini Pro</option>
            </select>
          </div>
        </div>

        <div>
          <label className="block text-xs text-green-400 mb-1">API Key</label>
          <MatrixInput
            type="password"
            value={provider.apiKey}
            onChange={(e) => updateLLMProvider(provider.id, { apiKey: e.target.value })}
            className="text-sm"
            placeholder="Enter API key..."
          />
        </div>

        <div className="space-y-4">
          <h4 className="text-sm font-bold text-green-400">Model Parameters</h4>
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-green-400 mb-1">Temperature (0-2)</label>
              <div className="relative">
                <MatrixInput
                  type="number"
                  step="0.1"
                  min="0"
                  max="2"
                  value={provider.parameters.temperature}
                  onChange={(e) => updateLLMProvider(provider.id, {
                    parameters: { ...provider.parameters, temperature: parseFloat(e.target.value) }
                  })}
                  className="text-sm"
                />
              </div>
            </div>
            <div>
              <label className="block text-xs text-green-400 mb-1">Max Tokens</label>
              <MatrixInput
                type="number"
                min="100"
                max="32768"
                value={provider.parameters.maxTokens}
                onChange={(e) => updateLLMProvider(provider.id, {
                  parameters: { ...provider.parameters, maxTokens: parseInt(e.target.value) }
                })}
                className="text-sm"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-green-400 mb-1">Top P (0-1)</label>
              <MatrixInput
                type="number"
                step="0.1"
                min="0"
                max="1"
                value={provider.parameters.topP}
                onChange={(e) => updateLLMProvider(provider.id, {
                  parameters: { ...provider.parameters, topP: parseFloat(e.target.value) }
                })}
                className="text-sm"
              />
            </div>
            <div>
              <label className="block text-xs text-green-400 mb-1">Frequency Penalty (-2 to 2)</label>
              <MatrixInput
                type="number"
                step="0.1"
                min="-2"
                max="2"
                value={provider.parameters.frequencyPenalty}
                onChange={(e) => updateLLMProvider(provider.id, {
                  parameters: { ...provider.parameters, frequencyPenalty: parseFloat(e.target.value) }
                })}
                className="text-sm"
              />
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <h4 className="text-sm font-bold text-green-400">Cost Configuration</h4>
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-green-400 mb-1">Input Price ($/1K tokens)</label>
              <MatrixInput
                type="number"
                step="0.001"
                min="0"
                value={provider.cost.inputPrice}
                onChange={(e) => updateLLMProvider(provider.id, {
                  cost: { ...provider.cost, inputPrice: parseFloat(e.target.value) }
                })}
                className="text-sm"
              />
            </div>
            <div>
              <label className="block text-xs text-green-400 mb-1">Output Price ($/1K tokens)</label>
              <MatrixInput
                type="number"
                step="0.001"
                min="0"
                value={provider.cost.outputPrice}
                onChange={(e) => updateLLMProvider(provider.id, {
                  cost: { ...provider.cost, outputPrice: parseFloat(e.target.value) }
                })}
                className="text-sm"
              />
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <h4 className="text-sm font-bold text-green-400">Features</h4>
          
          <div className="grid grid-cols-2 gap-4">
            {Object.entries(provider.features).map(([feature, enabled]) => (
              <label key={feature} className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={enabled}
                  onChange={(e) => updateLLMProvider(provider.id, {
                    features: { ...provider.features, [feature]: e.target.checked }
                  })}
                  className="w-4 h-4 accent-green-500"
                />
                <span className="text-sm text-green-400 capitalize">
                  {feature.replace(/_/g, ' ')}
                </span>
              </label>
            ))}
          </div>
        </div>

        <div className="matrix-card p-3 bg-black/30">
          <div className="flex items-center justify-between">
            <span className="text-xs text-green-400">Status</span>
            <div className="flex items-center gap-2">
              {provider.status === 'connected' ? (
                <CheckCircle className="w-4 h-4 text-green-400" />
              ) : provider.status === 'error' ? (
                <AlertCircle className="w-4 h-4 text-red-400" />
              ) : (
                <AlertCircle className="w-4 h-4 text-gray-400" />
              )}
              <span className={`text-xs font-bold ${
                provider.status === 'connected' ? 'text-green-400' :
                provider.status === 'error' ? 'text-red-400' : 'text-gray-400'
              }`}>
                {provider.status.toUpperCase()}
              </span>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderPromptTemplateConfig = (template: PromptTemplate) => {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-green-400 mb-1">Template Name</label>
            <MatrixInput
              value={template.name}
              onChange={(e) => updatePromptTemplate(template.id, { name: e.target.value })}
              className="text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-green-400 mb-1">Category</label>
            <select
              value={template.category}
              onChange={(e) => updatePromptTemplate(template.id, { category: e.target.value as PromptTemplate['category'] })}
              className="matrix-input w-full px-3 py-2 text-sm"
            >
              {Object.entries(promptCategories).map(([key, info]) => (
                <option key={key} value={key}>
                  {info.name}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div>
          <label className="block text-xs text-green-400 mb-1">Description</label>
          <textarea
            value={template.description}
            onChange={(e) => updatePromptTemplate(template.id, { description: e.target.value })}
            className="matrix-input w-full px-3 py-2 text-sm h-20 resize-none"
            placeholder="Describe this prompt template..."
          />
        </div>

        <div>
          <label className="block text-xs text-green-400 mb-1">Prompt Template</label>
          <textarea
            value={template.prompt}
            onChange={(e) => updatePromptTemplate(template.id, { prompt: e.target.value })}
            className="matrix-input w-full px-3 py-2 text-sm h-40 resize-none font-mono"
            placeholder="Enter your prompt template with variables like {{variable_name}}..."
          />
          <p className="text-xs text-green-600 mt-1">
            Use double curly braces for variables: {'{{'}variable_name{'}}'}
          </p>
        </div>

        <div className="space-y-4">
          <h4 className="text-sm font-bold text-green-400">
            Variables ({template.variables.length})
          </h4>
          
          {template.variables.map((variable, index) => (
            <div key={index} className="matrix-card p-4">
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <label className="block text-xs text-green-400 mb-1">Name</label>
                  <MatrixInput
                    value={variable.name}
                    onChange={(e) => {
                      const newVariables = [...template.variables];
                      newVariables[index] = { ...variable, name: e.target.value };
                      updatePromptTemplate(template.id, { variables: newVariables });
                    }}
                    className="text-sm"
                  />
                </div>
                <div>
                  <label className="block text-xs text-green-400 mb-1">Type</label>
                  <select
                    value={variable.type}
                    onChange={(e) => {
                      const newVariables = [...template.variables];
                      newVariables[index] = { ...variable, type: e.target.value as any };
                      updatePromptTemplate(template.id, { variables: newVariables });
                    }}
                    className="matrix-input w-full px-3 py-2 text-sm"
                  >
                    <option value="string">String</option>
                    <option value="number">Number</option>
                    <option value="boolean">Boolean</option>
                    <option value="array">Array</option>
                    <option value="object">Object</option>
                  </select>
                </div>
                <div className="flex items-end">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={variable.required}
                      onChange={(e) => {
                        const newVariables = [...template.variables];
                        newVariables[index] = { ...variable, required: e.target.checked };
                        updatePromptTemplate(template.id, { variables: newVariables });
                      }}
                      className="w-4 h-4 accent-green-500"
                    />
                    <span className="text-sm text-green-400">Required</span>
                  </label>
                </div>
              </div>
              
              <div className="mt-2">
                <label className="block text-xs text-green-400 mb-1">Description</label>
                <MatrixInput
                  value={variable.description}
                  onChange={(e) => {
                    const newVariables = [...template.variables];
                    newVariables[index] = { ...variable, description: e.target.value };
                    updatePromptTemplate(template.id, { variables: newVariables });
                  }}
                  className="text-sm"
                />
              </div>
            </div>
          ))}
          
          <MatrixButton
            size="sm"
            onClick={() => {
              const newVariables = [...template.variables, {
                name: `variable_${template.variables.length + 1}`,
                type: 'string' as const,
                required: false,
                description: 'New variable'
              }];
              updatePromptTemplate(template.id, { variables: newVariables });
            }}
          >
            <Plus className="w-3 h-3 mr-2" />
            Add Variable
          </MatrixButton>
        </div>

        <div className="matrix-card p-3 bg-black/30">
          <div className="flex items-center justify-between">
            <span className="text-xs text-green-400">Performance</span>
            <span className="text-xs font-bold text-green-400">
              {template.performance.usageCount} uses
            </span>
          </div>
          
          <div className="grid grid-cols-2 gap-4 mt-2">
            <div>
              <div className="text-xs text-green-600">Accuracy</div>
              <div className="text-sm font-bold text-green-400">
                {(template.performance.accuracy * 100).toFixed(1)}%
              </div>
            </div>
            <div>
              <div className="text-xs text-green-600">Avg Response Time</div>
              <div className="text-sm font-bold text-green-400">
                {template.performance.avgResponseTime.toFixed(0)}ms
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderCapabilityConfig = (capability: AICapability) => {
    const typeInfo = capabilityTypes[capability.type];
    
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-green-400 mb-1">Capability Name</label>
            <MatrixInput
              value={capability.name}
              onChange={(e) => updateCapability(capability.id, { name: e.target.value })}
              className="text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-green-400 mb-1">Type</label>
            <select
              value={capability.type}
              onChange={(e) => updateCapability(capability.id, { type: e.target.value as AICapability['type'] })}
              className="matrix-input w-full px-3 py-2 text-sm"
            >
              {Object.entries(capabilityTypes).map(([key, info]) => (
                <option key={key} value={key}>
                  {info.name}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div>
          <label className="block text-xs text-green-400 mb-1">Description</label>
          <textarea
            value={capability.description}
            onChange={(e) => updateCapability(capability.id, { description: e.target.value })}
            className="matrix-input w-full px-3 py-2 text-sm h-20 resize-none"
            placeholder="Describe this AI capability..."
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-green-400 mb-1">Priority (1-10)</label>
            <MatrixInput
              type="number"
              min="1"
              max="10"
              value={capability.priority}
              onChange={(e) => updateCapability(capability.id, { priority: parseInt(e.target.value) })}
              className="text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-green-400 mb-1">Confidence (0-1)</label>
            <MatrixInput
              type="number"
              step="0.01"
              min="0"
              max="1"
              value={capability.confidence}
              onChange={(e) => updateCapability(capability.id, { confidence: parseFloat(e.target.value) })}
              className="text-sm"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-green-400 mb-1">LLM Provider</label>
            <select
              value={capability.llmProvider || ''}
              onChange={(e) => updateCapability(capability.id, { llmProvider: e.target.value || undefined })}
              className="matrix-input w-full px-3 py-2 text-sm"
            >
              <option value="">Select Provider</option>
              {config.llmProviders.map(provider => (
                <option key={provider.id} value={provider.id}>
                  {provider.name}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs text-green-400 mb-1">Prompt Template</label>
            <select
              value={capability.promptTemplate || ''}
              onChange={(e) => updateCapability(capability.id, { promptTemplate: e.target.value || undefined })}
              className="matrix-input w-full px-3 py-2 text-sm"
            >
              <option value="">Select Template</option>
              {config.promptTemplates.map(template => (
                <option key={template.id} value={template.id}>
                  {template.name}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="matrix-card p-3 bg-black/30">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-green-400">Status</span>
            <div className="flex items-center gap-2">
              {capability.enabled ? (
                <CheckCircle className="w-4 h-4 text-green-400" />
              ) : (
                <AlertCircle className="w-4 h-4 text-gray-400" />
              )}
              <span className={`text-xs font-bold ${
                capability.enabled ? 'text-green-400' : 'text-gray-400'
              }`}>
                {capability.enabled ? 'ENABLED' : 'DISABLED'}
              </span>
            </div>
          </div>
          
          {capability.lastUpdated && (
            <div className="text-xs text-green-600">
              Last updated: {capability.lastUpdated.toLocaleString()}
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-green-800/30">
        <div>
          <h1 className="text-2xl font-bold matrix-text-glow text-green-400">
            AI STRATEGY CONFIGURATION
          </h1>
          <p className="text-green-600 text-sm">Configure AI providers, prompts, and capabilities</p>
        </div>
        
        <div className="flex items-center gap-2">
          <div className="text-right mr-4">
            <div className="text-xs text-green-600">Active Providers</div>
            <div className="text-lg font-bold text-green-400">
              {config.llmProviders.filter(p => p.status === 'connected').length}
            </div>
          </div>
          
          <div className="text-right mr-4">
            <div className="text-xs text-green-600">Enabled Capabilities</div>
            <div className="text-lg font-bold text-green-400">
              {config.capabilities.filter(c => c.enabled).length}
            </div>
          </div>
          
          {onSave && (
            <MatrixButton onClick={onSave}>
              <Save className="w-4 h-4 mr-2" />
              Save Configuration
            </MatrixButton>
          )}
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex border-b border-green-800/30">
        {[
          { id: 'overview', label: 'Overview', icon: <Brain className="w-4 h-4" /> },
          { id: 'providers', label: 'LLM Providers', icon: <Cpu className="w-4 h-4" /> },
          { id: 'prompts', label: 'Prompt Templates', icon: <MessageSquare className="w-4 h-4" /> },
          { id: 'capabilities', label: 'Capabilities', icon: <Zap className="w-4 h-4" /> },
          { id: 'decision-engine', label: 'Decision Engine', icon: <Target className="w-4 h-4" /> }
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors ${
              activeTab === tab.id 
                ? 'text-green-400 border-b-2 border-green-500' 
                : 'text-green-600 hover:text-green-400'
            }`}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel - Sidebar */}
        <div className="w-80 border-r border-green-800/30 flex flex-col">
          {/* Global Settings */}
          <div className="p-4 border-b border-green-800/30">
            <h3 className="text-sm font-bold text-green-400 mb-3">GLOBAL SETTINGS</h3>
            
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs text-green-400">AI Enabled</span>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={config.globalSettings.enabled}
                    onChange={(e) => onChange({
                      ...config,
                      globalSettings: { ...config.globalSettings, enabled: e.target.checked }
                    })}
                    className="w-4 h-4 accent-green-500"
                  />
                </label>
              </div>
              
              <div>
                <label className="block text-xs text-green-400 mb-1">Primary Provider</label>
                <select
                  value={config.globalSettings.primaryProvider}
                  onChange={(e) => onChange({
                    ...config,
                    globalSettings: { ...config.globalSettings, primaryProvider: e.target.value }
                  })}
                  className="matrix-input w-full px-3 py-2 text-xs"
                >
                  <option value="">Select Provider</option>
                  {config.llmProviders.map(provider => (
                    <option key={provider.id} value={provider.id}>
                      {provider.name}
                    </option>
                  ))}
                </select>
              </div>
              
              <div>
                <label className="block text-xs text-green-400 mb-1">Cost Limit ($/month)</label>
                <MatrixInput
                  type="number"
                  min="0"
                  value={config.globalSettings.costLimit}
                  onChange={(e) => onChange({
                    ...config,
                    globalSettings: { ...config.globalSettings, costLimit: parseFloat(e.target.value) }
                  })}
                  className="text-sm"
                />
              </div>
            </div>
          </div>

          {/* Quick Add */}
          <div className="flex-1 overflow-y-auto p-4">
            <h3 className="text-sm font-bold text-green-400 mb-3">QUICK ADD</h3>
            
            <div className="space-y-2">
              <MatrixButton 
                size="sm" 
                onClick={() => addLLMProvider('openai')}
                className="w-full justify-start"
              >
                <Plus className="w-3 h-3 mr-2" />
                Add OpenAI Provider
              </MatrixButton>
              
              <MatrixButton 
                size="sm" 
                onClick={() => addPromptTemplate('signal_generation')}
                className="w-full justify-start"
              >
                <Plus className="w-3 h-3 mr-2" />
                Add Signal Generator
              </MatrixButton>
              
              <MatrixButton 
                size="sm" 
                onClick={() => addCapability('analysis')}
                className="w-full justify-start"
              >
                <Plus className="w-3 h-3 mr-2" />
                Add Analysis Capability
              </MatrixButton>
            </div>
          </div>
        </div>

        {/* Right Panel - Main Content */}
        <div className="flex-1 overflow-y-auto">
          <AnimatePresence mode="wait">
            {activeTab === 'overview' && (
              <motion.div
                key="overview"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="p-6"
              >
                <div className="space-y-6">
                  {/* AI System Status */}
                  <MatrixCard className="p-6">
                    <h3 className="text-lg font-bold text-green-400 mb-4">
                      AI System Status
                    </h3>
                    
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div>
                        <div className="text-xs text-green-600">Total Providers</div>
                        <div className="text-2xl font-bold text-green-400">
                          {config.llmProviders.length}
                        </div>
                      </div>
                      <div>
                        <div className="text-xs text-green-600">Connected</div>
                        <div className="text-2xl font-bold text-green-400">
                          {config.llmProviders.filter(p => p.status === 'connected').length}
                        </div>
                      </div>
                      <div>
                        <div className="text-xs text-green-600">Active Capabilities</div>
                        <div className="text-2xl font-bold text-green-400">
                          {config.capabilities.filter(c => c.enabled).length}
                        </div>
                      </div>
                      <div>
                        <div className="text-xs text-green-600">Prompt Templates</div>
                        <div className="text-2xl font-bold text-green-400">
                          {config.promptTemplates.filter(t => t.isActive).length}
                        </div>
                      </div>
                    </div>
                  </MatrixCard>

                  {/* Provider Performance */}
                  <MatrixCard className="p-6">
                    <h3 className="text-lg font-bold text-green-400 mb-4">
                      Provider Performance
                    </h3>
                    
                    <div className="space-y-3">
                      {config.llmProviders.map((provider) => (
                        <div key={provider.id} className="matrix-card p-4">
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                              <div className="text-green-400">
                                <Cpu className="w-4 h-4" />
                              </div>
                              <span className="text-sm font-bold text-green-400">
                                {provider.name}
                              </span>
                              <span className="text-xs text-green-600">
                                ({provider.model})
                              </span>
                            </div>
                            
                            <div className="flex items-center gap-2">
                              {provider.status === 'connected' ? (
                                <CheckCircle className="w-4 h-4 text-green-400" />
                              ) : (
                                <AlertCircle className="w-4 h-4 text-gray-400" />
                              )}
                              <span className={`text-xs ${
                                provider.status === 'connected' ? 'text-green-400' : 'text-gray-400'
                              }`}>
                                {provider.status.toUpperCase()}
                              </span>
                            </div>
                          </div>
                          
                          <div className="grid grid-cols-3 gap-4 text-xs">
                            <div>
                              <div className="text-green-600">Temperature</div>
                              <div className="text-green-400 font-bold">{provider.parameters.temperature}</div>
                            </div>
                            <div>
                              <div className="text-green-600">Max Tokens</div>
                              <div className="text-green-400 font-bold">{provider.parameters.maxTokens}</div>
                            </div>
                            <div>
                              <div className="text-green-600">Cost/1K</div>
                              <div className="text-green-400 font-bold">
                                ${provider.cost.inputPrice.toFixed(3)}/${provider.cost.outputPrice.toFixed(3)}
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </MatrixCard>

                  {/* Capabilities Overview */}
                  <MatrixCard className="p-6">
                    <h3 className="text-lg font-bold text-green-400 mb-4">
                      AI Capabilities
                    </h3>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {Object.entries(capabilityTypes).map(([type, typeInfo]) => {
                        const capabilities = config.capabilities.filter(c => c.type === type);
                        const avgConfidence = capabilities.length > 0 
                          ? capabilities.reduce((sum, c) => sum + c.confidence, 0) / capabilities.length 
                          : 0;
                        
                        return (
                          <div key={type} className="matrix-card p-4">
                            <div className="flex items-center gap-2 mb-2">
                              <div className="text-green-400">
                                {typeInfo.icon}
                              </div>
                              <div>
                                <div className="text-sm font-bold text-green-400">
                                  {typeInfo.name}
                                </div>
                                <div className="text-xs text-green-600">
                                  {capabilities.length} capabilities
                                </div>
                              </div>
                            </div>
                            
                            <div className="space-y-2">
                              <div className="flex justify-between text-xs">
                                <span className="text-green-600">Avg Confidence</span>
                                <span className="text-green-400 font-bold">
                                  {(avgConfidence * 100).toFixed(1)}%
                                </span>
                              </div>
                              
                              <div className="w-full bg-gray-700 rounded-full h-2">
                                <div 
                                  className="h-2 bg-green-500 rounded-full"
                                  style={{ width: `${avgConfidence * 100}%` }}
                                />
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </MatrixCard>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
};