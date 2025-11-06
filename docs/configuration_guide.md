# Configuration Guide

Comprehensive guide for configuring and customizing the Day Trading Orchestrator system. This guide covers all configuration options, from basic setup to advanced customization.

## 📋 Table of Contents

1. [Configuration Overview](#configuration-overview)
2. [Basic Configuration](#basic-configuration)
3. [Broker Configuration](#broker-configuration)
4. [Risk Management Configuration](#risk-management-configuration)
5. [Strategy Configuration](#strategy-configuration)
6. [AI Configuration](#ai-configuration)
7. [UI Configuration](#ui-configuration)
8. [Performance Optimization](#performance-optimization)
9. [Security Configuration](#security-configuration)
10. [Database Configuration](#database-configuration)
11. [Logging Configuration](#logging-configuration)
12. [Advanced Customization](#advanced-customization)

---

## 🔧 Configuration Overview

### Configuration Files

The Day Trading Orchestrator uses multiple configuration files:

1. **config.json** - Main configuration file
2. **.env** - Environment variables
3. **brokers/*** - Individual broker configurations
4. **strategies/*** - Strategy-specific configurations
5. **ai/prompts/*** - AI prompt templates

### Configuration Hierarchy

```
System Configuration (config.json)
├── Environment Variables (.env)
│   ├── Override system defaults
│   └── Sensitive information
├── User Preferences (database)
│   ├── UI settings
│   ├── API preferences
│   └── Custom configurations
└── Runtime Configuration (in-memory)
    ├── Dynamic settings
    ├── Session-specific values
    └── Performance metrics
```

### Configuration Validation

```bash
# Validate configuration
python validate_config.py

# Check specific sections
python validate_config.py --section brokers
python validate_config.py --section risk
python validate_config.py --section ai
```

---

## ⚙️ Basic Configuration

### Main Configuration File (config.json)

**Structure Overview:**
```json
{
  "database": { ... },
  "ai": { ... },
  "brokers": { ... },
  "risk": { ... },
  "strategies": { ... },
  "market_data": { ... },
  "logging": { ... },
  "ui": { ... },
  "monitoring": { ... },
  "security": { ... },
  "performance": { ... }
}
```

### Environment Variables (.env)

**Create .env file:**
```bash
# Copy example file
cp .env.example .env

# Edit with your settings
nano .env
```

**Essential Environment Variables:**
```bash
# Trading Mode (ALWAYS start with PAPER)
ENVIRONMENT=development
TRADING_MODE=PAPER

# Database
DATABASE_URL=sqlite:///trading_orchestrator.db

# API Keys (Never commit these!)
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_secret
ALPACA_API_KEY=your_alpaca_key
ALPACA_API_SECRET=your_alpaca_secret

# AI Services
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Security
SECRET_KEY=your-secret-key-change-in-production
ENCRYPTION_KEY=your-encryption-key-32-chars
```

### Quick Setup Wizard

```bash
# Run setup wizard
python setup_wizard.py

# Interactive configuration
python -c "
from config.wizard import ConfigWizard
wizard = ConfigWizard()
wizard.run_interactive_setup()
"
```

---

## 🔗 Broker Configuration

### Broker Configuration Structure

```json
{
  "brokers": {
    "broker_name": {
      "enabled": true,
      "api_key": "YOUR_API_KEY",
      "secret_key": "YOUR_SECRET",
      "paper": true,
      "rate_limit": 1200,
      "config": {
        "base_url": "https://api.broker.com",
        "timeout": 30
      }
    }
  }
}
```

### Binance Configuration

**Setup for Binance:**
```json
{
  "brokers": {
    "binance": {
      "enabled": true,
      "api_key": "YOUR_BINANCE_API_KEY",
      "secret_key": "YOUR_BINANCE_SECRET",
      "testnet": true,
      "base_url": "https://testnet.binance.vision",
      "rate_limit": 1200,
      "config": {
        "time_sync_interval": 60,
        "recv_window": 5000,
        "testnet": true
      }
    }
  }
}
```

**Environment Variables:**
```bash
BINANCE_API_KEY=your_binance_key
BINANCE_API_SECRET=your_binance_secret
BINANCE_TESTNET=true
```

**Advanced Configuration:**
```json
{
  "brokers": {
    "binance": {
      "enabled": true,
      "api_key": "${BINANCE_API_KEY}",
      "secret_key": "${BINANCE_API_SECRET}",
      "config": {
        "base_url": "${BINANCE_BASE_URL:-https://api.binance.com}",
        "testnet": "${BINANCE_TESTNET:-false}",
        "rate_limit": 1200,
        "timeout": 30,
        "retry_attempts": 3,
        "retry_delay": 1.0,
        "time_sync_interval": 60,
        "recv_window": 5000,
        "websocket": {
          "enabled": true,
          "url": "wss://stream.binance.com:9443/ws",
          "ping_interval": 180,
          "reconnect_attempts": 5
        }
      }
    }
  }
}
```

### Alpaca Configuration

**Basic Setup:**
```json
{
  "brokers": {
    "alpaca": {
      "enabled": true,
      "api_key": "YOUR_ALPACA_API_KEY",
      "secret_key": "YOUR_ALPACA_SECRET",
      "paper": true,
      "base_url": "https://paper-api.alpaca.markets",
      "rate_limit": 200,
      "config": {
        "api_version": "v2",
        "use_numexpr": true,
        "raw_data": false
      }
    }
  }
}
```

**Advanced Configuration:**
```json
{
  "brokers": {
    "alpaca": {
      "enabled": true,
      "api_key": "${ALPACA_API_KEY}",
      "secret_key": "${ALPACA_API_SECRET}",
      "config": {
        "paper": "${ALPACA_PAPER:-true}",
        "base_url": "${ALPACA_BASE_URL:-https://paper-api.alpaca.markets}",
        "api_version": "v2",
        "rate_limit": 200,
        "timeout": 30,
        "retry_attempts": 3,
        "websocket": {
          "enabled": true,
          "feed": "iex",
          "raw_data": false
        },
        "market_data": {
          "feed": "iex",
          "currency": "USD"
        }
      }
    }
  }
}
```

### Interactive Brokers Configuration

**Setup for IBKR:**
```json
{
  "brokers": {
    "ibkr": {
      "enabled": true,
      "config": {
        "host": "127.0.0.1",
        "port": 7497,
        "client_id": 1,
        "paper": true,
        "max_concurrent_requests": 10,
        "request_timeout": 30,
        "connection_timeout": 60,
        "read_timeout": 30,
        "write_timeout": 30
      }
    }
  }
}
```

**IBKR Gateway Requirements:**
```bash
# IBKR TWS/Gateway must be running
# Enable API in TWS: Configure -> API -> Enable ActiveX and Socket Clients
# Set socket port (default 7497 for paper, 7496 for live)
```

### Trading 212 Configuration

```json
{
  "brokers": {
    "trading212": {
      "enabled": true,
      "api_key": "YOUR_TRADING212_API_KEY",
      "practice": true,
      "base_url": "https://practice.trading212.com",
      "rate_limit": 120,
      "config": {
        "version": "v1",
        "timeout": 30,
        "retry_attempts": 3
      }
    }
  }
}
```

### Broker Factory Configuration

**Multi-Broker Setup:**
```json
{
  "brokers": {
    "factory": {
      "default_broker": "binance",
      "routing_strategy": "best_execution",
      "failover_enabled": true,
      "health_check_interval": 60,
      "connection_timeout": 30,
      "max_retry_attempts": 3,
      "broker_priorities": {
        "execution_quality": ["binance", "alpaca", "ibkr"],
        "speed": ["alpaca", "binance", "ibkr"],
        "cost": ["alpaca", "trading212", "binance"]
      }
    }
  }
}
```

---

## ⚠️ Risk Management Configuration

### Basic Risk Settings

```json
{
  "risk": {
    "max_position_size": 10000,
    "max_daily_loss": 5000,
    "max_portfolio_risk": 0.20,
    "risk_per_trade": 0.02,
    "stop_loss_percentage": 0.05,
    "take_profit_percentage": 0.10,
    "max_daily_trades": 100
  }
}
```

### Circuit Breaker Configuration

```json
{
  "risk": {
    "circuit_breakers": {
      "enabled": true,
      "daily_loss_limit": 10000,
      "consecutive_loss_limit": 3,
      "drawdown_limit": 0.15,
      "position_concentration_limit": 0.30,
      "volatility_spike_threshold": 2.0,
      "correlation_threshold": 0.80,
      "reversal_sensitivity": 0.5
    },
    "auto_halt_conditions": [
      "daily_loss_limit_exceeded",
      "max_drawdown_reached",
      "consecutive_losses_limit",
      "system_error_critical"
    ],
    "manual_override_required": [
      "high_correlation_detected",
      "unusual_market_conditions"
    ]
  }
}
```

### Position Limits

```json
{
  "risk": {
    "position_limits": {
      "max_position_size": 10000,
      "max_portfolio_weight": 0.20,
      "max_sector_weight": 0.30,
      "min_liquidity_ratio": 0.10,
      "max_correlation_threshold": 0.70,
      "sector_classification": {
        "technology": ["AAPL", "GOOGL", "MSFT", "TSLA"],
        "financial": ["JPM", "BAC", "WFC", "C"],
        "healthcare": ["JNJ", "PFE", "UNH", "ABBV"]
      }
    },
    "dynamic_limits": {
      "enabled": true,
      "volatility_adjustment": true,
      "correlation_adjustment": true,
      "momentum_adjustment": false,
      "adjustment_intervals": {
        "volatility": 3600,
        "correlation": 1800,
        "momentum": 900
      }
    }
  }
}
```

### Risk Monitoring

```json
{
  "risk": {
    "monitoring": {
      "real_time_monitoring": true,
      "monitoring_interval": 60,
      "alert_thresholds": {
        "daily_loss_warning": 0.80,
        "drawdown_warning": 0.10,
        "correlation_warning": 0.70,
        "concentration_warning": 0.25
      },
      "notification_channels": {
        "email": {
          "enabled": true,
          "recipients": ["trader@example.com"]
        },
        "webhook": {
          "enabled": false,
          "url": "https://hooks.slack.com/..."
        },
        "sms": {
          "enabled": false,
          "phone_number": "+1234567890"
        }
      }
    }
  }
}
```

---

## 🧠 Strategy Configuration

### Strategy Library Configuration

```json
{
  "strategies": {
    "enabled_strategies": [
      "mean_reversion",
      "trend_following",
      "pairs_trading"
    ],
    "strategy_allocation": {
      "mean_reversion": 0.40,
      "trend_following": 0.30,
      "pairs_trading": 0.20,
      "arbitrage": 0.10
    },
    "performance_tracking": {
      "enabled": true,
      "rebalancing_frequency": "weekly",
      "min_trades_for_evaluation": 50,
      "performance_window": "1M"
    }
  }
}
```

### Individual Strategy Configuration

**Mean Reversion Strategy:**
```json
{
  "strategies": {
    "mean_reversion": {
      "enabled": true,
      "parameters": {
        "lookback_period": 20,
        "entry_threshold": 2.0,
        "exit_threshold": 0.5,
        "min_confidence": 0.75,
        "stop_loss": 0.03,
        "take_profit": 0.05
      },
      "risk_management": {
        "max_position_size": 5000,
        "max_daily_trades": 20,
        "correlation_limit": 0.70
      },
      "symbols": ["AAPL", "GOOGL", "MSFT", "AMZN"],
      "timeframes": ["1H", "4H"],
      "broker_preference": ["alpaca", "binance"]
    }
  }
}
```

**Pairs Trading Strategy:**
```json
{
  "strategies": {
    "pairs_trading": {
      "enabled": true,
      "parameters": {
        "lookback_period": 252,
        "entry_threshold": 2.0,
        "exit_threshold": 0.5,
        "min_correlation": 0.70,
        "half_life_threshold": 10,
        "hedge_ratio_method": "ordinary_least_squares"
      },
      "pair_selection": {
        "min_history_days": 60,
        "correlation_method": "pearson",
        "cointegration_test": "engle_granger",
        "max_pairs": 5
      },
      "risk_management": {
        "max_spread_bet": 0.10,
        "max_daily_pairs": 3
      }
    }
  }
}
```

**AI-Enhanced Strategy:**
```json
{
  "strategies": {
    "ai_enhanced_momentum": {
      "enabled": true,
      "ai_integration": {
        "enabled": true,
        "model_provider": "openai",
        "model": "gpt-4",
        "analysis_frequency": "hourly",
        "confidence_threshold": 0.80
      },
      "traditional_parameters": {
        "fast_ma_period": 10,
        "slow_ma_period": 30,
        "rsi_period": 14
      },
      "ai_enhancements": {
        "market_regime_detection": true,
        "sentiment_analysis": true,
        "news_impact_assessment": true,
        "volatility_adjustment": true
      }
    }
  }
}
```

### Strategy Portfolio Management

```json
{
  "strategies": {
    "portfolio_management": {
      "allocation_strategy": "risk_parity",
      "rebalancing": {
        "frequency": "weekly",
        "threshold": 0.05,
        "min_trades": 10
      },
      "performance_attribution": {
        "enabled": true,
        "factors": ["momentum", "mean_reversion", "volatility"],
        "benchmark": "SPY"
      },
      "risk_budgeting": {
        "enabled": true,
        "total_risk_budget": 0.15,
        "strategy_risk_limits": {
          "mean_reversion": 0.05,
          "trend_following": 0.04,
          "pairs_trading": 0.03,
          "arbitrage": 0.03
        }
      }
    }
  }
}
```

---

## 🤖 AI Configuration

### AI Model Configuration

```json
{
  "ai": {
    "trading_mode": "PAPER",
    "default_model_tier": "fast",
    "max_tokens_per_request": 2000,
    "request_timeout": 30,
    "models": {
      "reasoning_model": {
        "provider": "openai",
        "model": "gpt-4",
        "temperature": 0.1,
        "max_tokens": 2000
      },
      "fast_model": {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "temperature": 0.2,
        "max_tokens": 1000
      },
      "risk_model": {
        "provider": "anthropic",
        "model": "claude-3-sonnet",
        "temperature": 0.05,
        "max_tokens": 1500
      }
    }
  }
}
```

### API Keys Configuration

**Environment Variables:**
```bash
# OpenAI
OPENAI_API_KEY=your_openai_key
OPENAI_ORG_ID=your_org_id

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_key

# Local Models
LOCAL_MODELS_ENABLED=false
OLLAMA_BASE_URL=http://localhost:11434
LM_STUDIO_BASE_URL=http://localhost:1234
```

**Configuration File:**
```json
{
  "ai": {
    "local_models": {
      "enabled": false,
      "model_path": "./models/local",
      "preferred_backend": "ollama",
      "models": {
        "reasoning": {
          "name": "llama2:13b",
          "type": "chat",
          "context_length": 4096
        },
        "embedding": {
          "name": "nomic-embed-text",
          "type": "embedding",
          "dimensions": 768
        }
      }
    },
    "caching": {
      "enabled": true,
      "cache_ttl": 3600,
      "max_cache_size": "100MB"
    }
  }
}
```

### AI Features Configuration

```json
{
  "ai": {
    "features": {
      "market_analysis": {
        "enabled": true,
        "frequency": "hourly",
        "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA"],
        "analysis_depth": "comprehensive"
      },
      "strategy_selection": {
        "enabled": true,
        "auto_selection": true,
        "confidence_threshold": 0.75,
        "manual_override": true
      },
      "risk_assessment": {
        "enabled": true,
        "pre_trade_check": true,
        "continuous_monitoring": true,
        "real_time_adjustment": true
      },
      "news_analysis": {
        "enabled": true,
        "sources": ["reuters", "bloomberg", "marketwatch"],
        "sentiment_threshold": 0.6,
        "impact_decay_hours": 24
      },
      "pattern_recognition": {
        "enabled": true,
        "chart_patterns": true,
        "technical_patterns": true,
        "pattern_confidence": 0.80
      }
    }
  }
}
```

### Prompt Configuration

**Custom Prompt Templates:**
```json
{
  "ai": {
    "prompt_templates": {
      "market_analysis": "./ai/prompts/market_analysis.txt",
      "strategy_selection": "./ai/prompts/strategy_selection.txt",
      "risk_assessment": "./ai/prompts/risk_assessment.txt",
      "news_sentiment": "./ai/prompts/news_sentiment.txt"
    },
    "template_variables": {
      "current_time": "{{current_time}}",
      "portfolio_value": "{{portfolio_value}}",
      "risk_tolerance": "{{risk_tolerance}}",
      "market_regime": "{{market_regime}}"
    }
  }
}
```

---

## 🖥️ UI Configuration

### Terminal Interface Configuration

```json
{
  "ui": {
    "theme": "matrix",
    "colors": {
      "primary": "#00FF41",
      "secondary": "#008F11",
      "background": "#000000",
      "text": "#FFFFFF",
      "warning": "#FFA500",
      "error": "#FF0000",
      "success": "#00FF00",
      "info": "#0080FF"
    },
    "layout": {
      "panel_width": 40,
      "refresh_rate": 1000,
      "animation_enabled": true,
      "auto_hide_panels": false,
      "compact_mode": false
    }
  }
}
```

### Dashboard Configuration

```json
{
  "ui": {
    "dashboard": {
      "real_time_updates": true,
      "refresh_intervals": {
        "account": 5,
        "positions": 2,
        "orders": 1,
        "market_data": 1,
        "risk_metrics": 30
      },
      "panels": {
        "account_panel": {
          "visible": true,
          "position": "top-left",
          "auto_refresh": true
        },
        "positions_panel": {
          "visible": true,
          "position": "top-right",
          "sort_by": "unrealized_pnl",
          "show_percentages": true
        },
        "orders_panel": {
          "visible": true,
          "position": "bottom-left",
          "max_display": 10,
          "auto_refresh": true
        },
        "market_panel": {
          "visible": true,
          "position": "bottom-right",
          "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA"],
          "show_volume": true
        }
      }
    }
  }
}
```

### Keyboard Shortcuts

```json
{
  "ui": {
    "shortcuts": {
      "global": {
        "F1": "show_help",
        "F2": "toggle_fullscreen",
        "F3": "strategy_panel",
        "F4": "risk_panel",
        "F5": "portfolio_panel",
        "Tab": "cycle_focus",
        "Enter": "select_action",
        "Escape": "cancel_action",
        "Space": "pause_resume",
        "R": "refresh_data",
        "Q": "quit_application"
      },
      "order_entry": {
        "Ctrl+N": "new_order",
        "Ctrl+C": "cancel_order",
        "Ctrl+M": "modify_order"
      },
      "navigation": {
        "Ctrl+ArrowUp": "move_up",
        "Ctrl+ArrowDown": "move_down",
        "Ctrl+ArrowLeft": "move_left",
        "Ctrl+ArrowRight": "move_right"
      }
    }
  }
}
```

### Notifications Configuration

```json
{
  "ui": {
    "notifications": {
      "enabled": true,
      "sound": true,
      "sound_file": "./sounds/alert.wav",
      "volume": 0.5,
      "types": {
        "trade_execution": {
          "enabled": true,
          "sound": true,
          "popup": true
        },
        "risk_alerts": {
          "enabled": true,
          "sound": true,
          "popup": true,
          "email": false
        },
        "system_errors": {
          "enabled": true,
          "sound": true,
          "popup": true,
          "email": true
        },
        "strategy_signals": {
          "enabled": true,
          "sound": false,
          "popup": true
        }
      }
    }
  }
}
```

---

## ⚡ Performance Optimization

### Database Configuration

```json
{
  "performance": {
    "database": {
      "connection_pool_size": 20,
      "max_overflow": 30,
      "pool_timeout": 30,
      "pool_recycle": 3600,
      "query_timeout": 60,
      "enable_wal_mode": true,
      "synchronous_commit": false,
      "cache_size": "256MB",
      "temp_cache_size": "64MB"
    }
  }
}
```

### Caching Configuration

```json
{
  "performance": {
    "caching": {
      "enabled": true,
      "redis": {
        "host": "localhost",
        "port": 6379,
        "db": 0,
        "password": null,
        "ssl": false
      },
      "cache_tiers": {
        "memory": {
          "enabled": true,
          "max_size": "1GB",
          "ttl": 3600
        },
        "redis": {
          "enabled": true,
          "max_size": "4GB",
          "ttl": 86400
        },
        "disk": {
          "enabled": true,
          "path": "./cache/disk",
          "max_size": "10GB",
          "ttl": 604800
        }
      },
      "cache_strategies": {
        "market_data": {
          "tier": "memory",
          "ttl": 60,
          "max_size": "100MB"
        },
        "historical_data": {
          "tier": "redis",
          "ttl": 86400,
          "max_size": "1GB"
        },
        "strategy_signals": {
          "tier": "memory",
          "ttl": 300,
          "max_size": "50MB"
        }
      }
    }
  }
}
```

### Async Configuration

```json
{
  "performance": {
    "async": {
      "max_workers": 10,
      "worker_timeout": 60,
      "queue_size": 1000,
      "task_timeout": 300,
      "retry_attempts": 3,
      "backoff_factor": 2.0,
      "connection_pooling": {
        "http": {
          "max_connections": 100,
          "max_connections_per_host": 20,
          "keepalive_expiry": 5
        },
        "websocket": {
          "max_connections": 50,
          "ping_interval": 30,
          "pong_timeout": 10
        }
      }
    }
  }
}
```

### Market Data Optimization

```json
{
  "performance": {
    "market_data": {
      "compression": true,
      "batch_requests": true,
      "batch_size": 10,
      "request_coalescing": true,
      "priority_queues": {
        "realtime": {
          "max_size": 1000,
          "processing_threads": 4
        },
        "historical": {
          "max_size": 100,
          "processing_threads": 2
        }
      },
      "data_compression": {
        "enabled": true,
        "algorithm": "gzip",
        "level": 6
      },
      "stream_processing": {
        "enabled": true,
        "buffer_size": 1000,
        "batch_size": 100,
        "flush_interval": 1.0
      }
    }
  }
}
```

---

## 🔒 Security Configuration

### Encryption Settings

```json
{
  "security": {
    "encryption": {
      "enabled": true,
      "algorithm": "AES-256",
      "key_derivation": "PBKDF2",
      "iterations": 100000,
      "salt_length": 32,
      "key_length": 32
    },
    "api_keys": {
      "encryption_required": true,
      "rotation_interval_days": 90,
      "minimum_key_length": 32,
      "allowed_characters": "alphanumeric+special"
    }
  }
}
```

### Authentication Configuration

```json
{
  "security": {
    "authentication": {
      "api_keys_required": true,
      "session_timeout": 3600,
      "max_login_attempts": 5,
      "lockout_duration": 900,
      "password_policy": {
        "min_length": 12,
        "require_uppercase": true,
        "require_lowercase": true,
        "require_numbers": true,
        "require_special_chars": true,
        "prevent_common_passwords": true
      },
      "jwt": {
        "secret_key": "${JWT_SECRET_KEY}",
        "algorithm": "HS256",
        "access_token_expire_minutes": 30,
        "refresh_token_expire_days": 7
      }
    }
  }
}
```

### Network Security

```json
{
  "security": {
    "network": {
      "https_only": true,
      "certificate_path": "./certs/server.crt",
      "private_key_path": "./certs/server.key",
      "allowed_origins": [
        "https://localhost:3000",
        "https://yourdomain.com"
      ],
      "cors_enabled": true,
      "rate_limiting": {
        "enabled": true,
        "requests_per_minute": 1000,
        "burst_limit": 100
      },
      "ip_whitelist": {
        "enabled": false,
        "addresses": [
          "192.168.1.0/24",
          "10.0.0.0/8"
        ]
      }
    }
  }
}
```

### Audit Logging

```json
{
  "security": {
    "audit": {
      "enabled": true,
      "log_file": "logs/audit.log",
      "log_level": "INFO",
      "track_api_calls": true,
      "track_user_actions": true,
      "track_trade_executions": true,
      "retention_days": 2555,
      "encryption": true,
      "integrity_checking": true
    }
  }
}
```

---

## 🗄️ Database Configuration

### SQLite Configuration

```json
{
  "database": {
    "type": "sqlite",
    "url": "sqlite:///trading_orchestrator.db",
    "echo": false,
    "pool_size": 20,
    "max_overflow": 30,
    "pool_timeout": 30,
    "pool_recycle": 3600,
    "sqlite_settings": {
      "synchronous": "NORMAL",
      "journal_mode": "WAL",
      "cache_size": 10000,
      "temp_store": "memory",
      "mmap_size": 268435456
    }
  }
}
```

### PostgreSQL Configuration

```json
{
  "database": {
    "type": "postgresql",
    "url": "postgresql://user:password@localhost/trading_db",
    "pool_size": 20,
    "max_overflow": 30,
    "pool_timeout": 30,
    "pool_recycle": 3600,
    "echo": false,
    "postgresql_settings": {
      "connect_timeout": 10,
      "options": "-c search_path=trading_orchestrator",
      "application_name": "TradingOrchestrator",
      "sslmode": "prefer"
    },
    "connection_args": {
      "connect_timeout": 10,
      "command_timeout": 60
    }
  }
}
```

### Database Migrations

```json
{
  "database": {
    "migrations": {
      "enabled": true,
      "auto_migrate": true,
      "migration_path": "database/migrations",
      "backup_before_migration": true,
      "backup_path": "database/backups"
    }
  }
}
```

### Connection Pooling

```json
{
  "database": {
    "connection_pool": {
      "pool_size": 20,
      "max_overflow": 30,
      "pool_timeout": 30,
      "pool_recycle": 3600,
      "pool_pre_ping": true,
      "pool_reset_on_return": "commit"
    }
  }
}
```

---

## 📝 Logging Configuration

### Basic Logging

```json
{
  "logging": {
    "version": 1,
    "disable_existing_loggers": false,
    "formatters": {
      "default": {
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
      },
      "detailed": {
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
      },
      "json": {
        "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
        "format": "%(asctime)s %(name)s %(levelname)s %(filename)s %(lineno)d %(funcName)s %(message)s"
      }
    },
    "handlers": {
      "console": {
        "class": "logging.StreamHandler",
        "level": "INFO",
        "formatter": "default",
        "stream": "ext://sys.stdout"
      },
      "file": {
        "class": "logging.handlers.RotatingFileHandler",
        "level": "DEBUG",
        "formatter": "detailed",
        "filename": "logs/trading_orchestrator.log",
        "maxBytes": 10485760,
        "backupCount": 5
      },
      "error_file": {
        "class": "logging.handlers.RotatingFileHandler",
        "level": "ERROR",
        "formatter": "detailed",
        "filename": "logs/error.log",
        "maxBytes": 10485760,
        "backupCount": 5
      }
    },
    "root": {
      "level": "INFO",
      "handlers": ["console", "file"]
    },
    "loggers": {
      "trading_orchestrator": {
        "level": "DEBUG",
        "handlers": ["console", "file"],
        "propagate": false
      },
      "trading_orchestrator.orders": {
        "level": "INFO",
        "handlers": ["console", "file"],
        "propagate": false
      },
      "trading_orchestrator.risk": {
        "level": "WARNING",
        "handlers": ["console", "file", "error_file"],
        "propagate": false
      }
    }
  }
}
```

### Trading-Specific Logging

```json
{
  "logging": {
    "trading_logs": {
      "enabled": true,
      "trade_file": "logs/trades.log",
      "strategy_file": "logs/strategies.log",
      "risk_file": "logs/risk.log",
      "broker_file": "logs/brokers.log",
      "ai_file": "logs/ai.log",
      "trade_details": true,
      "strategy_performance": true,
      "risk_events": true,
      "format": "json",
      "rotation": {
        "max_bytes": 52428800,
        "backup_count": 10
      }
    }
  }
}
```

### Performance Logging

```json
{
  "logging": {
    "performance": {
      "enabled": true,
      "log_slow_queries": true,
      "slow_query_threshold": 1.0,
      "log_api_calls": true,
      "log_market_data": false,
      "metrics_file": "logs/metrics.log",
      "profiling": {
        "enabled": false,
        "profile_file": "logs/profile.prof",
        "sample_rate": 0.01
      }
    }
  }
}
```

---

## 🔧 Advanced Customization

### Plugin Architecture

**Custom Strategy Plugin:**
```json
{
  "plugins": {
    "strategies": {
      "enabled": true,
      "paths": ["./custom_strategies"],
      "auto_reload": true,
      "plugins": {
        "my_custom_strategy": {
          "enabled": true,
          "config_file": "./custom_strategies/my_strategy.json"
        }
      }
    }
  }
}
```

**Custom Broker Plugin:**
```json
{
  "plugins": {
    "brokers": {
      "enabled": true,
      "paths": ["./custom_brokers"],
      "plugins": {
        "my_custom_broker": {
          "enabled": true,
          "module": "my_custom_broker.broker",
          "class": "MyCustomBroker"
        }
      }
    }
  }
}
```

### Custom Data Sources

```json
{
  "data_sources": {
    "custom_feeds": {
      "enabled": true,
      "sources": {
        "my_news_feed": {
          "type": "websocket",
          "url": "wss://my-news-feed.com/stream",
          "format": "json",
          "rate_limit": 100,
          "authentication": {
            "type": "api_key",
            "header": "X-API-Key"
          }
        },
        "alternative_data": {
          "type": "rest_api",
          "base_url": "https://api.alternative-data.com",
          "rate_limit": 1000,
          "retry_attempts": 3
        }
      }
    }
  }
}
```

### Custom Indicators

```json
{
  "indicators": {
    "custom_indicators": {
      "enabled": true,
      "paths": ["./custom_indicators"],
      "indicators": {
        "my_momentum_indicator": {
          "module": "my_indicators.momentum",
          "class": "MyMomentumIndicator",
          "parameters": {
            "period": 14,
            "threshold": 0.5
          }
        }
      }
    }
  }
}
```

### Webhook Configuration

```json
{
  "webhooks": {
    "enabled": true,
    "endpoints": {
      "trade_notifications": {
        "url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
        "method": "POST",
        "headers": {
          "Content-Type": "application/json"
        },
        "events": ["trade_executed", "order_filled", "position_opened"],
        "filter": {
          "min_amount": 1000
        }
      },
      "risk_alerts": {
        "url": "https://discord.com/api/webhooks/YOUR/DISCORD/WEBHOOK",
        "method": "POST",
        "events": ["risk_limit_breach", "circuit_breaker_triggered"],
        "retry_attempts": 3
      }
    }
  }
}
```

### Custom Alerts

```json
{
  "alerts": {
    "custom_rules": [
      {
        "name": "Large Position Alert",
        "condition": "position_value > 50000",
        "action": "send_email",
        "parameters": {
          "to": "trader@example.com",
          "subject": "Large Position Alert"
        }
      },
      {
        "name": "Unusual Volatility",
        "condition": "volatility > portfolio_volatility * 2",
        "action": "send_webhook",
        "parameters": {
          "url": "https://hooks.slack.com/...",
          "channel": "#alerts"
        }
      }
    ]
  }
}
```

---

## 🚀 Configuration Management

### Configuration Validation

```python
# Validate configuration
python -c "
from config.validator import ConfigValidator
validator = ConfigValidator()
result = validator.validate_all()
print(f'Valid: {result.is_valid}')
print(f'Errors: {result.errors}')
print(f'Warnings: {result.warnings}')
"
```

### Configuration Templates

**Create Template:**
```bash
# Create configuration template
python config_manager.py --create-template production

# Create template for specific section
python config_manager.py --create-template brokers
```

### Environment-Specific Configs

**Development Configuration:**
```json
{
  "environment": "development",
  "debug": true,
  "trading_mode": "PAPER",
  "brokers": {
    "binance": {
      "testnet": true,
      "rate_limit": 12000
    }
  },
  "logging": {
    "level": "DEBUG",
    "console": true
  }
}
```

**Production Configuration:**
```json
{
  "environment": "production",
  "debug": false,
  "trading_mode": "LIVE",
  "brokers": {
    "binance": {
      "testnet": false,
      "rate_limit": 1200
    }
  },
  "security": {
    "https_only": true,
    "encryption": true
  },
  "logging": {
    "level": "WARNING",
    "file_rotation": true,
    "remote_logging": true
  }
}
```

### Configuration Profiles

```bash
# Switch configuration profile
python config_manager.py --profile development
python config_manager.py --profile paper_trading
python config_manager.py --profile live_trading

# Create new profile
python config_manager.py --create-profile my_profile
```

This comprehensive configuration guide provides all the tools and options needed to customize the Day Trading Orchestrator system for your specific trading needs. Remember to always test configuration changes in paper trading mode before applying them to live trading.
