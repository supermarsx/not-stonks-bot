# Broker API Integration Warnings

## Critical Notices for Unsupported Brokers

This document provides essential information about broker APIs that are NOT supported by the Day Trading Orchestrator System due to discontinuation, legal risks, or lack of official APIs.

---

## 1. XTB (xAPI) - DISCONTINUED

### Status: API SERVICE DISABLED

**Effective Date**: March 14, 2025

### Official Statement
XTB officially discontinued its API service as of March 14, 2025. All previously used hosts and endpoints (main socket and streaming WebSocket) are no longer accessible.

### Impact on Users
- **All API Access**: Completely unavailable
- **Historical Endpoints**: Non-functional
- **WebSocket Streams**: Discontinued
- **Legacy Integrations**: Must be migrated immediately

### What Was Available (Historical Reference)
The xAPI v2.5.0 provided:
- JSON-based protocol over socket and WebSocket
- Market data: ticks, candles, symbols, trading hours
- Order management: tradeTransaction, status polling
- Account information: balance, margin, P&L
- Rate limits: 200ms cadence, 1kB per command

### Migration Path

#### Recommended Alternative Brokers
1. **Alpaca Trading**
   - Developer-friendly REST and streaming APIs
   - US equities and ETFs focus
   - Generous rate limits (200 RPM)
   - Paper trading available
   - Deep historical data (5-6+ years)

2. **Interactive Brokers (IBKR)**
   - TWS API with broad product coverage
   - Professional-grade tooling
   - High rate limits (~3000 RPM)
   - Paper trading environment
   - Historical data to inception

3. **Oanda**
   - V20 REST API with robust FX focus
   - Strong research tools
   - High rate limits (~7200 RPM)
   - Demo account available
   - Historical data since 2005

4. **Tradier**
   - Clean brokerage API for US stocks and options
   - Straightforward onboarding
   - Transparent fees
   - Full historical data
   - ~120 RPM rate limit

### Migration Steps
1. **Freeze Trading**: Stop all XTB API-based trading immediately
2. **Select Alternative**: Choose replacement broker from recommendations
3. **Credential Setup**: Create API keys in new broker's sandbox/demo
4. **Streaming Mapping**: Replace XTB push semantics with new broker's model
5. **Order Workflow**: Translate xAPI order types to target broker's format
6. **Testing**: Complete end-to-end testing in paper/demo environment
7. **Go Live**: Execute staged rollout with monitoring

### Technical Reference
- xAPI Protocol Documentation: `research/xtb/xtb_api_analysis.md`
- Feature comparison matrix in migration section
- Rate limit and endpoint mapping guide

---

## 2. DEGIRO - NO OFFICIAL API

### Status: UNOFFICIAL ACCESS ONLY (HIGH RISK)

### Official Position
DEGIRO does not offer an official, public-facing API. The helpdesk explicitly states:
> "Connections to external applications are not supported, and the platform must be accessed via its web and mobile interfaces."

### Legal and Contractual Risks

#### Terms Violation
DEGIRO's Customer Agreement and Terms and Conditions do NOT provide for programmatic access outside official channels. Unauthorized use of access paths, programs, or interfaces can result in:
- Account suspension or termination
- Loss of access to funds during investigation
- Potential liability for terms violations

#### Available Unofficial Tools
Several community libraries exist (Python, TypeScript, Dart) but:
- **Not Endorsed**: By DEGIRO
- **No Support**: Community-maintained with frequent breakages
- **Breakage Risk**: App updates break clients without notice
- **Rate Limits**: Unknown and unenforceable
- **Authentication Fragility**: Header changes can cause failures

### Why We Don't Support DEGIRO

1. **Legal Exposure**: Violates Terms and Conditions
2. **No SLA**: Unofficial clients have no reliability guarantees
3. **Maintenance Burden**: Frequent breakages require constant upkeep
4. **Account Risk**: Potential for account termination
5. **Compliance**: Cannot meet production risk standards

### If You Must Use Unofficial Access

**WARNING**: Proceed at your own risk. This is NOT recommended.

If organizational requirements force unofficial access:
- **Read-Only Operations Only**: No trading automation
- **Conservative Pacing**: Avoid aggressive polling
- **Backend Deployment**: Never browser-based (CORS blocks)
- **Fallback Plans**: Manual processes ready for breakage
- **Legal Review**: Document risk acceptance
- **Monitor**: Active surveillance for upstream changes

### Recommended Alternatives
Use officially supported brokers:
- Interactive Brokers (IBKR) - Professional grade
- Alpaca - Developer-friendly
- Trading 212 - Official API (beta, limited features)
- Binance - Crypto focus with official API

### Technical Reference
- Research: `research/degiro/degiro_unofficial_api_analysis.md`
- Unofficial libraries: TypeScript (icastillejogomez), Python (pytr, DegiroAPI)
- Terms documents referenced in research

---

## 3. Trade Republic - NO OFFICIAL API

### Status: UNOFFICIAL ACCESS ONLY (CONTRACTUAL VIOLATION)

### Official Position
Trade Republic does NOT offer an official public API. All programmatic access is through reverse-engineered, unofficial clients.

### Legal and Contractual Exposure

#### Customer Agreement Prohibition
Trade Republic's Customer Agreement explicitly states:
> "Services can only be used via this Application... as well as other access channels provided by Trade Republic."

> "Any use of the features and services provided by Trade Republic through access paths, programs and/or other interfaces not provided by Trade Republic outside of the Application" is prohibited.

#### Consequences of Violation
- **Extraordinary Termination**: Account can be terminated immediately
- **No Recourse**: Violation of agreement removes user protections
- **Operational Risk**: Sudden loss of trading access
- **Reputational Risk**: For organizations using unofficial access

### Available Unofficial Tools
Community libraries exist (Python: pytr, TradeRepublicApi) but:
- **Contractual Violation**: Use breaches Customer Agreement
- **Device Pairing**: Only one device at a time; resets log out mobile app
- **No Rate Limits**: Undocumented and subject to change
- **Frequent Breakages**: App updates break clients regularly
- **No Support**: Community-maintained with gaps

### Why We Don't Support Trade Republic

1. **Explicit Contract Violation**: Customer Agreement prohibits use
2. **Extraordinary Termination Risk**: Account closure without appeal
3. **Operational Fragility**: Breakages are common and unpredictable
4. **Device Constraints**: Single-device pairing complicates automation
5. **Legal Liability**: Organizations face compliance exposure

### Absolutely NO Trading Automation

**CRITICAL**: Do NOT use unofficial Trade Republic access for:
- Automated order placement
- Algorithmic trading strategies
- Production trading systems
- Any mission-critical workflows

### Acceptable Use (If Any)
Only consider for:
- Read-only portfolio analytics (non-critical)
- Manual document retrieval
- Internal reconciliation (with manual fallback)

**Even then**: Requires explicit legal review and risk acceptance.

### Recommended Alternatives
Use officially supported brokers with proper APIs:
- Interactive Brokers (IBKR) - Professional API
- Alpaca - Commission-free with official API
- Binance - Crypto with REST + WebSocket
- Trading 212 - Official beta API

### Technical Reference
- Research: `research/trade_republic/trade_republic_unofficial_api_analysis.md`
- Unofficial libraries: pytr (most complete), TradeRepublicApi
- Customer Agreement and legal documents referenced in research

---

## Summary Decision Matrix

| Broker | Status | API Quality | Legal Risk | Trading Risk | Recommendation |
|--------|--------|-------------|------------|--------------|----------------|
| XTB | Discontinued | N/A | N/A | N/A | **MIGRATE IMMEDIATELY** |
| DEGIRO | No Official API | Unofficial Only | HIGH | HIGH | **DO NOT USE** |
| Trade Republic | No Official API | Unofficial Only | VERY HIGH | VERY HIGH | **DO NOT USE** |
| Binance | Official API | Excellent | LOW | LOW | **RECOMMENDED** |
| IBKR | Official API | Professional | LOW | LOW | **RECOMMENDED** |
| Alpaca | Official API | Developer-Friendly | LOW | LOW | **RECOMMENDED** |
| Trading 212 | Official Beta API | Limited (Beta) | LOW | MEDIUM | **USE WITH CAUTION** |

---

## Resources

### Migration Guides
- XTB to Alpaca: `docs/migrations/xtb_to_alpaca.md`
- XTB to IBKR: `docs/migrations/xtb_to_ibkr.md`
- General migration checklist: `docs/migrations/CHECKLIST.md`

### Legal Documentation
- DEGIRO Terms: `research/degiro/degiro_unofficial_api_analysis.md`
- Trade Republic Customer Agreement: `research/trade_republic/trade_republic_unofficial_api_analysis.md`
- XTB Discontinuation Notice: `research/xtb/xtb_api_analysis.md`

### Contact
For questions about broker migration or alternative recommendations, consult:
- Official broker documentation
- Community forums (for technical questions only, not legal advice)
- Legal counsel (for contractual and compliance questions)

---

**Last Updated**: 2025-11-05  
**Document Version**: 1.0  
**Maintainer**: MiniMax Agent
