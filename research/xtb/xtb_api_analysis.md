# XTB API (xAPI) for Trading System Integration: Capabilities, Discontinuation, and Viable Alternatives

## Executive Summary

XTB’s xAPI (xStation5 API) provided a JSON-based protocol over socket and WebSocket connections for market data, account information, and order management. Official documentation describes a two-connection model: a main socket for request–reply commands and a streaming WebSocket for real-time subscriptions, with SSL required for live trading. All officially documented endpoints and mechanics—ranging from symbol metadata and tick streams to trade transactions and profit/margin calculations—are now historical only. As of March 14, 2025, XTB disabled its API service endpoints, removing access to the previously advertised REST-like hosts and the WebSocket gateways; XTB’s official help center directs users to the web platform and mobile application instead.[^1][^2][^3]

This discontinuation forces teams to pivot. The recommended path is to select and integrate a broker API that meets the organization’s asset-class needs, latency profile, data requirements, and developer experience expectations. Where a legacy integration must be decommissioned, the pragmatic approach is to freeze trading, preserve transactional continuity via the broker’s web or desktop platforms, and execute a staged migration to an alternative API—typically with parallel running and validation, replay protection, and monitoring of rate-limit and streaming semantics in the new environment.[^1][^2][^3]

To orient decision-makers quickly, the table below summarizes current feasibility and next steps.

Table 1: Status overview—XTB xAPI features and what to do now

| Capability area | What xAPI historically provided | Access status (2025-03-14 onward) | Immediate action |
|---|---|---|---|
| Transport and protocol | JSON over clean socket (main) and WebSocket (streaming), SSL for live | Disabled | Treat all historical protocol details as reference-only |
| Authentication | Login via userId/password; streamingSessionId for subscriptions | Disabled | N/A; move credentials management to new broker API |
| Market data | Tick prices (on-change), 1-minute candles, symbol metadata, trading hours, calendar, news | Disabled | Map requirements to alternative broker data feeds (push/pull, depth) |
| Order management | tradeTransaction, status polling and streaming of trade state, trades and history | Disabled | Re-implement OMS to target selected broker’s order API semantics |
| Account information | Balance, margin level, profit streaming | Disabled | Recreate account snapshot and streaming for new broker |
| Rate limits & reliability | 200 ms command cadence; ≤1 kB per command; 50k candles/request; various error codes | N/A | Engineer to new broker’s limits; build backoff, metrics, and SLOs |
| Trading hours & time zones | CET/CEST candle times; trading hours per symbol | N/A | Update calendaring and session logic for target markets |

In short, XTB’s xAPI is no longer available; treat all integration work as legacy maintenance or migration planning. The rest of this report details what the API was, how it behaved, and which alternatives merit consideration.[^1][^2][^3]

## What xAPI Was: Protocol, Transport, and Architecture

The xAPI protocol, documented at version 2.5.0, standardized interactions using JSON over two distinct transport channels. The main connection provided synchronous request–reply semantics, while a dedicated streaming connection delivered real-time updates. Data was encoded in UTF-8, with time expressed as Unix milliseconds. Each command formed a compact JSON object with a “header” containing a “type” and a “data” payload. Servers enforced strict formatting and cadence rules, closing connections that violated them.[^4]

### Protocol and Data Model

The protocol’s structure was intentionally simple. Every message included a header (with at least a type field) and a data object specific to that type. A customTag field, if supplied in a command, was echoed back in the response to facilitate correlation. Numeric values used a period as the decimal separator. Time was uniformly represented as milliseconds since the Unix epoch, enabling consistent timestamping across functions such as getServerTime, candle timestamps (ctm), and trading hours boundaries.[^4]

### Connection Types and Hosts (Historical)

The architecture comprised:

- A main clean socket connection for request–reply commands (SSL for live trading).
- A streaming WebSocket connection for push updates (SSL across all servers).

Historical documentation and mirror sites referenced demo and real servers on distinct ports and hosts, with WebSocket endpoints identified for both streaming and synchronous operations. The discontinuation notice supersedes these details; these hosts and ports are no longer accessible.[^4][^5][^2]

Table 2: Historical connection endpoints and ports (for reference only)

| Environment | Main socket (historical) | Streaming WebSocket (historical) | Notes |
|---|---|---|---|
| Demo | Dedicated main port and host; see official docs/mirror | Dedicated streaming port and host; see official docs/mirror | Disabled since 2025-03-14 |
| Real | Dedicated main port and host; see official docs/mirror | Dedicated streaming port and host; see official docs/mirror | Disabled since 2025-03-14 |

The operational takeaway is that a dual-socket design was central to xAPI’s model: the main socket coordinated command execution while the streaming socket minimized polling and delivered event-driven updates.[^4]

## Authentication and Session Management

xAPI authenticated sessions via a Login command carrying userId and password. Upon success, the response returned a streamSessionId, which was required for subsequent streaming subscriptions. A logout command terminated the session. In practice, open-source wrappers reflected this by requiring accountId, password, and type (“real” or “demo”), then establishing both sockets to operate the full API surface.[^4][^7][^8]

Table 3: Authentication parameters and session artifacts

| Artifact | Description | Usage |
|---|---|---|
| userId | Account identifier | Login request |
| password | Account password | Login request |
| streamSessionId | Token returned on successful login | Required to subscribe on streaming connection |
| logout | Command to end session | Clean disconnect |

From a security perspective, teams must assume credential lifecycle management is no longer relevant for xAPI. For future brokers, treat API credentials and tokens as secrets, rotate regularly, and ensure least-privilege configurations.[^4]

## Market Data Feeds

xAPI’s market data suite was broad. Symbol metadata exposed trading constraints and pricing properties; quotes were delivered as on-change ticks; charts provided minute-based candles; and ancillary feeds included trading hours, news, and economic calendars. Wrappers in Python and Node.js demonstrated these patterns in practice: symbol queries, tick subscriptions, and historical candle retrieval with optional windows.[^4][^7][^8]

Table 4: Market data capabilities matrix

| Capability | Request–reply | Streaming | Key fields/notes |
|---|---|---|---|
| Symbol metadata | getAllSymbols, getSymbol | N/A | SYMBOL_RECORD: ask, bid, precision, lotMin/lotMax, leverage, stopsLevel, swapType, etc. |
| Tick prices | getTickPrices | stream_getTickPrices | TICK_RECORD: ask, bid, askVolume, bidVolume, timestamp; on-change delivery |
| Candles | getChartLastRequest, getChartRangeRequest | stream_getCandles | Period codes M1–MN1; 1-minute streaming candle cadence |
| Trading hours | getTradingHours | N/A | Quotes vs trading times per symbol |
| News | getNews | stream_getNews | NEWS_TOPIC_RECORD with title/body and metadata |
| Calendar | getCalendar | N/A | Economic events with impact levels |

### Symbol and Quote Characteristics

Symbol records were rich, capturing pricing, volume steps, margin rules, and operational constraints. Precision and pipsPrecision informed rounding behavior; stopsLevel and trailingEnabled shaped order placement; swap parameters determined overnight financing effects. Tick records provided quote, volume on both sides, and a timestamp suitable for event-driven processing.[^4]

Table 5: Representative SYMBOL_RECORD fields and usage

| Field | Meaning | Usage |
|---|---|---|
| ask, bid | Current prices | Price discovery, slippage modeling |
| precision, pipsPrecision | Decimal places | Formatting and numeric tolerance |
| lotMin, lotMax, lotStep | Volume constraints | Order sizing validation |
| leverage | Effective leverage | Margin calculations, risk limits |
| stopsLevel | Minimum distance for SL/TP | Order placement and compliance |
| swapType, swapLong, swapShort | Financing parameters | Overnight carry modeling |
| trailingEnabled | Trailing stop availability | Order type support decisions |

### Historical Data

Historical candles were available across standard periods. Streaming getCandles produced one-minute candles continuously; request–reply endpoints filled ranges between timestamps. The protocol capped data volumes per request and advised cadence rules to protect server stability.[^4]

Table 6: Historical data availability windows

| Period code | Typical availability window |
|---|---|
| M1 | Under 1 month |
| M30 | 1 to 7 months |
| H4 | 7 to 13 months |
| D1 | About 13 months and earlier |

Max candles per request were limited, with specific error codes triggered if thresholds were exceeded. Teams had to implement chunking, backoff, and stateful replay logic to avoid hitting EX009 (“data limit potentially exceeded”) and similar protections.[^4]

## Order Management and Trading Workflows

xAPI unified trading commands under tradeTransaction. A successful response indicated that the request was received, not that the order was accepted; clients had to poll tradeTransactionStatus or subscribe to streaming trade updates to determine the final state. The API supported opening, closing, modifying, and deleting orders, with rich transaction types (e.g., BUY, SELL, BUY_LIMIT, BUY_STOP). Wrappers implemented helper methods that mapped to these codes, reflecting practical usage patterns.[^4][^8]

Table 7: Transaction types and operations

| Field | Values | Meaning |
|---|---|---|
| cmd | 0: BUY, 1: SELL, 2: BUY_LIMIT, 3: SELL_LIMIT, 4: BUY_STOP, 5: SELL_STOP, 6: BALANCE, 7: CREDIT | Trade direction or instrument operation |
| type | 0: OPEN, 1: PENDING, 2: CLOSE, 3: MODIFY, 4: DELETE | Action to perform |
| price | Floating number | Limit or stop price where applicable |
| sl, tp | Floating numbers | Stop loss and take profit levels |
| volume | Floating number | Trade size in lots |
| order | Integer | Existing order for close/modify/delete |

Table 8: Transaction status values

| Status code | Meaning |
|---|---|
| 0 | ERROR |
| 1 | PENDING |
| 3 | ACCEPTED |
| 4 | REJECTED |

The streaming layer added visibility: stream_getTrades and stream_getTradeStatus pushed updates for open positions and transaction state changes, while stream_getProfits delivered real-time P&L. Together, these formed a robust basis for event-driven OMS integration.[^4][^7]

## Account Information and Calculations

Account snapshots and calculations were first-class functions. getCurrentUserData and getMarginLevel exposed balances, equity, margin, and margin level; streaming variants pushed updates on balance changes. Profit, margin, and commission could be computed via dedicated commands, allowing pre-trade validation and risk checks.[^4]

Table 9: Account indicator fields

| Field | Meaning |
|---|---|
| balance | Account balance |
| credit | Credit amount |
| currency | Base currency |
| equity | Equity |
| margin | Margin used |
| margin_free | Free margin |
| margin_level | Margin level percentage |

Table 10: Calculation commands and outputs

| Command | Inputs | Outputs | Use case |
|---|---|---|---|
| getProfitCalculation | openPrice, closePrice, cmd, symbol, volume | profit | Pre-trade P&L estimation |
| getMarginTrade | symbol, volume | margin | Sizing and margin checks |
| getCommissionDef | symbol, volume | commission, rateOfExchange | Fee modeling and FX impact |

These utilities simplified risk workflows by consolidating calculation logic server-side, reducing client-side errors and discrepancies.[^4]

## Real-time Streaming Semantics

Streaming subscriptions were central to real-time operation. The connection emitted keep-alive signals periodically, and clients were expected to manage subscriptions and back-pressure. Rate and payload controls were explicit: commands should be spaced by 200 milliseconds, each command capped at roughly 1 kilobyte, and connections limited per client address. Violations led to connection drops or errors.[^4]

Table 11: Streaming commands and cadence expectations

| Command | Purpose | Cadence/behavior |
|---|---|---|
| stream_getTickPrices | Real-time tick updates | On-change delivery; minArrivalTime and maxLevel configurable |
| stream_getCandles | One-minute candle stream | New candle each minute |
| stream_getTrades | Trade updates | Push on open/close/modify events |
| stream_getTradeStatus | Transaction state | Push status changes for submitted orders |
| stream_getProfits | Real-time P&L | Push updates as positions change |
| stream_getNews | News headlines | Push in real time |
| stream_getKeepAlive | Keep-alive messages | Periodic emissions to maintain session |

These semantics encouraged event-driven architectures that were resilient to temporary disconnections and that honored server constraints.[^4]

## Rate Limits and Reliability

xAPI enforced several reliability constraints:

- Command cadence: send requests at ≥200 ms intervals; repeated violations could cause disconnection.
- Payload size: ≤1 kB per command invocation.
- Connection count: maximum of 50 simultaneous connections from the same client address; exceeding this triggered EX008.
- Historical data cap: up to 50,000 candles per request; EX009 signaled potential exceeded limits.
- Timeouts and input validation: commands with invalid parameters returned error responses without closing the connection; overloaded systems returned specific internal error codes.[^4]

Table 12: Rate limits and connection constraints

| Constraint | Value | Error/behavior |
|---|---|---|
| Command cadence | ≥200 ms between sends | Connection dropped after repeated violations |
| Payload size | ≤1 kB per command | Invalid format yields error; connection may persist |
| Simultaneous connections | 50 per client IP | EX008: connection limit reached |
| Candles per request | Up to 50,000 | EX009: data limit potentially exceeded |

Table 13: Representative error codes and handling guidance

| Code | Description | Client action |
|---|---|---|
| BE005 / EX007 | Invalid credentials or disabled login | Retry after interval; verify credentials |
| BE014 / BE016–BE017 | Frequent trade requests / too many trade requests | Backoff; implement request batching |
| EX008 | Connection limit reached | Reduce concurrent connections; re-establish selectively |
| EX009 | Data limit exceeded | Chunk requests; apply pagination |
| BE099 / EX000–EX006 | Internal errors and overload | Exponential backoff; alerting |

Robust clients treated these constraints as SLO boundaries and built telemetry and backoff strategies around them.[^4]

## Discontinuation of XTB API Service (2025-03-14)

XTB officially discontinued API access as of March 14, 2025. The announcement cited disabling access via the previously used hosts and encouraged users to utilize the web platform and mobile application. This change affects both the main socket and streaming WebSocket endpoints, and renders any remaining references to hosts or ports non-functional.[^1][^2][^3][^6]

Table 14: Discontinuation timeline and impacted components

| Date | Event | Impacted components |
|---|---|---|
| 2025-03-14 | API service disabled | Main socket, streaming WebSocket, legacy hosts |

Migration pathways now focus on alternative brokers and APIs rather than internal workarounds.[^1][^2][^3]

## Alternative Broker APIs: Landscape and Shortlist

Selection criteria should prioritize asset coverage (e.g., equities vs FX vs crypto), real-time streaming quality, historical data depth, demo availability, rate limits, SDKs, documentation quality, and fee impacts on throughput strategies. Independent 2025 reviews highlight multiple viable options with strong API programs.[^10][^11]

- Alpaca Trading: developer-friendly REST and streaming APIs, deep historical data, generous rate limits, demo availability, and multi-language SDKs; especially strong for US equities and ETFs.[^10]
- Interactive Brokers (IBKR): TWS API with broad product coverage, professional-grade tooling, high rate limits, and extensive historical depth.[^10]
- Oanda: V20 REST API with robust FX focus, strong research tools, and high rate limits.[^10]
- Tradier: clean brokerage API for US stocks and options; straightforward onboarding and transparent fees.[^10]
- Additional options from broader surveys include FOREX.com (REST + Lightstreamer streaming), Dukascopy (JForex and FIX), and Gemini (REST, WebSocket, and FIX), among others.[^11]

Table 15: Alternative comparison (indicative highlights)

| Broker | Primary strengths | Streaming | Historical data | Rate limits | Demo |
|---|---|---|---|---|---|
| Alpaca | Developer-centric API; US equities focus | Push feed | 5–6+ years | ~200 RPM | Yes |
| IBKR | Broad markets; pro tools | Push feed | To inception (daily) | ~3000 RPM | Yes |
| Oanda | FX focus; research quality | Push feed | Since 2005 | ~7200 RPM | Yes |
| Tradier | Stocks/options; low fees | Push feed | Full history (daily) | ~120 RPM | Yes |

These choices should be evaluated against the organization’s asset mix, latency budget, and deployment footprint.[^10][^11]

## Migration Strategy from XTB xAPI to Alternative Brokers

A disciplined migration reduces operational risk and shortens time to stability.

1. Broker selection: Map current instruments and workflows to a broker with equivalent or superior APIs. Confirm streaming behavior, historical coverage, and rate limits.
2. Credential and environment setup: Create API keys or equivalent credentials in a sandbox/demo environment. Establish secrets management and least-privilege access.
3. Streaming equivalence: Replace xAPI’s push semantics with the target’s streaming model. Define subscription sets, replay/backfill logic, and disconnect recovery.
4. Order workflow mapping: Translate cmd/type semantics into the target order model; replace tradeTransactionStatus polling with event-driven status streams.
5. Rate-limit adaptation: Instrument throughput and build backoff logic suited to the new broker’s limits.
6. Testing: Conduct end-to-end regression testing in a demo environment, including failure scenarios and replay behavior.

Table 16: Feature mapping from xAPI to target broker API (illustrative)

| xAPI feature | Typical target broker equivalent | Notes |
|---|---|---|
| stream_getTickPrices | WebSocket tick stream | Confirm on-change vs snapshot cadence |
| stream_getCandles (M1) | Bar stream or historical bars API | Validate frequency and backfill limits |
| tradeTransaction | Place/modify order endpoints | Confirm idempotency and status semantics |
| stream_getTradeStatus | Order status event stream | Align with replace/cancel behaviors |
| getMarginTrade | Risk/margin calculation endpoint | Reconcile margin models |
| getCurrentUserData | Account snapshot endpoints | Align currency and balance fields |

Table 17: Migration milestones and validation

| Milestone | Outcome |
|---|---|
| Demo integration complete | Streaming and basic order flows verified |
| Regression test passed | Functional parity across core workflows |
| Performance test under limits | No throttling; stable latency and throughput |
| Go-live checklist signed | Fallbacks, monitoring, and incident runbooks in place |

Given the discontinuation, the immediate operational priority is stabilizing trading outside API access while the migration proceeds.[^1]

## Risks, Compliance, and Operational Considerations

Post-discontinuation, the most salient risks relate to API availability, credentials, rate-limit behavior, and data discrepancies. The control strategy is pragmatic: maintain alternative execution routes via web or desktop platforms, adopt resilient streaming patterns, implement secrets hygiene, and align rate-limit handling with the target broker’s documented constraints. Historical discrepancies between tick and candle data underline the importance of validating upstream data semantics during integration testing.[^4][^12]

Table 18: Risk register and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| API unavailability | High | Severe | Use web/desktop execution; migrate to supported API |
| Credential misuse | Medium | High | Secrets vaulting; least privilege; rotation |
| Rate-limit throttling | Medium | Medium | Backoff and batching; throughput tuning |
| Streaming disconnects | Medium | Medium | Auto-reconnect; replay/backfill |
| Data discrepancies | Medium | Medium | Cross-validate ticks vs candles; contract tests |
| OMS state drift | Medium | High | Event sourcing; audit logs; reconciliation jobs |

## Appendix: Field and Command Reference (Historical)

This appendix consolidates commonly used fields and commands for legacy maintenance and migration reference.

Table 19: SYMBOL_RECORD fields and descriptions (selected)

| Field | Description |
|---|---|
| ask, bid | Quote prices |
| precision, pipsPrecision | Decimal precision |
| lotMin, lotMax, lotStep | Volume constraints |
| leverage | Account leverage |
| stopsLevel | Minimum SL/TP distance |
| trailingEnabled | Trailing stop support |
| swapType, swapLong, swapShort | Swap parameters |

Table 20: TICK_RECORD and STREAMING_CANDLE_RECORD fields

| Record | Fields (selected) |
|---|---|
| TICK_RECORD | ask, bid, askVolume, bidVolume, high, low, timestamp |
| STREAMING_CANDLE_RECORD | open, high, low, close, ctm, ctmString, vol |

Table 21: Calculation commands and outputs (summary)

| Command | Inputs | Outputs |
|---|---|---|
| getProfitCalculation | openPrice, closePrice, cmd, symbol, volume | profit |
| getMarginTrade | symbol, volume | margin |
| getCommissionDef | symbol, volume | commission, rateOfExchange |

Table 22: Error code ranges and handling guidance (summary)

| Range | Notes |
|---|---|
| BE0xx | Business logic and validation errors (e.g., invalid price, insufficient funds) |
| EX0xx | System errors and limits (e.g., overload, access, data limits) |

These references derive from the xAPI protocol documentation and should be used solely for context during migration to other APIs.[^4]

## Information Gaps and Assumptions

- XTB’s official rationale and long-term替代 (replacement) roadmap beyond the discontinuation notice are not provided.
- Complete and current lists of supported assets per region under the discontinued API are unavailable.
- Definitive REST semantics are not documented; the protocol historically used JSON over socket/WebSocket.
- Up-to-date rate-limit figures for 2025 from XTB do not exist; the last documentation specified cadence and payload constraints.
- Regional regulatory notes for future broker APIs require fresh analysis per jurisdiction.[^1][^2][^4]

## References

[^1]: Do you offer API? - Help center | XTB (EN). https://www.xtb.com/en/help-center/our-platforms/do-you-offer-api  
[^2]: XTB API 2.5 Notice (developers.xstore.pro). http://developers.xstore.pro/  
[^3]: Trading hours, press releases, rollovers, and more. Page 29 | XTB. https://www.xtb.com/cy/company-news?page=29  
[^4]: XAPI PROTOCOL DOCUMENTATION | Financial Technology Provider. https://xopenhub.pro/api/xapi-protocol-documentation/  
[^5]: API Documentation — xapi-node (xStation5 Trading API for NodeJS/JS). https://peterszombati.github.io/xapi-node/  
[^6]: timirey/xapi-php: PHP wrapper for XTB xAPI (DEPRECATED). https://github.com/timirey/xapi-php  
[^7]: A Python API for the XTB global trading platform. https://github.com/caiomborges/Python-XTB-API  
[^8]: API Reference — XtbClient v0.1.1 (HexDocs). https://hexdocs.pm/xtb_client_ex/api-reference.html  
[^9]: xapi documentation ( Stav GitHub Pages). https://stav.github.io/xapi/  
[^10]: Best Brokers for Algorithmic Trading in 2025 - BrokerChooser. https://brokerchooser.com/best-brokers/best-brokers-for-algo-trading  
[^11]: Best Brokers With API Access 2025 | Top API Trading Platforms. https://www.daytrading.com/apis  
[^12]: XTB API Discrepancies in candles vs tick prices - Stack Overflow. https://stackoverflow.com/questions/71896214/xtb-api-discrepancies-in-candles-vs-tick-prices