# Binance Spot API for Trading System Integration: Endpoints, Websockets, Limits, Authentication, Order Types, and Restrictions

## Executive Summary and Integration Objectives

Binance’s Spot Application Programming Interface (API) provides a comprehensive suite for integrating trading applications with the exchange’s market data, order management, and account information. At its core, the API offers REST endpoints for market data and trading operations, and WebSocket streams for real-time market data and user account updates. Responses follow a consistent JavaScript Object Notation (JSON) format with millisecond precision on timestamps, and the platform implements rate limits and order constraints that directly shape system architecture and operational resilience. These characteristics define the integration surface that trading systems must address, including compliance with authentication and time discipline, careful management of rate budgets, and robust error handling across synchronous and asynchronous interfaces.[^2][^12]

This report’s objectives are to outline the capabilities and constraints of the Binance Spot API, with practical guidance for trading system integration. Specifically, it covers market data endpoints, order management operations, account and permissions queries, WebSocket streams, rate limits and error responses, authentication methods, supported order types and conditional parameters, trading restrictions and filters, best practices in time synchronization, error handling, and strategies for order book integrity and rate-limit-aware architectures. The report is intended for software engineers, quantitative developers, solution architects, and technical product managers responsible for building and maintaining robust integrations.

## API Foundations: Structure, Base Endpoints, and Data Conventions

The Binance Spot API adopts straightforward conventions that reduce ambiguity and facilitate predictable integration. By default, responses are returned in JSON with timestamps in milliseconds. When microsecond-level precision is required, clients can set the `X-MBX-TIME-UNIT` header to `MICROSECOND` (or microsecond). REST requests have a server-side timeout of approximately ten seconds, and responses generally return data in chronological order unless specified otherwise. Time filtering parameters such as `startTime` and `endTime` follow clear semantics, and the system provides explicit guidance on error states, including the TIMEOUT error code for cases where the backend matching engine’s execution status may be unknown.[^2]

Base endpoints exist for REST and WebSocket communication, with distinct hosts for public market data. The public market data base endpoint is designated to handle high volumes of non-private queries, and developers are advised to leverage this endpoint for market data-only use cases. While the developer center redacts exact URLs for some endpoints, it highlights that multiple REST base endpoints exist, and that the last four endpoints provide improved performance with a tradeoff in stability. This performance-stability note encourages operational designs that are resilient to endpoint selection and failover, especially under load.[^2]

To illustrate the endpoint strategy and its practical implications, the following table enumerates base endpoints and their roles.

Table 1: Base endpoints overview

| Interface Type        | Role                                           | Notes                                                                                         |
|-----------------------|------------------------------------------------|-----------------------------------------------------------------------------------------------|
| REST base endpoints   | General trading, market data, account queries  | Multiple base endpoints exist; the last four offer better performance but less stability.     |
| Public market data    | Public-only market data access                 | Dedicated base endpoint for market data-only queries; preferred for non-private data.         |
| WebSocket streams     | Real-time market data and user data            | Streams include raw and combined channels; connection validity typically lasts 24 hours.      |

These conventions and endpoint choices frame the broader integration architecture. In practice, separating public market data traffic from account-specific operations improves performance isolation and simplifies compliance with rate limits.[^2]

### Time and Data Ordering Semantics

Timestamp handling is central to trading applications. By default, all time fields are measured in milliseconds. Developers can opt into microsecond precision by sending the `X-MBX-TIME-UNIT` header with the value `MICROSECOND` or microsecond. Data ordering behaviors depend on whether `startTime`, `endTime`, or both are provided. Without time filters, responses return the most recent items up to the applicable limit. With `startTime` only, the API returns the oldest items from the specified time onward. With `endTime` only, it returns the most recent items up to the end time. When both `startTime` and `endTime` are present, behavior follows the `startTime` semantics but constrained to not exceed the `endTime`.[^2]

Table 2: Time parameter behavior matrix

| Parameters Provided          | Returned Data Semantics                                                                            |
|-----------------------------|-----------------------------------------------------------------------------------------------------|
| None                        | Most recent items up to the limit.                                                                  |
| startTime only              | Oldest items from `startTime` up to the limit.                                                      |
| endTime only                | Most recent items up to `endTime` and the limit.                                                    |
| startTime and endTime       | Behaves like `startTime`, but does not exceed `endTime`.                                            |
| Timestamp precision control | Default milliseconds; opt into microseconds via `X-MBX-TIME-UNIT: MICROSECOND` header.             |

These semantics support deterministic pagination and backfill strategies, allowing systems to coordinate incremental data ingestion with known reset intervals for rate limits.[^2]

## Market Data Endpoints (REST): Capabilities and Weights

Binance’s REST market data endpoints cover depth, trades, aggregated trades, candlesticks (klines), UI-optimized klines, average price, 24-hour ticker statistics, trading day statistics, symbol price tickers, best book tickers, and rolling window statistics. Each endpoint has specific parameters, weights, and data sources, and several impose practical constraints such as maximum `limit` values or symbol filtering by trading status.[^3]

Before detailing individual endpoints, it is useful to summarize weights for high-impact endpoints, particularly those that scale with the number of symbols requested or the depth requested.

Table 3: Market data endpoint weights summary (selected endpoints)

| Endpoint                               | Weight / Cost Model                                                                                                           |
|----------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| GET /api/v3/depth (order book)         | 5 (limit 1–100); 25 (101–500); 50 (501–1000); 250 (1001–5000).                                                                |
| GET /api/v3/ticker/24hr                | 2 (symbol provided, single); 80 (symbol omitted, returns all); scales by number of symbols: 2 (≤20), 40 (21–100), 80 (≥101). |
| GET /api/v3/ticker/price               | 2 (single symbol); 4 (symbol omitted or symbols list).                                                                        |
| GET /api/v3/ticker/bookTicker          | 2 (single symbol); 4 (symbol omitted or symbols list).                                                                        |
| GET /api/v3/ticker/tradingDay          | 4 per requested symbol, capped at 200 when `symbols` count > 50.                                                              |
| GET /api/v3/ticker (rolling window)    | 4 per requested symbol; cap at 200 when `symbols` count > 50.                                                                 |

The order book endpoint’s weight scales with the depth requested, incentivizing conservative `limit` choices. Ticker endpoints differentiate costs between single-symbol queries and bulk (all symbols) queries, requiring careful batching and caching strategies for large symbol universes.[^3]

Table 4: Order book depth limit vs weight mapping

| Depth Limit (levels per side) | Request Weight |
|-------------------------------|----------------|
| 1–100                         | 5              |
| 101–500                       | 25             |
| 501–1000                      | 50             |
| 1001–5000                     | 250            |

These weights directly affect the REQUEST_WEIGHT rate budget and must be considered in throttling design, especially for systems that periodically refresh deep order books across many symbols.[^3]

### Order Book, Trades, and Kline Endpoints

The depth endpoint (GET /api/v3/depth) returns the order book for a symbol. Weights scale with the requested `limit` and are designed to encourage judicious use of very deep snapshots. The `symbolStatus` filter allows clients to request depth only for symbols with specific trading statuses (e.g., TRADING, HALT, BREAK); mismatches return errors for single-symbol requests or simply exclude non-matching symbols for multi-symbol contexts.[^3]

Recent trades (GET /api/v3/trades), historical trades (GET /api/v3/historicalTrades), and compressed aggregate trades (GET /api/v3/aggTrades) provide trade flows with varying data sources and parameterization, including optional filtering by `fromId`, `startTime`, and `endTime`. Candlestick endpoints (GET /api/v3/klines and GET /api/v3/uiKlines) return open-high-low-close-volume data and related metadata, with support for kline intervals ranging from seconds to months, and a `timeZone` parameter that governs the timezone used to interpret intervals (note that `startTime` and `endTime` remain in UTC regardless of `timeZone`).[^3]

Table 5: Kline intervals and time zone handling

| Category       | Values                                                                                  | Notes                                                                                                              |
|----------------|------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| Intervals      | 1s; 1m, 3m, 5m, 15m, 30m; 1h, 2h, 4h, 6h, 8h, 12h; 1d, 3d; 1w; 1M                        | `timeZone` governs interval interpretation; `startTime` and `endTime` are always UTC.                              |
| timeZone       | Hours and minutes (e.g., -1:00, 05:45), only hours (e.g., 0, 8, 4), accepted [-12:00 to +14:00] | Ensures consistent alignment of open/close windows when users operate in non-UTC locales.                         |

The UI klines endpoint provides chart-optimized fields for presentation, while the standard klines endpoint is suited for analytics and strategy computation. In both cases, `limit`, `startTime`, and `endTime` support pagination, and `timeZone` enables consistent charting across regional settings.[^3]

### Ticker and Book Ticker Endpoints

The 24-hour ticker endpoint (GET /api/v3/ticker/24hr) offers rolling window statistics for single or multiple symbols, and also a `type` parameter that toggles between FULL and MINI responses. Single-symbol requests with a mismatched `symbolStatus` return errors; multi-symbol or all-symbol requests exclude non-matching symbols gracefully. The trading day ticker (GET /api/v3/ticker/tradingDay) computes statistics aligned to trading days, with similar toggles for FULL/MINI and a weight of four per requested symbol, capped at 200 when many symbols are provided. Symbol price tickers (GET /api/v3/ticker/price) and best book tickers (GET /api/v3/ticker/bookTicker) provide lightweight views suitable for frequent polling, with low weights for single-symbol queries. Rolling window tickers (GET /api/v3/ticker) compute statistics for windows up to 59,999 milliseconds beyond the requested size; `openTime` always starts on a minute, while `closeTime` reflects the current request time, which can slightly widen the effective window.[^3]

Table 6: Ticker endpoint parameters and weight comparison

| Endpoint                     | Key Parameters                          | Weight / Cost Model                                                         |
|-----------------------------|-----------------------------------------|-----------------------------------------------------------------------------|
| /api/v3/ticker/24hr         | `symbol` or `symbols`; `type`; `symbolStatus` | 2 (single symbol); 80 (all symbols); scales 2, 40, 80 by symbol count ranges. |
| /api/v3/ticker/tradingDay   | `symbol`/`symbols`; `timeZone`; `type`  | 4 per symbol; capped at 200 when `symbols` > 50.                             |
| /api/v3/ticker/price        | `symbol`/`symbols`; `symbolStatus`      | 2 (single symbol); 4 (symbol omitted or list).                              |
| /api/v3/ticker/bookTicker   | `symbol`/`symbols`; `symbolStatus`      | 2 (single symbol); 4 (symbol omitted or list).                              |
| /api/v3/ticker (rolling)    | `symbol`/`symbols`; `windowSize`; `type` | 4 per symbol; cap at 200 when `symbols` > 50; effective window may be wider. |

These endpoints’ cost profiles encourage targeted queries and caching for frequently used symbols, and caution against blanket polling of all symbols, which can exhaust REQUEST_WEIGHT budgets quickly.[^3]

## WebSocket Streams: Real-time Market Data and Connection Management

Binance offers WebSocket streams for real-time market data, including aggregate trades, trades, klines, tickers (individual and all-market), rolling window statistics, book tickers, partial depth, and diff depth updates. The platform supports raw streams and combined streams, with a maximum of 1024 streams per connection. The server issues a `ping` frame approximately every 20 seconds, and clients must respond with a `pong` within one minute to avoid disconnection. The connection’s validity typically spans 24 hours, and per-IP connection limits enforce fairness and protect server resources. While the developer center provides redacted base URLs, the documented streams, payloads, and management messages provide a clear operational model for subscribing, unsubscribing, listing subscriptions, and configuring stream properties.[^4]

Table 7: WebSocket stream types and update speeds

| Stream Type                                  | Stream Name Pattern             | Update Speed                     | Notes                                                                                     |
|----------------------------------------------|----------------------------------|----------------------------------|-------------------------------------------------------------------------------------------|
| Aggregate Trade                               | `@aggTrade`                      | Real-time                        | Aggregated trade events.                                                                  |
| Trade                                         | `@trade`                         | Real-time                        | Individual trade events.                                                                  |
| Kline (UTC)                                   | `@kline_<interval>`              | ~1000ms for 1s; 2000ms others    | Intervals: 1s, 1m–30m, 1h–12h, 1d–3d, 1w, 1M.                                             |
| Kline (UTC+8)                                 | `@kline_<interval>@+08:00`       | ~1000ms for 1s; 2000ms others    | Intervals open/close in UTC+8; event timestamps remain UTC.                               |
| Mini Ticker (individual)                      | `@miniTicker`                    | 1000ms                           | 24hr rolling window, compact payload.                                                     |
| Mini Ticker (all-market)                      | `!miniTicker@arr`                | 1000ms                           | Only changed tickers included.                                                            |
| Ticker (individual)                           | `@ticker`                        | 1000ms                           | 24hr rolling window, full payload.                                                        |
| Ticker (all-market)                           | `!ticker@arr`                    | 1000ms                           | Only changed tickers included.                                                            |
| Rolling Window Statistics (individual)        | `@ticker_<windowSize>`           | 1000ms                           | Window sizes: 1h, 4h, 1d; effective window may be wider.                                  |
| Rolling Window Statistics (all-market)        | `!ticker_<windowSize>@arr`       | 1000ms                           | Only changed tickers included.                                                            |
| Book Ticker (individual)                      | `@bookTicker`                    | Real-time                        | Best bid/ask updates.                                                                     |
| Average Price                                 | `@avgPrice`                      | 1000ms                           | Windowed average price.                                                                   |
| Partial Book Depth                            | `@depth` or `@depth@100ms`       | 1000ms or 100ms                  | Levels: 5, 10, 20.                                                                        |
| Diff Depth                                    | `@depth` or `@depth@100ms`       | 1000ms or 100ms                  | Incremental updates to order book.                                                        |

Table 8: WebSocket connection and message constraints

| Constraint                               | Value / Behavior                                                                 |
|------------------------------------------|-----------------------------------------------------------------------------------|
| Incoming messages per second             | 5                                                                                 |
| Max streams per connection               | 1024                                                                              |
| Max connections per IP                   | 300 per 5 minutes                                                                 |
| Ping frequency                           | Server `ping` every ~20 seconds                                                  |
| Pong response requirement                | Client must `pong` within 1 minute                                               |
| Connection validity                      | Typically valid for 24 hours                                                     |
| Time precision                           | Milliseconds by default; microseconds via `timeUnit=MICROSECOND` parameter        |
| Subscription management                  | SUBSCRIBE, UNSUBSCRIBE, LIST_SUBSCRIPTIONS, SET_PROPERTY, GET_PROPERTY           |

Subscription messages follow a simple JSON pattern, with `method`, `params` (list of stream names), and `id` for correlation. The server acknowledges subscription lifecycle events, and clients can introspect current subscriptions or toggle combined payloads via property management messages.[^4] For derivatives market streams and further streaming conventions, see the derivatives WebSocket documentation, keeping in mind that this report focuses on Spot.[^16]

### Local Order Book Management

Maintaining a consistent local order book requires a disciplined synchronization workflow. The official process specifies buffering diff events, fetching a depth snapshot, verifying update IDs, discarding stale events, and applying incremental updates in order. Systems must handle gaps and mismatches by restarting the synchronization process. The depth snapshot is limited to 5000 levels per side; updates to levels beyond this bound do not show quantity changes unless those levels are later updated within the 5000-level window.[^4]

Table 9: Diff depth synchronization procedure

| Step                                              | Purpose                                                                 | Error Handling                                               |
|---------------------------------------------------|-------------------------------------------------------------------------|--------------------------------------------------------------|
| Buffer diff events and note first event’s `U`     | Prepare to align snapshot and incremental updates.                      | If snapshot `lastUpdateId` < buffered event `U`, re-fetch.   |
| Fetch depth snapshot via REST                     | Establish baseline order book state.                                    | If mismatch persists, restart process.                       |
| Discard buffered events with `u` ≤ snapshot ID    | Ensure only subsequent updates are applied.                              | First buffered event’s `U` must fall within [U; u] range.    |
| Apply buffered and subsequent diff updates        | Reconcile local order book with incremental changes.                     | Ignore if `u` < local update ID; restart if `U` > local ID.  |
| Remove levels with zero quantity; update quantities | Maintain accurate depth levels.                                         | Set order book update ID to event `u` after applying.        |

This procedure ensures deterministic alignment with the exchange’s canonical order book and reduces race conditions in fast-moving markets. Engineers should instrument gap detection and automatic restart logic to mitigate packet reordering or transient connectivity issues.[^4]

## Account Information and Permissions

Account information is exposed via REST endpoints that provide balances, trading permissions, commission rates, filters, order status, order lists, trade history, and features related to Self-Trade Prevention (STP) and Smart Order Routing (SOR). Permissions flags such as `canTrade`, `canWithdraw`, and `canDeposit` indicate the account’s capabilities and are essential for pre-trade validation. Additional fields summarize commissions and account-level constraints, and symbol-specific filters inform acceptable order parameters and asset-level limits.[^10]

Table 10: Account endpoints and weights

| Endpoint                         | Weight | Key Parameters                                        | Notes                                                                                                  |
|----------------------------------|--------|-------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| GET /api/v3/account              | 20     | `omitZeroBalances`, `recvWindow`, `timestamp`         | Balances, permissions, commissionRates; data source memory ⇒ database.                                 |
| GET /api/v3/account/commission   | 20     | `symbol`                                              | Symbol-specific commission rates including discounts (e.g., BNB).                                      |
| GET /api/v3/myFilters            | 40     | `symbol`, `recvWindow`, `timestamp`                   | Exchange, symbol, and asset filters (e.g., MAX_NUM_ORDERS, MAX_ASSET).                                 |
| GET /api/v3/order                | 4      | `symbol`, `orderId` or `origClientOrderId`, timestamp | Order status; either `orderId` or `origClientOrderId` required.                                        |
| GET /api/v3/openOrders           | 6 (single symbol); 80 (none) | `symbol`, `recvWindow`, `timestamp`             | Open orders; caution when called without symbol (high weight).                                         |
| GET /api/v3/allOrders            | 20     | `symbol`, `orderId`, `startTime`, `endTime`, `limit`  | Time window ≤ 24 hours.                                                                                |
| GET /api/v3/order/amendments     | 4      | `symbol`, `orderId`, `fromExecutionId`, `limit`       | Amendment history per order.                                                                            |
| GET /api/v3/orderList            | 4      | `orderListId` or `origClientOrderId`, `timestamp`     | Specific order list (e.g., OCO).                                                                       |
| GET /api/v3/allOrderList         | 20     | `fromId`, `startTime`, `endTime`, `limit`             | Time window ≤ 24 hours.                                                                                |
| GET /api/v3/openOrderList        | 6      | `recvWindow`, `timestamp`                             | Open order lists.                                                                                       |
| GET /api/v3/myTrades             | 20 (without orderId); 5 (with orderId) | `symbol`, `orderId`, `startTime`, `endTime` | Time window ≤ 24 hours; trade history for account and symbol.                                          |
| GET /api/v3/myPreventedMatches   | 2 (invalid symbol or by preventedMatchId); 20 (by orderId) | `symbol`, `preventedMatchId`, `orderId` | List of orders expired due to STP modes.                                                                |
| GET /api/v3/myAllocations        | 20     | `symbol`, `startTime`, `endTime`, `fromAllocationId`, `limit` | Allocations resulting from SOR placements; time window ≤ 24 hours.                                     |
| GET /api/v3/rateLimit/order      | 40     | `recvWindow`, `timestamp`                             | Unfilled order count across intervals (SECOND, DAY) vs limits.                                         |

These endpoints give a comprehensive view of account constraints and operational state. For example, `/api/v3/account` returns `canTrade`, `canDeposit`, `canWithdraw`, commission details, and `permissions`, while `/api/v3/myFilters` exposes exchange, symbol, and asset filters that materially affect order placement and volume constraints. The unfilled order count endpoint is crucial for monitoring ORDERS rate budgets.[^10]

### Permissions and Filters

Permissions flags are the primary gate for trading operations. `canTrade` indicates whether orders can be submitted, `canDeposit` and `canWithdraw` denote asset movement capabilities, and `permissions` enumerate account-level scopes (e.g., SPOT). The `/api/v3/myFilters` endpoint returns account and symbol-level filters, such as maximum numbers of orders and order lists, and asset-specific limits under `MAX_ASSET`. Engineers should integrate filter validation into pre-trade checks to avoid rejected orders and unnecessary rate consumption.[^10]

## Order Management (Trading Endpoints): Placement, Modification, Query, and Lists

Order management relies on signed REST endpoints for placing, testing, modifying, and canceling orders; querying order status; and handling order lists. The trading endpoints enforce signature validation and the mandatory inclusion of `timestamp` and `recvWindow` parameters for signed requests. Additionally, the documentation clarifies response modes (ACK, RESULT, FULL) and provides an explicit list of conditional fields for order responses, signaling when certain parameters may or may not be present based on order type and state.[^11]

Table 11: Trading endpoint matrix (overview)

| Operation Type       | Endpoint / Method                                | Security Type           | Response Modes                     | Notes                                                                                           |
|----------------------|---------------------------------------------------|-------------------------|------------------------------------|-------------------------------------------------------------------------------------------------|
| Test new order       | POST /api/v3/order/test                          | TRADE (SIGNED)          | ACK-like (validation only)         | Validates signature and parameters; does not send to matching engine.                           |
| Create order         | POST /api/v3/order                               | TRADE (SIGNED)          | ACK, RESULT, FULL                   | Behavior depends on `newOrderRespType`; MARKET and LIMIT default to FULL; others default to ACK.|
| Query order status   | GET /api/v3/order                                | TRADE (SIGNED)          | N/A                                 | Requires `symbol` and either `orderId` or `origClientOrderId`.                                  |
| Cancel order         | DELETE /api/v3/order                             | TRADE (SIGNED)          | ACK-like                             | Cancel by `orderId` or `origClientOrderId`.                                                    |
| Cancel by symbol     | DELETE /api3/openOrders                          | TRADE (SIGNED)          | ACK-like                             | Cancel all open orders for a symbol.                                                           |
| Open orders          | GET /api/v3/openOrders                           | TRADE (SIGNED)          | N/A                                 | High weight when symbol omitted; use judiciously.                                              |
| All orders           | GET /api/v3/allOrders                            | TRADE (SIGNED)          | N/A                                 | Time window ≤ 24 hours; use pagination for large histories.                                     |
| Order amendments     | GET /api/v3/order/amendments                     | TRADE (SIGNED)          | N/A                                 | Amendment history; supports filtering by execution ID.                                         |
| Order list (OCO)     | GET /api/v3/orderList; GET /api/v3/allOrderList; GET /api/v3/openOrderList | TRADE (SIGNED) | N/A | OCO/OTOCO order list lifecycles and statuses.                                                  |

The trading endpoints’ security type indicates signed endpoints requiring valid API keys and signatures; `recvWindow` defines request validity window, and `timestamp` ensures time-bounded execution. The test endpoint is recommended for dry-run validations in production systems, particularly to prevent invalid signatures or filter violations from consuming order rate budgets.[^11]

### Order Lists (OCO and OTOCO)

Order lists support complex strategies like One Cancels the Other (OCO) and One Triggers the Other (OTOCO). These constructs combine contingent orders—such as limit and stop-limit legs—so that execution or triggering of one leg cancels the other. List status endpoints report lifecycle states, and the system enforces a 24-hour window constraint for time-filtered queries, necessitating pagination for extended historical retrieval.[^10]

## Rate Limits and Error Handling

Binance implements a multi-layered rate limit model that combines REQUEST_WEIGHT per-IP tracking, ORDERS per-account tracking for successfully placed orders, and RAW_REQUESTS enumerated in `exchangeInfo`. Clients receive rate limit usage via response headers and can opt to include rate limit information in WebSocket responses via `returnRateLimits` parameters. Ban durations scale from minutes to days for repeated violations, and the platform uses standard HTTP error codes to signal rate-limit breaches (429) and IP bans (418). Engineers must design backoff strategies that honor `retryAfter` guidance and coordinate with interval reset behaviors (e.g., fixed-window resets aligned to seconds or UTC midnight).[^1][^7][^6]

Table 12: Rate limit types and tracking

| Rate Limit Type   | Tracked At          | How to Query                                                |
|-------------------|---------------------|-------------------------------------------------------------|
| REQUEST_WEIGHT    | Per IP              | Response headers; `exchangeInfo` rateLimits; WebSocket `rateLimits`. |
| ORDERS            | Per account         | WebSocket account request `account.rateLimits.orders`; REST `GET api/v3/rateLimit/order`.     |
| RAW_REQUESTS      | Per IP/account      | `exchangeInfo` rateLimits enumeration.                      |

Table 13: Common HTTP error codes and recommended handling

| Code | Meaning                                     | Retry Guidance                                                                                   |
|------|---------------------------------------------|---------------------------------------------------------------------------------------------------|
| 429  | Too Many Requests                           | Backoff until `retryAfter`; review weight usage; consider batching or caching to reduce load.     |
| 418  | IP Banned                                   | Stop traffic until `retryAfter`; implement cooldown and reduce request cadence; audit clients.   |
| -1007| TIMEOUT                                     | Do not assume matching engine outcome; query user data stream or status endpoints to reconcile.  |

Intervals used for rate limits include SECOND, MINUTE, HOUR, and DAY. Some intervals are nested (e.g., 10-second windows within a minute), and exhausting a shorter interval requires waiting for it to expire, even if longer intervals show capacity. Reset times are fixed (e.g., DAY resets at 00:00 UTC; 10-second windows reset at 00, 10, 20, ... seconds). WS API connections are constrained per IP (e.g., 300 connections per 5 minutes), and connecting to the WS API itself consumes REQUEST_WEIGHT (e.g., 2 weight per connection), which should be included in budgeting. Client designs should avoid tight reconnect loops and consolidate streams to minimize weight overhead.[^1][^7]

### Order Rate Limits (Spot)

Spot order rate limits track successfully placed orders per account across intervals. As of the 2023 update, limits increased to 100 orders per 10 seconds and 200,000 orders per 24 hours. Importantly, orders that are filled—partially or fully—do not count against the order rate limit, incentivizing strategies that achieve immediate execution. Systems should monitor the unfilled order count endpoint to avoid exhausting ORDERS budgets and should treat filled orders as rate-neutral for order counting.[^6]

Table 14: Spot order rate limits by interval and monitoring

| Interval     | Limit      | Monitoring Endpoint                   | Notes                                                      |
|--------------|------------|----------------------------------------|------------------------------------------------------------|
| 10 seconds   | 100 orders | GET api/v3/rateLimit/order             | Count excludes filled orders (partial or full).            |
| 24 hours     | 200,000    | GET api/v3/rateLimit/order             | Per account; shared across API keys.                       |

## Authentication and Security

Binance supports multiple algorithms for signing requests: Hash-based Message Authentication Code (HMAC), RSA, and Ed25519. For REST, the canonical signing sequence constructs the signature from the query string (for GET) or request body (for POST), using the API secret as the key. The signature is included as the `signature` parameter alongside `timestamp` and `recvWindow`, which control time validity windows. WebSocket API session authentication messages follow specific parameter ordering and security requirements; the platform provides a WS API request security page that details secure methods and session logon semantics. The public documentation clarifies supported key types and provides guidance on header and payload construction.[^2][^15][^8][^21]

Table 15: Authentication methods comparison

| Algorithm | Typical Use Case                               | Key Management Implications                                       | Operational Considerations                                         |
|-----------|-------------------------------------------------|----------------------------------------------------------------------|--------------------------------------------------------------------|
| HMAC      | Broad support across REST and WS APIs           | Secret key used to sign requests; must be kept confidential         | Deterministic signature parameter ordering; common in trading bots. |
| RSA       | Environments preferring public-key cryptography | Keypair generation and secure storage required                      | Heavier compute cost; ensure robust signature encoding.            |
| Ed25519   | Modern elliptic curve signature scheme          | Similar to RSA; supports compact signatures                         | Validate library support; ensure consistent encoding.              |

For HMAC, Binance employs HMAC-SHA256 with recommended key lengths and best practices for secure storage and rotation. Developers should generate keys via cryptographically secure random number generators, avoid hardcoding secrets, enforce role-based access controls, and implement auditing. The signature itself is not case-sensitive, and `totalParams` refers to the full parameter string used as the HMAC input.[^8] Engineers should standardize signature construction in shared libraries to avoid subtle inconsistencies across services.

### HMAC, RSA, and Ed25519 in Practice

While all three algorithms are supported, HMAC-SHA256 remains the most common for general trading integration due to its simplicity and performance. RSA and Ed25519 may be preferred in environments with specific cryptographic policies or existing key management infrastructure. Regardless of algorithm, `timestamp` and `recvWindow` are mandatory parameters for signed endpoints, and clients should synchronize time sources carefully to avoid signature expiry or validation errors.[^2]

## Supported Order Types and Parameters

Binance supports a range of Spot order types: Market, Limit, Stop-Limit, Stop-Market, OCO (One Cancels the Other), Trailing Stop, OTO (One Triggers the Other), and OTOCO (One Triggers a One Cancels the Other). These order types map to conditional behaviors that trigger market or limit orders when specified criteria are met. Time in force (TIF) options such as Good-Til-Canceled (GTC), Immediate-or-Cancel (IOC), and Fill-or-Kill (FOK) govern execution finality, and iceberg quantities allow splitting larger orders into controlled slices. Self-Trade Prevention modes influence order behavior in the presence of potential self-matching, and Smart Order Routing (SOR) allocations provide transparency into execution paths. The trading endpoints clarify conditional fields in responses, and the support FAQ offers accessible descriptions of each order type’s behavior.[^11][^9]

Table 16: Order types vs trigger conditions and execution behavior

| Order Type      | Trigger Condition                           | Execution Behavior                                   | Typical Use Case                                  |
|-----------------|----------------------------------------------|------------------------------------------------------|---------------------------------------------------|
| Market          | None                                         | Fills immediately at best market price               | Immediate entry/exit.                             |
| Limit           | Price threshold                              | Executes at specified limit or better                | Price-controlled entry/exit.                      |
| Stop-Limit      | Stop price reached                           | Places a limit order at set price                    | Risk management; breakout strategies.             |
| Stop-Market     | Stop price reached                           | Places a market order for immediate fill             | Urgent止损 with market execution.                 |
| OCO             | Either leg executes or triggers              | Cancels the counterpart leg                          | Bracketed strategies (entry + exit).              |
| Trailing Stop   | Price moves against by a defined percent     | Follows price; exits when reverse movement occurs    | Trend following with dynamic stops.               |
| OTO             | Primary order condition met                  | Triggers secondary order                             | Two-stage execution without cancellation link.    |
| OTOCO           | Primary order condition met                  | Triggers an OCO order list                           | Advanced bracketed strategies.                    |

Table 17: TIF and conditional parameters mapping

| Parameter     | Meaning                                       | Applies To                  | Notes                                             |
|---------------|-----------------------------------------------|-----------------------------|---------------------------------------------------|
| timeInForce   | GTC, IOC, FOK                                  | Limit orders and variants   | Defines order lifetime and fill constraints.      |
| icebergQty    | Slice quantity for iceberg orders              | Limit and related variants  | Hides total size while executing in slices.       |
| stopPrice     | Trigger price for conditional orders           | Stop/OCO/OTOCO variants     | Triggers placement of market or limit orders.     |
| trailingDelta | Percent or amount trailing from market price   | Trailing stop orders        | Follows price; updates stop upon favorable moves. |

Order responses may include conditional fields depending on the order type and state, and systems should tolerate absent fields gracefully. For example, `stopPrice` and `icebergQty` appear only when applicable, while `timeInForce` is relevant for limit orders. The trading endpoints documentation provides detailed enumeration of conditional fields, and engineers should align parsing logic to these specifications.[^11]

## Trading Restrictions and Filters

Trading restrictions are defined via exchange and symbol filters, including limits on the number of orders and order lists, and asset-specific caps under `MAX_ASSET`. The `/api/v3/myFilters` endpoint enumerates filters relevant to the account and symbol, which materially affect pre-trade validations. Trading permissions are exposed via `canTrade`, `canDeposit`, and `canWithdraw`, and must be confirmed before submitting orders. The platform also enforces symbol status filters, and REST responses may include errors such as `-1220 SYMBOL_DOES_NOT_MATCH_STATUS` when requesting data for symbols that do not match the specified trading status. These restrictions and filters should be integrated into the trading system’s validation pipeline.[^10][^3]

Table 18: Filter types and examples

| Filter Type              | Example Field(s)                               | Purpose                                           |
|--------------------------|-------------------------------------------------|---------------------------------------------------|
| EXCHANGE_MAX_NUM_ORDERS  | `maxNumOrders`                                 | Caps total number of active orders.               |
| MAX_NUM_ORDER_LISTS      | `maxNumOrderLists`                             | Caps active order lists (e.g., OCO).              |
| MAX_ASSET                | `asset`, `limit`                               | Caps holdings or balances for specific assets.    |

Table 19: Permissions flags and implications

| Flag       | Description                       | Pre-trade Checks                                 |
|------------|-----------------------------------|---------------------------------------------------|
| canTrade   | Orders can be placed              | Must be true to submit trading requests.          |
| canDeposit | Funds can be deposited            | Not required for orders but affects funding flow. |
| canWithdraw| Funds can be withdrawn            | Operational capability; not directly order-related.|

These filters and permissions constrain order placement at the API level. Integrating validation steps against `myFilters` reduces unnecessary order submissions and helps prevent rate-limit incidents caused by rejected orders.[^10]

## Integration Architecture and Best Practices

A robust integration architecture benefits from a clear separation of concerns between market data ingestion and trading operations, disciplined time synchronization, and rate-limit-aware request scheduling.

- Market data only: Use the designated public base endpoint for non-private queries. This isolates public load from account operations and leverages infrastructure optimized for high-volume market data.[^2]
- Real-time data: Prefer WebSocket streams for trades, tickers, and order book diffs. Use REST for snapshotting (e.g., depth) and for filling gaps after disconnections or out-of-order events.[^4]
- Time synchronization: Enforce strict clock discipline with `timestamp` and `recvWindow`. When needed, opt into microsecond precision via `X-MBX-TIME-UNIT`, particularly for high-frequency trading or precise audit trails.[^2]
- Error handling: Treat `-1007 TIMEOUT` as an indeterminate outcome. Do not assume a timeout implies failure or success in the matching engine. Query user data streams or order status endpoints to reconcile states and avoid double submission.[^2]
- Rate limits: Monitor response headers for REQUEST_WEIGHT usage and query account-level ORDERS usage via `/api/v3/rateLimit/order`. Design backoff strategies that honor `retryAfter` and align to reset intervals. Consider REQUEST_WEIGHT budgets when choosing endpoint parameters (e.g., depth `limit` and bulk ticker requests).[^1][^7]
- Order book integrity: Follow the official diff depth synchronization workflow and instrument gap detection. On mismatches between `U`/`u` and snapshot IDs, restart the synchronization rather than attempting ad hoc reconciliation.[^4]

These practices minimize operational risk and improve performance under volatile market conditions. Where the developer center highlights performance-stability tradeoffs among base endpoints, client designs should incorporate endpoint selection strategies and failover plans.[^2][^1][^7]

## Appendices: Configuration and Environment Setup

Binance provides test environments and developer resources that streamline integration and operational readiness.

- Spot Testnet: Use the Spot Testnet for sandbox testing of API key creation, signing workflows, and order lifecycles.[^22]
- Demo environment: Binance Demo offers simulated trading for broader product experiments.[^23]
- Developer forum and support: Engage the community and access support channels for updates, clarifications, and best practices.[^24]

Operational readiness includes configuring API key permissions, aligning rate-limit monitoring with internal dashboards, and integrating backoff strategies at the client level. Testing in the Spot Testnet mitigates production risk and reveals edge cases in signature construction and time discipline before going live.

## Information Gaps

The following gaps exist in the publicly accessible Spot documentation referenced in this report and should be validated in implementation:

- Exact base URL endpoints for some REST and WebSocket hosts are redacted in the developer center pages. Implementers must confirm hosts via the latest official documentation or `exchangeInfo` responses.[^2]
- WebSocket User Data Stream subscription details (account, order, and balance update channels) are referenced indirectly by general documentation but not fully detailed here; consult the User Data Stream pages for exact payload schemas and subscribe/unsubscribe messages.[^4]
- Ed25519 and RSA parameterization specifics (key formats, header names) require direct review of the API key types FAQ and request security pages for complete implementation details.[^15]
- Comprehensive definitions for all order response conditional fields and their precise presence rules should be cross-checked in the trading endpoints documentation for specific order types and response modes.[^11]
- Exchange filters such as `PRICE_FILTER`, `LOT_SIZE`, `MARKET_LOT_SIZE`, `MIN_NOTIONAL`, and `NOTIONAL` are mentioned implicitly via account-level filters but not fully enumerated; implementers should consult the Filters documentation for complete specifications.[^10]
- WS API session authentication example messages (e.g., session logon and signature payloads) should be validated against the WS API request security documentation.[^15]

## References

[^1]: LIMITS | Binance Open Platform. https://developers.binance.com/docs/binance-spot-api-docs/rest-api/limits  
[^2]: General API Information | Binance Open Platform. https://developers.binance.com/docs/binance-spot-api-docs/rest-api/general-api-information  
[^3]: Market Data endpoints | Binance Open Platform. https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints  
[^4]: WebSocket Streams | Binance Open Platform. https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams  
[^5]: Rate limits | Binance Open Platform (WebSocket API). https://developers.binance.com/docs/binance-spot-api-docs/websocket-api/rate-limits  
[^6]: Binance Spot API to Increase Single Account Order Rate Limit. https://www.binance.com/en/support/announcement/detail/a9eb5f629afd46b48a4f3f24d31b9bb5  
[^7]: How to Avoid Getting Banned by Rate Limits? https://www.binance.com/en/academy/articles/how-to-avoid-getting-banned-by-rate-limits  
[^8]: HMAC Signature: What It Is and How to Use It for Binance API Security. https://www.binance.com/en/academy/articles/hmac-signature-what-it-is-and-how-to-use-it-for-binance-api-security  
[^9]: Different Order Types in Spot Trading - Binance. https://www.binance.com/en/support/faq/detail/8a2973eef1de429dbfad38ab878aa3eb  
[^10]: Account Endpoints | Binance Open Platform. https://developers.binance.com/docs/binance-spot-api-docs/rest-api/account-endpoints  
[^11]: Trading endpoints | Binance Open Platform. https://developers.binance.com/docs/binance-spot-api-docs/rest-api/trading-endpoints  
[^12]: Binance APIs. https://www.binance.com/en/binance-api  
[^13]: Binance Spot API Docs (GitHub). https://github.com/binance/binance-spot-api-docs  
[^14]: Binance.US API Documentation: Introduction. https://docs.binance.us/  
[^15]: Request Security (WebSocket API) | Binance Open Platform. https://developers.binance.com/docs/binance-spot-api-docs/websocket-api/request-security  
[^16]: Websocket Market Streams (USDT-M Futures) | Binance Open Platform. https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams  
[^17]: Websocket API General Info (USDT-M Futures) | Binance Open Platform. https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-api-general-info  
[^18]: General Info (USDT-M Futures) | Binance Open Platform. https://developers.binance.com/docs/derivatives/usds-margined-futures/general-info  
[^19]: SBE Market Data Streams | Binance Open Platform. https://developers.binance.com/docs/binance-spot-api-docs/sbe-market-data-streams  
[^20]: Frequently Asked Questions on API - Binance. https://www.binance.com/en/support/faq/detail/360004492232  
[^21]: Request Security (REST API) | Binance Open Platform. https://developers.binance.com/docs/binance-spot-api-docs/rest-api/request-security  
[^22]: Binance Spot Testnet. https://testnet.binance.vision/  
[^23]: Binance Demo. https://demo.binance.com/  
[^24]: Binance Developer Community Forum. https://dev.binance.vision