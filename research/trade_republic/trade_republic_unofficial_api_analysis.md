# Trade Republic Unofficial API Implementations: Libraries, Capabilities, Limits, and Legal Exposure

## Executive Summary

Trade Republic does not offer an official, public-facing application programming interface (API) for developers. All programmatic access used by the community today is “unofficial” in that it is neither published nor endorsed by Trade Republic Bank GmbH. Practitioners instead rely on reverse-engineered clients, web scrapers, and microservice wrappers that integrate with Trade Republic’s private interfaces, most notably the asynchronous WebSocket streams used by the official application for portfolio, market data, orders, and timeline feeds. The principal libraries are the Python package pytr, which exposes these private streams via subscriptions and commands, a follow-on repository by Zarathustra2 that documents additional capabilities and endpoints, and a lighter-weight Python wrapper named trade-republic-api distributed via PyPI. Complementary tooling includes a production-ready scraper for the Lang & Schwarz Trade Center portal and a microservice approach (tradepipe) that hosts these interactions as a network service.[^3][^1][^5][^9][^14]

Unofficial access splits into two families. The first is a WebSocket-based, asynchronous subscription model that mirrors the official app’s real-time flows for portfolio, cash, watchlists, tickers, performance histories, timelines, news, savings plans, and orders. pytr provides a curated command surface, tab-complete CLI, mass document downloads, transaction exports, and price alarm management. The second is a scraper-based approach targeting Lang & Schwarz Trade Center content, often used when API access is infeasible or when only web-visible data is needed. The microservice pattern wraps either approach into a deployable, networked component for integrations.[^2][^3][^9][^14]

Authentication flows mirror official channels. pytr supports a “web login” (the same path used by app.traderepublic.com, involving an app/SMS four-digit code) and an older “app login” requiring a device reset that generates and persists a private key to a keyfile and forces logout from mobile devices. Account pairing rules and two-factor authentication (2FA) are consistent with Trade Republic’s security posture: one paired device at a time, with re-linking subject to additional checks and potential temporary blocks.[^3][^15]

Capabilities vary by library. pytr’s asynchronous core can subscribe to portfolio, timeline, and market data topics; retrieve instrument and performance histories; manage watchlists; read and set price alarms; and place and cancel orders (market, limit, stop) with configurable expiry and warnings. It can also expose order cost queries and savings plan creation, parameterization, and cancellation via synchronous REST-style calls. Zarathustra2’s repository documents retrieval for cash, portfolio, instruments, tickers, and “Neon news,” as well as timeline event access and CSV export flows. trade-republic-api provides a minimal wrapper with documentation but sparse capability detail. Scrapers such as the Apify actor focus on Lang & Schwarz Trade Center content rather than the private WebSocket streams.[^2][^3][^1][^5][^9]

Limitations are material. There are no published, enforceable rate limits or service-level agreements (SLAs), and breaking changes跟进 upstream app updates are common; issues logged across repos show failures after app updates, intermittent “failed” or “not found” errors, and missing event handlers. Repositories vary in maintenance cadence, with some showing stagnation. Device logout on pairing resets is by design. In short, these are best-effort tools with non-deterministic stability.[^12][^13][^3][^1]

The legal exposure is unambiguous. Trade Republic’s Customer Agreement and Light User Agreement prohibit the use of access paths, programs, or interfaces not provided by Trade Republic outside of the Application. Violation can trigger extraordinary termination. The documents carve out exceptions only for dedicated interfaces provided by Trade Republic and certain regulated payment or account information services. The privacy notice further underscores data protection obligations. Community portfolio trackers explicitly state that Trade Republic does not provide an API, confirming the lack of official developer access. In practice, this means any use of unofficial interfaces carries contractual risk that must be acknowledged and actively mitigated.[^16][^17][^18][^19]

Actionable guidance: teams should confine usage to non-trading automations such as read-only analytics, document retrieval, and internal reconciliation; implement conservative concurrency and backoff; rotate credentials prudently; and prepare contingency plans (e.g., manual processes, export workarounds, or scrapers) in case upstream changes break clients overnight. Clear governance, legal review, and operational monitoring are mandatory.[^16][^2][^9]


## Methodology and Source Validation

This report is based on primary source materials from official Trade Republic legal documents, public-facing support pages on web login security and device linking, and the most active unofficial client repositories and package distributions. We triangulated capability claims and operational caveats across pytr (GitHub, PyPI, and documentation), the Zarathustra2 follow-on repository, the trade-republic-api wrapper (PyPI and Read the Docs), and the Apify Lang & Schwarz Trade Center scraper. We supplemented this with community issue trackers to characterize failure modes and breakages. Official engineering blog posts were used to frame Trade Republic’s internal engineering practices (for context only) and to clarify that there is no official public API program.[^1][^3][^4][^5][^7][^8][^9][^12][^13][^20][^21]

We adopted three validation principles. First, when capability claims conflicted across repos, we favored the most recent, actively maintained, and clearly documented client (pytr). Second, we treated legal agreements as authoritative on policy and risk, not as technical constraints to be worked around. Third, we treated issue trackers as reality checks on reliability and breakages, noting when repositories showed prolonged inactivity or unresolved regressions.

Constraints remain. The private protocols and endpoints used by clients are not publicly documented by Trade Republic and are subject to change without notice. Rate limits and SLAs are not published and therefore cannot be verified. Maintenance status of independent repos can change rapidly, and this report reflects the state of repositories as of the current time reference.


## Landscape of Unofficial Tools and Libraries

Two patterns dominate the unofficial landscape. The first is a reverse-engineered client that subscribes to the same real-time streams used by the official app, delivering low-latency updates for portfolio, market data, orders, and timelines. The second is scraper-based integration against web portals such as Lang & Schwarz Trade Center, extracting information that is visible in the browser but outside the private app channels. A third, smaller pattern wraps these integrations into a microservice for easier deployment and interoperation in larger systems.

pytr is the most feature-complete Python client. It uses an asynchronous WebSocket subscription model with a CLI and Python API for mass document downloads, portfolio views, price alarm management, transaction exports, and order placement/cancellation via specific calls. Zarathustra2’s repository documents a similar, earlier set of functions with both blocking and async interfaces, emphasizing timeline, cash, portfolio, instrument, ticker, and “Neon news” access. The trade-republic-api package is a lighter wrapper with minimal capability disclosures. For web-visible data, Apify hosts a Lang & Schwarz Trade Republic scraper with an HTTP API and SDKs, and Sannrox’s tradepipe packages the private API interactions as a microservice. The GitHub topic “trade-republic” captures additional utilities, such as data exporters geared toward Portfolio Performance.[^2][^3][^5][^9][^10][^14]

To ground the discussion, Table 1 compares the most referenced libraries across their core attributes.

To illustrate the differences in a single view, the following table contrasts scope, transport, authentication, capabilities, maintenance posture, license, and indicative community traction.

Table 1. Comparative overview of unofficial libraries and tools

| Project | Language | Primary transport | Authentication | Key capabilities | Maintenance (observed) | License | Community signals |
|---|---|---|---|---|---|---|---|
| pytr (pytr-org) | Python | Async WebSocket subscriptions + selected synchronous calls | Web login (app/SMS 4-digit code); App login via device reset and keyfile | Portfolio, cash, watchlists, market data, performance histories, timeline, news, price alarms, savings plans; place/cancel orders; cost queries; mass doc downloads and transaction export | Active releases and commits; issues tracked with recent updates | MIT | PyPI releases, CLI, docs; GitHub stars/forks reflect active use[^2][^3][^4] |
| TradeRepublicApi (Zarathustra2) | Python | Blocking and async clients | Phone number + PIN (constructor args) | Timeline events, cash, portfolio, instrument/stock details, ticker, “Neon news”; CSV export examples | Last major commit in 2023; open issues and reported breakages post-app updates | MIT | Community issues show regressions and feature requests[^1][^12] |
| trade-republic-api | Python | Wrapper around private interfaces (unspecified) | Not disclosed | Unofficial API wrapper (capability details sparse) | Last release in 2023; docs hosted on Read the Docs | BSD-2-Clause | PyPI package page with basic metadata[^5][^6][^7][^8] |
| Apify Lang & Schwarz Trade Republic scraper | Python/JS (clients) | HTTP API to scraping actor | Apify token | Scrapes LS-TC portal (web-visible), not private app streams | Platform-maintained actor; API documented | Not specified on page | API and SDKs; ready-made endpoints[^9][^10][^11] |
| tradepipe (microservice) | Not specified | Microservice wrapper for private API | Not specified | Hosts private API interactions as a service for integrations | Community repo; maintenance varies | Not specified | Exposes non-CLI service surface[^14] |

The key insight from Table 1 is not merely feature counts but the different operational philosophies. pytr embraces the private streaming model end-to-end, offering a comprehensive command surface and developer ergonomics. Zarathustra2’s project captures similar capabilities but shows more evidence of breakage as upstream changes roll out. Scrapers and microservices are valuable complements for web-visible data or for plugging into service-oriented architectures, but they do not substitute for private streams where near-real-time positions and order lifecycle visibility are required.

### pytr (Python interface to Trade Republic’s private API)

pytr is both a CLI and a Python library. Its asynchronous core subscribes to topics and receives typed updates via a receive loop. Subscriptions are managed explicitly and asynchronously for portfolio, cash, watchlists, market data (including performance histories), timeline, news, savings plans, and orders. For developer convenience, pytr also exposes blocking variants for common reads (e.g., portfolio) and selected synchronous calls for settings, order and savings plan cost queries, and payout confirmations.[^2][^3]

Authentication supports two paths. The default “web login” aligns with the app.traderepublic.com flow and relies on a four-digit code delivered via the Trade Republic app or SMS; this path keeps the primary device logged in, with occasional re-entry of the code. The older “app login” requires a device reset: pytr generates a private key pinned to the “device,” saves it to a keyfile, and logs out the existing mobile app. Credentials are stored locally (phone number and PIN) if the user opts in. The CLI includes subcommands for login, mass document downloads, portfolio views, ISIN details, price alarm CRUD, and exporting deposits/withdrawals to formats compatible with Portfolio Performance. Verbosity levels and debug logging are configurable.[^3]

### Zarathustra2/TradeRepublicApi (Python, blocking/async)

This repository codifies access to core data domains—portfolio, cash, instrument details, stock details, ticker, “Neon news”—and the account timeline with examples for CSV exports. It offers both blocking and asynchronous APIs, with the constructor taking phone number and PIN. Observed issues include failures after official app updates, intermittent “failed N” responses, and requests failing with platform errors. The repository notes a first-run error that may require a second attempt and explicitly acknowledges that device pairing will log out other devices. Maintenance cadence appears less active since 2023, with open issues accumulated over time.[^1][^12]

### trade-republic-api (PyPI wrapper)

trade-republic-api is a minimal, BSD-2-Clause licensed wrapper published on PyPI. Publicly visible documentation and metadata provide limited detail about capabilities, and the last release was in 2023. It is best treated as a skeleton or proof-of-concept rather than a full client.[^5][^6][^7][^8]

### Apify Lang & Schwarz Trade Republic Scraper (web scraping)

Apify provides a hosted scraping actor for the Lang & Schwarz Trade Center (LS-TC) portal. It exposes a conventional HTTP API with SDKs and example code. This approach is suitable when only web-visible data is needed or when private endpoints are inaccessible. The actor does not access Trade Republic’s private app streams; it scrapes the public LS-TC site.[^9][^10][^11]

### tradepipe (microservice)

tradepipe wraps private API interactions in a microservice, enabling non-Python or service-oriented deployments to integrate with Trade Republic’s private interfaces indirectly. While the exact language and runtime are not specified in the summary, the intent is clear: decouple the reverse-engineered client from application code by exposing a network interface.[^14]


## Authentication and Device Management

pytr provides two authentication methods that mirror the official experiences.

First, “web login” is the default and follows the same path as the official web application. The user initiates login, receives a four-digit code in the Trade Republic app or via SMS, and completes authentication in the browser or CLI context. This flow tends to keep the primary device session active, with occasional prompts to re-enter the code. It is the lowest-friction path for read-only operations and exports.[^3]

Second, the older “app login” requires a device reset. pytr generates a private key that pins the “device,” persists it to a keyfile, and logs out any active mobile device. The process requires the account phone number and PIN, followed by a 2FA token (sent via SMS) to complete the reset. This flow is more intrusive operationally but provides a persistent device identity that the server recognizes. It is documented as the original method and is maintained for backwards compatibility and certain automation needs.[^3]

These flows align with Trade Republic’s official security posture. The support page for web login confirms 2FA is always enforced; a password plus a second factor are required. The device-linking page explains that only one mobile device can be paired at a time, and re-linking may trigger additional identification measures, temporary blocks, or delays—all by design to protect the account. These constraints explain why unofficial clients report device logout during resets and why simultaneous mobile and CLI use is not supported. The privacy notice situates these flows within Trade Republic’s data protection regime, reinforcing the sensitivity of credentials and paired devices.[^15][^22][^18]

To clarify operational differences, Table 2 compares the two authentication methods.

Table 2. Authentication methods comparison

| Method | Steps | Device impact | 2FA requirements | Persistence | Operational risks | Notes |
|---|---|---|---|---|---|---|
| Web login | CLI or code initiates login; user receives 4-digit code in app/SMS and confirms | Typically keeps primary device logged in; occasional code re-entry | Four-digit code via app/SMS | Session-based | Re-prompts for code; potential session expiry | Default in pytr; mirrors app.traderepublic.com flow[^3][^15] |
| App login (device reset) | Provide phone and PIN; initiate device reset; provide SMS token; private key generated and saved to keyfile | Logs out mobile app; pairs this “device” via private key | SMS token required to complete reset | Keyfile persists device identity | Intrusive; forces mobile logout; keyfile must be safeguarded | Older method; still supported for automations[^3] |

The practical takeaway is straightforward. For short-lived automations, web login is convenient but may require periodic human interaction. For durable unattended operation, app login provides a persistent device identity at the cost of mobile logout and the obligation to protect the keyfile. Both methods must be operated with credential hygiene and awareness of the single-device pairing constraint.


## Capabilities by Library

While library descriptions overlap, their capabilities are not identical. The sections below synthesize what each client can reliably do, based on documented command surfaces and observed behaviors. Because unofficial endpoints can change without notice, teams should validate specific actions (e.g., order placement) against current library releases before production use.

### pytr: Capabilities

pytr’s asynchronous core centers on subscribing to topics and processing updates via a receive loop. Subscriptions cover portfolio and cash, watchlists, market data and performance histories, timeline events and details, news, savings plans, and orders. For convenience, it also exposes blocking calls for common reads (e.g., portfolio), plus synchronous methods for settings, order and savings plan cost queries, and payout workflows.

Market data includes instrument details and suitability, stock details, tickers per exchange, and performance histories with configurable timeframes and resolution. Timeline operations allow enumerating events since a cursor and retrieving detail views for orders and savings plans. Search and derivative search capabilities support instrument discovery. On the order side, pytr can fetch order overviews, available cash and sizes, pricing for orders, and place market, limit, and stop-market orders with configurable expiry (good-for-day, good-till-date, good-till-cancelled) and optional warnings. It can cancel orders by ID. Savings plans can be created, changed, and cancelled with parameters such as amount, interval, and start date configuration. Price alarms can be created and cancelled. Document operations include mass-downloading all PDFs from the timeline and exporting transactions to CSV and JSON for downstream analysis or reconciliation. The CLI offers tab completion and configurable verbosity and debug logging.[^2][^3]

To anchor these capabilities in a concise map, Table 3 lists representative functions by domain and whether they are asynchronous or blocking.

Table 3. pytr capability map (selected functions)

| Domain | Example functions | Sync/Async |
|---|---|---|
| Portfolio and cash | portfolio(), cash(), available_cash_for_payout(), portfolio_status(), portfolio_history(timeframe) | Async; blocking convenience wrappers available[^2] |
| Watchlists | watchlist(), add_watchlist(isin), remove_watchlist(isin) | Async[^2] |
| Market data | instrument_details(isin), stock_details(isin), ticker(isin, exchange), performance(isin, exchange), performance_history(..., resolution) | Async[^2] |
| Timeline | timeline(after), timeline_detail(timeline_id), timeline_detail_order(order_id), timeline_detail_savings_plan(savings_plan_id) | Async[^2] |
| Search | search(query, ...), search_derivative(underlying_isin, product_type), suggested tags | Async[^2] |
| Orders | order_overview(), cash/size available, price_for_order(), market_order(), limit_order(), stop_market_order(), cancel_order(order_id) | Async; selected cost queries are synchronous[^2] |
| Savings plans | savings_plan_overview(), savings_plan_parameters(), create/change/cancel savings plan | Async; cost queries synchronous[^2] |
| Price alarms | price_alarm_overview(), create_price_alarm(), cancel_price_alarm() | Async[^2] |
| News | news(isin), news_subscriptions(), (un)subscribe_news(isin) | Async[^2] |
| Documents | dl_docs (download timeline PDFs; export transactions CSV and JSON) | CLI; underlying calls async[^3] |
| Payout/settings | payout(amount), confirm_payout(process_id, code), settings() | Synchronous[^2] |

The operational pattern is to set up subscriptions for topics of interest, then drive a receive loop that dispatches updates by subscription ID or type. This model scales well for dashboards and monitoring services but requires careful error handling and reconnection logic to cope with upstream instabilities documented in issue trackers.[^2][^12]

### Zarathustra2/TradeRepublicApi: Capabilities

The repository documents access to the account timeline (including dividends, deposits, withdrawals, savings plan executions, buys, and sells), cash and portfolio data, instrument details, stock details, ticker access, and “Neon news.” It offers both blocking and asynchronous implementations and includes examples for exporting the timeline to CSV. Known caveats include a first-run error that may require a second execution, German-only export testing, and explicit statements that using the API will log out other devices due to pairing constraints. Maintenance has been less active since 2023 and issues track breakages after app updates and intermittent failures.[^1][^12]

### trade-republic-api: Capabilities

trade-republic-api presents as an unofficial wrapper with limited public detail. Given the 2023 release and sparse documentation, it should be approached as a minimal interface or starting point for experimentation rather than a production client.[^5][^7]

### Apify LS-TC Scraper: Capabilities

Apify’s actor is designed for the Lang & Schwarz Trade Center web portal. It is a scraper, not a private API client, and is best used for web-visible data or when private endpoints are inaccessible. The actor exposes a documented HTTP API and SDKs for integration, enabling programmatic scraping workflows with standard HTTP authentication via an Apify token.[^9][^10][^11]


## Market Data and Account Information Access

From a developer’s perspective, theunofficial clients provide a coherent, if unofficial, mirror of what the official application shows. Market data spans instrument details, stock-level attributes, tickers per exchange, and performance histories with selectable timeframes and resolution. Account information includes portfolio snapshots and histories, cash positions, watchlists, and the full timeline of events with detail drilldowns for orders and savings plans. Search endpoints support discovery by tags and derivatives. Price alarms can be read and managed, and news can be subscribed to per instrument.

To ground the scope, Table 4 maps common data domains to example calls. This is indicative rather than exhaustive; the exact method names vary by client and version.

Table 4. Data domains and example calls (indicative)

| Domain | Example calls (pytr-style) | Notes |
|---|---|---|
| Instruments | instrument_details(isin), stock_details(isin), instrument_suitability(isin) | Includes metadata for exchanges and suitability flags[^2] |
| Tickers and performance | ticker(isin, exchange), performance(isin, exchange), performance_history(isin, timeframe, exchange, resolution) | Timeframes include short and long ranges; resolution in milliseconds[^2] |
| Portfolio and cash | portfolio(), cash(), portfolio_history(timeframe), portfolio_status() | Portfolio includes positions and valuation; cash includes balances[^2] |
| Watchlists | watchlist(), add_watchlist(isin), remove_watchlist(isin) | Watchlist management for user-defined sets[^2] |
| Timeline | timeline(after), timeline_detail(timeline_id), timeline_detail_order(order_id), timeline_detail_savings_plan(savings_plan_id) | Enumerates events and provides detail payloads[^2] |
| Price alarms | price_alarm_overview(), create_price_alarm(isin, price), cancel_price_alarm(alarm_id) | Alarm management for notifications[^2][^3] |
| News | news(isin), (un)subscribe_news(isin) | Instrument-specific news streams[^2] |
| Search | search(query, ...), search_derivative(underlying_isin, product_type), suggested tags | Supports filters and derivates lookup[^2] |

Because these are unofficial, developers should expect structural changes over time. Libraries may add fields, drop support for certain topics, or change response shapes when upstream app semantics change, which is reflected in issue logs across repositories.[^2][^12]


## Order Placement and Savings Plans

Order-related functions in pytr cover the full lifecycle: querying order overview, available cash and sizes, indicative prices, and placing market, limit, and stop-market orders with granular expiry and warning controls. Orders can be cancelled by ID. Savings plans can be created, changed, and cancelled, with parameterization for amount, interval (weekly, twice per month, monthly, quarterly), and start-date rules. Cost queries are exposed via synchronous methods, as are payout flows and confirmations.[^2]

To make the parameter space tangible, Table 5 catalogs order methods with key parameters and constraints as documented. Not all parameters apply to all order types; the library surfaces them conditionally.

Table 5. Order methods and parameters (pytr)

| Method | Key parameters | Notes |
|---|---|---|
| market_order | isin, exchange, order_type (buy/sell), size, sell_fractions (bool), expiry (gfd/gtd/gtc), expiry_date (optional), warnings_shown (list) | Market orders with optional fractional sell handling[^2] |
| limit_order | isin, exchange, order_type, size, limit, expiry, expiry_date (optional), warnings_shown | Limit price controls execution threshold[^2] |
| stop_market_order | isin, exchange, order_type, size, stop, expiry, expiry_date (optional), warnings_shown | Stop-market triggers for sell/buy protection[^2] |
| cancel_order | order_id | Cancels pending orders by ID[^2] |
| price_for_order | isin, exchange, order_type | Returns indicative price for planning[^2] |
| size_available_for_order | isin, exchange | Returns tradable size constraints[^2] |
| order_overview | none | Returns current orders for the account[^2] |
| savings_plan_* | isin, amount, interval, start_date, start_date_type, start_date_value | Create/change/cancel; cost queries via synchronous methods[^2] |

Because unofficial endpoints evolve, teams should test order placement in paper or highly controlled conditions, log full request/response traces, and implement idempotent cancellation and verification loops to detect partial fills and reconcile execution states. This is particularly important given the reliability issues documented in community trackers.[^12]


## Technical Limitations and Reliability Risks

The most important fact about unofficial APIs is that they are not documented or guaranteed by Trade Republic. Endpoints, payloads, authentication tokens, and device pairing rules can change at any time as part of normal product development. Libraries that worked yesterday may break after an app update, and there is no official support to fall back on. Issue trackers across projects provide ample evidence: users report “failed N” errors, “could not find resource” messages during downloads, and portfolio subscriptions that stop working after upstream changes. Some repos show prolonged periods without maintenance, with issues piling up over many months. In practice, every production deployment needs an explicit plan to detect, contain, and recover from breakages.[^12][^13]

Device pairing rules introduce operational friction. Because only one mobile device can be paired at a time, app-login device resets will log out the mobile app. Web-login sessions can require periodic re-entry of the 4-digit code. These are not mere annoyances; they are constraints baked into the authentication model and must be designed around.

Rate limits, if any, are opaque. There are no published quotas or SLAs for unofficial clients, and attempts to infer them empirically are fragile. Conservative design calls for concurrency caps, exponential backoff, jitter, and circuit breakers. For critical workflows, have a fallback process ready—manual execution, export-based reconciliation, or scraper alternatives—so that a sudden breakage does not halt operations.[^12][^13]

Table 6 summarizes recurring reliability issues observed in community trackers.

Table 6. Reliability issues (illustrative excerpts)

| Project | Issue title | Symptom | Status | Opened | Notes |
|---|---|---|---|---|---|
| Zarathustra2/TradeRepublicApi | After App update requests fail with “failed 30” | Client errors after upstream app updates | Open | 2023-10-12 | Indicates breaking changes in private endpoints[^12] |
| pytr-org/pytr | Getting error when trying to dl_docs NOT_FOUND | Continuous errors during mass document download | Open | 2024–2025 | Suggests downstream resource changes or timeline schema shifts[^13] |
| pytr-org/pytr | Could not find resource when executing dl_docs | Download fails at a reproducible point | Open/likely resolved in later release | 2023–2024 | Users asked to try latest release; regression history suggests moving target[^13] |
| pytr-org/pytr | Should this library be able to create orders? | Orders method appears unresponsive | Discussion | 2024 | Community uncertainty about order support status and scope[^13] |

The central lesson is not to over-automate mission-critical flows without guardrails. Use read-only functions for dashboards and analytics, implement robust retries with backoff, and plan for manual overrides.


## Legal and Compliance Considerations

Trade Republic’s legal agreements set clear boundaries. The Customer Agreement provides that services “can only be used via this Application… as well as other access channels provided by Trade Republic.” It prohibits “any use of the features and services provided by Trade Republic through access paths, programs and/or other interfaces not provided by Trade Republic outside of the Application.” Violation “of this prohibition” may result in “extraordinary termination.” The agreement recognizes exceptions only for dedicated interfaces provided by Trade Republic and regulated payment initiation or account information services, and it permits customers to use their credentials with carefully selected third-party services of those types. The Light User Agreement contains equivalent prohibitions for non-custody app-only usage. Trade Republic’s privacy notice frames the handling of personal data in this ecosystem.[^16][^17][^18]

This contractual posture has practical implications. Any use of reverse-engineered interfaces or scrapers to access Trade Republic content or functionality outside the official app or sanctioned interfaces creates a risk of account termination and other remedial actions. For organizations, this risk extends to operational disruption and potential reputational harm. Community resources, such as the help page used by a portfolio tracker, confirm that Trade Republic does not provide an API—underscoring that there is no sanctioned developer program to join.[^19]

Table 7 provides a clause-to-risk matrix to orient teams.

Table 7. Legal clause-to-risk mapping

| Agreement and clause (paraphrase) | Obligation or prohibition | Risk to user | Mitigation |
|---|---|---|---|
| Customer Agreement: services only via Application and official channels | Must use official app/channels | Contract breach if using unofficial interfaces; potential extraordinary termination | Avoid trading automation via unofficial APIs; confine to read-only or sanctioned tools[^16] |
| Customer Agreement: prohibition on non-provided access paths, programs, interfaces | No use of unofficial interfaces outside the app | Same as above | Perform legal review; consider scrapers for public data only if compliant[^16] |
| Customer Agreement: exceptions for dedicated interfaces and regulated payment/account information services | Permitted use via sanctioned interfaces only | Unclear status for other third parties | Confirm interface status with Trade Republic before use[^16] |
| Light User Agreement: prohibition on non-provided interfaces | Same as above for light users | Termination of light access | Treat light accounts as non-trading; avoid automation[^17] |
| Privacy Notice: data protection obligations | Personal data handling must comply | Regulatory and contractual exposure | Minimize data collection; secure storage; access controls[^18] |

In short, the legal environment permits only narrow, regulated third-party access and only through interfaces explicitly provided by Trade Republic. Unofficial access lies outside those boundaries and should not be used for trading operations without explicit, written approvals—which are not currently available to the public.


## Community Resources and Maintenance Channels

The unofficial ecosystem is held together by public repositories, package distributions, and issue trackers. The pytr project is the most active, with published releases, a CLI, documentation, and an issues tracker where regressions and breakages are logged and sometimes resolved. Zarathustra2’s repository aggregates capability examples and issues that often trace back to upstream changes. The trade-republic-api wrapper offers limited documentation via Read the Docs and a PyPI page for distribution. Apify maintains a managed scraping actor and API for LS-TC content. Microservice wrappers like tradepipe provide integration surfaces but inherit the underlying stability risks. The broader GitHub topic page collects utilities such as exporters aimed at Portfolio Performance. Unofficial channels do not include a dedicated forum maintained by Trade Republic; their official engineering blog discusses internal practices but does not publish a public API.[^2][^12][^7][^9][^14][^21][^20]

From a support standpoint, community issue trackers are the de facto venues for problem diagnosis. Teams should expect to contribute reproductions, logs, and sometimes code patches. Given the absence of SLAs or official support, the maintenance outlook is determined by maintainers’ availability and upstream stability. Where possible, prefer open issues with active engagement and recent releases.


## Implementation Best Practices and Security Hardening

Given the legal and technical constraints, the safest path is to use unofficial clients for non-critical, read-mostly operations and to adopt disciplined engineering practices where any automation is necessary.

First, favor non-trading automations: portfolio analytics, document retrieval, and reconciliation exports. Avoid programmatic order placement unless you have explicit risk acceptance and a tested rollback path. The Customer Agreement’s prohibition on unofficial interfaces, combined with the potential for instantaneous breakages, makes trading automation disproportionately risky.[^16]

Second, handle credentials and device pairing with care. If using web login, store credentials securely and anticipate periodic 2FA prompts. If using app-login device reset, protect the keyfile as you would a private key; loss or compromise undermines account security. Respect the single-device pairing rule and design for forced mobile logout when resetting devices. Follow the spirit of Trade Republic’s security guidance: never share credentials, avoid unsecured storage, and keep devices patched and locked.[^3][^15]

Third, operate conservatively. Implement client-side rate limiting, bounded concurrency, exponential backoff with jitter, and circuit breakers for subscriptions and calls. Monitor libraries for new releases and test upgrades in staging before production. Subscribe to upstream change signals via issue trackers and release notes. Maintain fallback modes—manual execution, export-based reconciliation, or scrapers—so you can degrade gracefully when private endpoints change.[^12][^13]

Fourth, log comprehensively and keep traceable audit trails. Capture request parameters, subscription IDs, response shapes, and timing. Store exports (CSV/JSON) and document digests with immutable identifiers and hashes to support audits. For order-adjacent workflows, implement verification loops and idempotent cancellations to mitigate partial states.

Finally, govern responsibly. Conduct legal reviews before deploying any automation, document risk decisions, and require sign-off for production use. If you operate a microservice wrapper (e.g., tradepipe), segment access, enforce authentication, and apply least privilege across your own consumers.[^14][^16]


## Conclusion and Strategic Guidance

Unofficial Trade Republic tools provide powerful, albeit fragile, access to portfolio, market, and account data, and in some cases, to order placement and savings plan management. pytr is the most capable and actively maintained, leveraging an asynchronous subscription model with a practical CLI and export utilities. Zarathustra2’s project documents similar capabilities but with clearer signs of upstream breakage and maintenance drift. trade-republic-api is a minimal wrapper, and Apify’s scraper addresses web-visible data via LS-TC. Microservices like tradepipe can simplify integration architectures but do not remove underlying legal and reliability risks.[^2][^1][^5][^9][^14]

The strategic posture for practitioners should be risk-first. Legal agreements prohibit unofficial interfaces; use of these tools therefore carries contractual exposure, particularly for trading operations. Technically, breakages are common and unpredictable, with no SLAs or official support. The responsible approach is to limit automation to non-trading workflows, secure credentials and keyfiles, implement conservative operational controls, and maintain explicit fallback procedures.

For near-term work, restrict usage to read-only data, mass document downloads, and export-based reconciliation. Keep close watch on issue trackers for signs of regressions, and test upgrades before rolling them into production. For mid-term planning, maintain a strict governance process, re-validate legal risks if circumstances change, and consider whether web scraping or manual processes can cover critical use cases with lower exposure. Over the long term, the only way to eliminate this class of risk is to avoid unofficial interfaces entirely or to operate within the narrow, regulated exceptions that Trade Republic recognizes. In the absence of an official developer program, the conservative strategy remains the sustainable one.[^16][^2]


## Information Gaps

- Trade Republic publishes no official public API documentation; all endpoints and protocols are reverse-engineered and subject to change without notice.
- There are no published rate limits or SLAs for unofficial clients; stability is unquantified and varies by library and upstream changes.
- Maintenance status of community repositories can change rapidly; the assessments herein reflect current observations.
- Definitive scope for order placement via unofficial clients is inconsistent across libraries; capabilities may change after app updates.
- Official exceptions for regulated third-party services do not translate into an open developer API; permissioned interfaces are not publicly documented.


## References

[^1]: Zarathustra2/TradeRepublicApi: Unofficial trade republic API - GitHub. https://github.com/Zarathustra2/TradeRepublicApi  
[^2]: pytr-org/pytr: Use TradeRepublic in terminal and mass document download - GitHub. https://github.com/pytr-org/pytr  
[^3]: pytr · PyPI. https://pypi.org/project/pytr/  
[^4]: Overview - trade-republic-api 0.0.1 documentation - Read the Docs. https://trade-republic-api.readthedocs.io/en/latest/readme.html  
[^5]: trade-republic-api · PyPI. https://pypi.org/project/trade-republic-api/  
[^6]: unimun/trade-republic-api - GitHub. https://github.com/unimun/trade-republic-api  
[^7]: Contents — trade-republic-api 0.0.1 documentation - Read the Docs. https://trade-republic-api.readthedocs.io/  
[^8]: trade-republic-api 0.0.1 documentation - Read the Docs. https://trade-republic-api.readthedocs.io/  
[^9]: Lang & Schwarz Trade Republic Scraper (www.ls-tc.de) - Apify. https://apify.com/anbusch/lang-schwarz/api  
[^10]: Lang & Schwarz Trade Republic Scraper API - Apify. https://apify.com/anbusch/lang-schwarz/api/python  
[^11]: Lang & Schwarz Trade Republic Scraper · API reference. https://apify.com/anbusch/lang-schwarz/api  
[^12]: Issues · Zarathustra2/TradeRepublicApi - GitHub. https://github.com/Zarathustra2/TradeRepublicApi/issues  
[^13]: Issues · pytr-org/pytr - GitHub. https://github.com/pytr-org/pytr/issues  
[^14]: Sannrox/tradepipe: Microservice for the private Trade Republic API - GitHub. https://github.com/Sannrox/tradepipe  
[^15]: Is my login to the web version secure? - Trade Republic Support. https://support.traderepublic.com/en-fi/685-Is-my-login-to-the-web-version-secure  
[^16]: Customer Agreement (NL-en) - Trade Republic (PDF). https://assets.traderepublic.com/assets/files/CA_NL-en.pdf  
[^17]: Light User Agreement (2023-09-23) - Trade Republic (PDF). https://assets.traderepublic.com/documents/PT/LIGHT_CONTRACT_CUSTOMER_AGREEMENT_20231023090727.pdf  
[^18]: Privacy Notice for the Trade Republic App (PDF). https://assets.traderepublic.com/assets/files/app_privacy_policy_en.pdf  
[^19]: Trade Republic | Delta by eToro Help Center. https://support.delta.app/en/articles/10576200-trade-republic  
[^20]: Trade Republic Engineering. https://engineering.traderepublic.com/  
[^21]: GitHub Topics: trade-republic. https://github.com/topics/trade-republic  
[^22]: How can I link another device to my account? - Trade Republic Support. https://support.traderepublic.com/nl-nl/1541-How-can-I-link-another-device-to-my-account  
[^23]: Trade Republic: Broker & Bank — Google Play. https://play.google.com/store/apps/details?id=de.traderepublic.app&hl=en_US