# DEGIRO Unofficial APIs: Libraries, Capabilities, Risks, and Community Ecosystem (2025)

## Executive Summary

DEGIRO does not offer an official, public-facing application programming interface (API). Its helpdesk explicitly states that connections to external applications are not supported, and the platform must be accessed via its web and mobile interfaces. This policy frames a parallel ecosystem of reverse‑engineered, unofficial clients that replicate web sessions and invoke internal endpoints to deliver features such as order placement, market data retrieval, and portfolio access.[^1]

The unofficial library landscape is diverse but fragmented. The most active and feature‑rich projects are a TypeScript client for Node.js backends and multiple Python packages, including synchronous and asynchronous implementations. Notably, icastillejogomez’s TypeScript library covers orders, portfolio, cash funds, product search, and account data, and explicitly enforces backend‑only usage due to Cross‑Origin Resource Sharing (CORS) constraints.[^5][^6] In Python, projects range from the long‑standing DegiroAPI (login, product search, real‑time and historical series, orders, transactions, order history) to newer asynchronous variants such as degiroasync (two‑layer API, with high‑level abstractions and a low‑level mirror of DEGIRO’s HTTP interface), and degiro‑connector (PyPI distribution with 2FA support).[^8][^9][^11][^12][^13]

Authentication in these libraries typically uses username/password plus a one‑time password (OTP) when two‑factor authentication (2FA) is enabled. Session tokens such as JSESSIONID are commonly reused to avoid repeated OTP prompts. However, community reports highlight fragility: changes in headers such as User‑Agent may break previously working sessions, and library code can stop functioning after platform changes, as evidenced by archived repositories and intermittent outages.[^4][^6][^10]

Operationally, unofficial clients face constraints. Front‑end use is blocked by CORS, pushing implementers to server‑side deployment. Libraries report rate limiting and throttling without official thresholds, and robustness is undermined by frequent, unannounced changes to DEGIRO’s internal endpoints. The absence of official support and service‑level agreements (SLAs) translates into tangible risks such as sudden breakage, account impact, and potential terms‑of‑service violations.[^4][^6][^8][^10]

In practice, the libraries can cover practical trading workflows: authenticating (including 2FA), listing and searching instruments, retrieving quotes and historical series, viewing portfolio and cash funds, placing and deleting orders (market, limit, stop, stop‑limit), and reviewing orders and transactions. Yet, ongoing maintenance and careful operational design are essential to mitigate breakage and compliance risk. Teams considering integration should prefer actively maintained libraries, implement backend‑only architectures, use conservative request pacing, and prepare fallbacks to manual processes if the unofficial interface fails.[^1][^5][^8][^9]

Information gaps remain. DEGIRO does not disclose internal rate limits or throttling policies. Recent changes in platform endpoints and responses may have altered functionality since the last commits of some libraries, and the exact mapping of all order fields requires deeper reverse‑engineering or fresh empirical testing. Legal interpretations of client agreements are context‑specific and should be addressed with counsel.

## Official Context: DEGIRO’s API Stance and Platform Boundaries

DEGIRO’s official position is unambiguous: there is no public API, and direct connections from external applications to a user’s DEGIRO account are not supported. Users must access the platform through the official web interface or the mobile application.[^1] The mobile app provides broad trading and portfolio functionality and incorporates robust security measures, including two‑factor authentication and encryption. Nevertheless, mobile usage remains within DEGIRO‑controlled clients rather than third‑party integrations.[^2][^3]

Unofficial clients emerge from reverse‑engineering the web interface’s HTTP traffic. They mimic authenticated sessions, parse responses, and invoke internal endpoints to provide programmability. This approach allows feature parity in some areas but lives outside official support and is subject to breakage whenever DEGIRO updates its frontend, authentication checks, or endpoint structures.

### Implications of No Official API

The absence of an official API forces implementers into a cat‑and‑mouse game with platform changes. Libraries may function for months and then fail abruptly when DEGIRO modifies its web application. This instability has played out repeatedly in community channels and repository histories: an archived Node.js client that once enabled order automation is now read‑only, reflecting the maintenance burden and the inherent fragility of reverse‑engineered integrations.[^4] Teams relying on such clients must invest in ongoing upkeep, adopt conservative operational patterns, and maintain contingency plans to avoid trading disruption.

## Unofficial Library Landscape (2025)

The unofficial ecosystem spans multiple programming languages, with the most mature projects in Python and TypeScript for Node.js. Packaging, documentation quality, and maintenance activity vary. Some repositories have not been updated in years; others continue to evolve with typed interfaces and structured endpoints.

To illustrate the breadth of implementations, the following table summarizes notable libraries and their attributes.

Table 1: Unofficial libraries overview

| Repository / Package | Language | License | Key capabilities | 2FA / OTP support | Maintenance status | Stars (approx.) |
|---|---|---|---|---|---|---|
| icastillejogomez/degiro‑api | TypeScript (Node.js) | MIT | Orders (create, execute, delete), portfolio, cash funds, product search, account data; typed responses; backend‑only due to CORS | Yes (OTP) | Active last commits 2022; released 1.0.4 in Dec 2021 | ~225 |
| pladaria/degiro | Node.js | MIT | Buy/sell automation, product search, ask/bid retrieval, portfolio/cash funds; session reuse; 2FA parameter | Yes | Archived since 2021; last commit 2019 | ~365 |
| lolokraus/DegiroAPI | Python | MIT | Login/logout; search_products; product_info; real‑time/historical prices; stock lists; orders (LIMIT/MARKET/STOPLIMIT/STOPLOSS); transactions; order history | Not explicitly stated; library predates broader 2FA adoption | Last commit 2020; release 0.9.5 in Jun 2020 | ~227 |
| degiroasync | Python | Not specified in docs | Async two‑layer API: high‑level (api) and low‑level (webapi); login with 2FA; portfolio; products; price series; search; orders/check/confirm (webapi) | Yes (both layers) | Docs updated 2022–2023; GitHub activity through Dec 2023 | ~2 |
| degiro‑connector | Python | BSD‑3‑Clause (GitHub) | Connection parameters; 2FA parameter; real‑time tag; PyPI distribution | Yes | PyPI package available; ongoing use in community | N/A |
| python‑degiro (PyPI) | Python | Not specified on page | Unofficial API; claims real‑time and historical data for products | Not specified | PyPI listing; details vary by version | N/A |

Sources: library READMEs, documentation, and package registries.[^5][^6][^8][^9][^11][^12][^13][^14][^16][^17]

Two significant themes emerge. First, the TypeScript client offers the most comprehensive and typed coverage of orders, account, and portfolio endpoints, explicitly acknowledging backend‑only deployment. Second, Python options provide flexibility (sync/async) and a rich set of market data functions, but stability guarantees vary and documentation depth differs across projects.

### Node.js / TypeScript Clients

The icastillejogomez TypeScript client is backend‑only by design; attempts to use it from a browser will hit CORS restrictions enforced by DEGIRO’s web application. The client surfaces session management functions (login, logout, isLogin, getJSESSIONID) and exposes account endpoints such as getAccountConfig, getAccountData, getAccountState, and getAccountReports. It also includes portfolio, cash funds, product search, and a full set of order operations (getOrders, getHistoricalOrders, createOrder, executeOrder, deleteOrder). OTP is supported, and environment variables assist with debugging and session reuse.[^5][^6]

By contrast, the older pladaria Node.js library demonstrates how fragile such projects can be. Although it once provided order automation and product search with optional 2FA, it was archived in 2021 and is now read‑only—illustrating the maintenance risk teams assume when building on unofficial foundations.[^4]

### Python Clients

DegiroAPI is a well‑known Python library covering authentication, product discovery (search and info), real‑time and historical price series, stock lists for indices, portfolio and cash funds, transactions, order history, and order placement. Order types include LIMIT, MARKET, STOPLIMIT, and STOPLOSS, with execution time types such as daily and unlimited validity. The library’s last commit dates to 2020, and users have reported breakage episodes that sometimes require header adjustments (for example, changing the User‑Agent) to restore functionality.[^8][^9][^10]

degiroasync takes a different approach by offering two layers. The high‑level API (degiroasync.api) abstracts common tasks such as login (including 2FA), portfolio retrieval, product information, price series, search, news, company profiles, order checking, and orders/history. The low‑level layer (degiroasync.webapi) closely mirrors DEGIRO’s HTTP interface, enabling finer control at the cost of fewer guards. Documentation emphasizes API instability until version 1.0 and notes a current gap: the high‑level API lacks confirm_order, requiring use of the low‑level layer for order confirmation.[^11][^12]

degiro‑connector is distributed via PyPI and highlights connection parameters for username/password and 2FA, catering to implementers who prefer a packaged distribution. Meanwhile, python‑degiro on PyPI positions itself as an unofficial API with real‑time and historical data capabilities for products, though its features may vary by release. The FKatenbrink project is another Python module reported in the ecosystem, offering a simple installation path via pip.[^13][^14][^16][^17]

### Other Language Implementations

Beyond Python and TypeScript/Node.js, a Dart wrapper provides a simplified interface to DEGIRO’s APIs, explicitly noting their unofficial nature and cautioning that it does not enable certain capabilities. It demonstrates interest across language communities, albeit with narrower scope and fewer guarantees than the primary TypeScript and Python clients.[^15]

## Authentication and Session Management

Unofficial clients replicate the web login flow: username and password, followed by OTP when 2FA is enabled. Implementation details vary by library, but the objective is consistent—establish an authenticated session and reuse it to avoid repeated OTP prompts. The TypeScript client, for example, offers getJSESSIONID and related session helpers, while the Node.js pladaria library permits passing a oneTimePassword directly or via environment variables. degiroasync includes login with 2FA in both its high‑level and low‑level APIs. degiro‑connector exposes a 2FA parameter in its connection configuration.[^4][^5][^11][^13]

Community experience points to subtle dependencies in session management. A reported breakage for a Python client was resolved by changing the User‑Agent header, indicating that DEGIRO may validate header patterns as part of anti‑automation or security controls. In other words, a seemingly innocuous header change can be the difference between a successful login and an unexpected failure.[^10]

Table 2: Authentication mechanisms by library

| Library | Credentials + OTP | 2FA flow | Session token reuse | Notes |
|---|---|---|---|---|
| icastillejogomez/degiro‑api | Yes | OTP supported | JSESSIONID and session helpers | Backend‑only deployment; environment variables assist debugging[^5][^6] |
| pladaria/degiro | Yes | oneTimePassword parameter | sessionId and account params | Archived; functional but not maintained[^4] |
| DegiroAPI (Python) | Yes | Not explicitly documented | Session via login | Reported header sensitivity (User‑Agent)[^8][^10] |
| degiroasync (Python) | Yes | Login with 2FA | Not specified; session reuse implied | Two‑layer API; webapi mirrors HTTP[^11][^12] |
| degiro‑connector (Python) | Yes | Explicit 2FA parameter | Not specified | PyPI distribution[^13] |
| python‑degiro (PyPI) | Unclear | Unclear | Unclear | Features vary by release[^14] |
| FKatenbrink/degiro‑api (Python) | Unclear | Unclear | Unclear | Simple pip install; limited docs[^17] |

## Market Data Access

Market data features in the unofficial libraries cover product search, instrument metadata, real‑time quotes, and historical time series. DegiroAPI, for instance, provides search_products and product_info, returns real‑time price data by product ID, and supports historical series through interval types such as One_Day, One_Week, One_Month, and longer spans. It also surfaces stock lists for specific indices via group IDs, allowing users to retrieve constituent securities.[^8]

degiroasync organizes price series under its high‑level API and enables product search, exchange information, news retrieval, and company profiles. These functions are useful for research and monitoring workflows and can feed portfolio analytics and alerting pipelines.[^11][^12]

Community projects often repurpose DEGIRO data for specific tasks. The degiro_portfolio_analytics repository integrates account connections for portfolio dashboards; the screener project exports all stock securities from DEGIRO into a Pandas dataframe for screening and analysis. Such tools demonstrate how unofficial clients can power higher‑level analytics even when formal APIs are unavailable.[^19][^23]

Table 3: Market data coverage by library

| Library | Search | Product info | Real‑time quotes | Historical series | News / company info | Stock lists |
|---|---|---|---|---|---|---|
| DegiroAPI (Python) | Yes | Yes | Yes | Yes (multiple intervals) | No | Yes (via group IDs)[^8] |
| degiroasync (Python) | Yes | Yes | Not explicitly stated | Yes (price series) | Yes | No[^11][^12] |
| icastillejogomez/degiro‑api (TypeScript) | Yes | Yes (getProductDetails) | Not explicitly stated | Not explicitly stated | Yes (getNews) | getFavouriteProducts, getPopularStocks[^5][^6] |
| python‑degiro (PyPI) | Yes (claims) | Yes (claims) | Yes (claims) | Yes (claims) | Not specified | Not specified[^14] |

### Practical Data Retrieval Patterns

Typical patterns include resolving instrument identifiers (product IDs) through search, then fetching metadata and price series for analytics or display. Libraries expose common interval types, enabling daily, weekly, monthly, and multi‑year views. Some projects list index constituents via predefined group IDs, which is convenient for basket operations and rebalancing. In practice, implementers should cache metadata locally to minimize calls, refresh quotes and series on a measured cadence, and avoid burst polling that may trigger throttling.[^8]

## Order Placement and Management

Order placement is a core capability of several unofficial clients. The TypeScript library provides comprehensive order functions: createOrder, executeOrder, and deleteOrder, alongside retrieval of current and historical orders. It supports multiple order types, including Limited, Market, Stop‑Loss, and Stop‑Loss Limit, and references account and portfolio context in its typed API surface.[^5][^6]

DegiroAPI implements buy/sell orders with order types LIMIT, MARKET, STOPLIMIT, and STOPLOSS, and offers execution time types such as daily validity and unlimited validity. It also supports deletion of open orders by order ID. Product identification relies on DEGIRO’s internal product IDs, which are resolved via search functions prior to order placement.[^8]

The Node.js pladaria library also exposes order types and time‑in‑force parameters, reflecting the breadth of options available via the web interface. Notably, its support for stop‑loss variants underscores the importance of careful parameter validation for order correctness and risk management.[^4]

Table 4: Supported order types and parameters

| Library | Order types | Time‑in‑force | Parameter notes |
|---|---|---|---|
| icastillejogomez/degiro‑api (TypeScript) | Limited, Market, Stop‑Loss, Stop‑Loss Limit | Not explicitly documented | Requires productId; typed order creation; backend‑only due to CORS[^5][^6] |
| DegiroAPI (Python) | LIMIT, MARKET, STOPLIMIT, STOPLOSS | Daily (1), Unlimited (3) | Requires productId; limit/stop prices as applicable[^8] |
| pladaria/degiro (Node.js) | Limited, Market, Stop‑Loss, Stop‑Limited | Day, Permanent | stopPrice required for stop variants; size and price fields[^4] |

### Order Workflow Nuances

In practice, robust order workflows include pre‑trade checks (e.g., validating product ID and price bounds), up‑to‑date account state, and confirmation routines before submission. degiroasync’s high‑level API currently lacks confirm_order, so implementers must drop to the low‑level webapi layer to finalize orders. Typed clients reduce simple errors, but the underlying fragility of unofficial endpoints demands extra validation and cautious retries.[^11]

## Account Information and Portfolio Access

Account and portfolio features are well covered across libraries. The TypeScript client retrieves account configuration, state, data, and reports, and surfaces portfolio and cash funds. DegiroAPI fetches portfolio data (with options to filter zero‑size products), cash funds, transactions within a time window, and order history (including open orders). degiroasync’s high‑level API aggregates portfolio totals, product information, and order histories, while the low‑level webapi mirrors the HTTP calls for precision and control.[^6][^8][^11][^12]

Table 5: Account data endpoints by library

| Library | Account data | Cash funds | Portfolio | Transactions | Orders history | Reports |
|---|---|---|---|---|---|---|
| icastillejogomez/degiro‑api (TypeScript) | Yes | Yes | Yes | Not specified | Yes | Yes[^6] |
| DegiroAPI (Python) | Not specified | Yes | Yes | Yes | Yes (max 90 days) | Not specified[^8] |
| degiroasync (Python) | Client info | Not explicitly stated | Yes (incl. totals) | Not explicitly stated | Yes | Not specified[^11][^12] |

## Technical Constraints and Robustness

The most consequential constraint for implementers is CORS: DEGIRO’s web application blocks browser‑originated requests to its internal endpoints, forcing a backend‑only architecture. This avoids preflight failures and enables session reuse without browser sandbox restrictions.[^4][^5][^6]

Rate limiting and throttling are widely reported challenges. Libraries do not document formal thresholds, and behavior may change without notice. Community guidance emphasizes conservative request pacing, back‑off on errors, and avoidance of aggressive polling. Header sensitivity has been observed anecdotally; one Python client failure was resolved by modifying the User‑Agent, reinforcing the need to replicate expected browser headers faithfully.[^10]

Stability is the dominant long‑term risk. Some libraries are archived; others show sparse recent commits. Even active projects can break when the platform updates its authentication flows or endpoint schemas. This reality underscores the importance of versioning, monitoring, and testing, and it cautions against embedding DEGIRO interactions in mission‑critical workflows without contingency plans.

## Legal and Compliance Considerations

Using unofficial clients necessarily implicates DEGIRO’s Terms and Conditions and Client Agreement. The client agreement governs usage of investment services and sets expectations regarding platform access, communications, and compliance. While these documents are broad and not tailored to API usage per se, they establish that platform access should occur through official channels and that automated scraping or scripting could be considered non‑compliant. Organizations should seek legal counsel to interpret obligations in their jurisdiction and operational context.[^25][^26][^27]

DEGIRO’s privacy and cookie statement and exchange rules further shape data handling obligations and trading conduct. Teams must consider whether their intended use of data and order automation aligns with these constraints, especially if distributing portfolios or trading in jurisdictions with stricter oversight.[^28][^29]

### Risk Assessment and Mitigation

Three categories of risk dominate: service stability, account impact, and compliance. Service stability risk stems from unannounced changes that break integrations. Account impact risk arises if DEGIRO flags automated access as suspicious. Compliance risk involves potential breaches of terms or exchange rules.

Mitigations include strict backend‑only deployment, minimal session footprints, careful logging and monitoring, conservative polling and back‑off strategies, and fallback operational procedures. Open‑source maintainers caution that unofficial APIs can change at any time and may cause losses; implementers should heed such warnings and design for resilience.[^6]

Table 6: Risk mapping and mitigations

| Risk | Description | Impact | Mitigations | Evidence |
|---|---|---|---|---|
| Service breakage | Endpoint changes invalidate client calls | Trading interruption; re‑development cost | Use actively maintained libs; test suites; feature flags; quick rollback | Archived and aged repos; library warnings[^4][^6] |
| Account impact | Automated access flagged as suspicious | Access restrictions; account review | Backend‑only; low request rates; avoid burst patterns | Community reports; docs warnings[^6][^10] |
| Compliance | Potential breach of terms or exchange rules | Regulatory scrutiny; reputational risk | Legal review; data minimization; manual fallbacks | Terms and Client Agreement[^25][^26][^27] |

## Community Ecosystem and Learning Resources

The unofficial ecosystem is anchored by GitHub repositories and documentation sites, supplemented by Reddit threads where users share troubleshooting tips and report outages. The “degiro‑api” topic page aggregates repositories and provides a sense of the ecosystem’s breadth and activity.[^7] TypeScript documentation offers endpoint references and examples; degiroasync provides a structured quickstart and API overview.[^6][^12]

Table 7: Community repositories and resources

| Resource | Purpose | Language | Last known activity (approx.) | Notes |
|---|---|---|---|---|
| icastillejogomez/degiro‑api + docs | Full‑featured client; typed endpoints | TypeScript | Active through 2022; release 1.0.4 (Dec 2021) | Backend‑only; order automation[^5][^6] |
| DegiroAPI (Python) | Broad data + orders | Python | Last commit 2020; release 0.9.5 (Jun 2020) | Header sensitivity; outages reported[^8][^10] |
| degiroasync (Python) + docs | Async two‑layer API | Python | Docs 2022–2023; commits through Dec 2023 | High‑level lacks confirm_order[^11][^12] |
| degiro_portfolio_analytics | Portfolio dashboard | Python | 2022 | Connects to account for analytics[^19] |
| degiro‑portfolio‑rebalancer | Rebalancing tool | Python | 2023 | Asset allocation maintenance[^20] |
| degiro‑lambda‑bot | Scheduled buying bot | TypeScript | 2024 | Serverless automation[^21] |
| screener | Export securities list | Python | 2025 | Data extraction to Pandas df[^23] |

Table 8: Learning resources and support channels

| Resource | Coverage | Utility |
|---|---|---|
| TypeScript client documentation | Endpoints, typed objects, examples | Quick lookup for API surface and deployment constraints[^6] |
| degiroasync docs | Quickstart, API vs webapi distinction | Guides for async usage and order confirmation strategy[^12] |
| GitHub topics page | Ecosystem overview | Discovery of additional repos and tools[^7] |
| Reddit thread (unofficial API not working) | User-Agent workaround | Practical troubleshooting narrative[^10] |

## Implementation Guidance and Decision Framework

Selecting a library begins with matching language and deployment constraints to project needs. Teams building Node.js backends benefit from the typed interfaces and comprehensive endpoints of the TypeScript client, with clear guidance to avoid browser deployment due to CORS. Python teams can choose between synchronous libraries like DegiroAPI and asynchronous options like degiroasync, weighing ease of use against performance and control. degiro‑connector offers a packaged entry point with explicit 2FA support.[^5][^8][^11][^13]

Authentication setup should follow a secure pattern: store credentials and OTP secrets in a secure vault, implement session reuse with tokens such as JSESSIONID, and adopt periodic re‑authentication to manage expiry. Request pacing must be conservative—cache metadata, stagger price refreshes, and back‑off on errors. Order flows require validation, including pre‑trade checks and confirmation routines. If the high‑level API lacks confirm_order, use the low‑level layer to finalize orders and handle responses robustly.[^6][^11]

Operational resilience hinges on monitoring for schema changes, alerting on error spikes, and maintaining circuit breakers to pause automation when anomalies occur. Maintain rollback paths and manual alternatives so trading is not interrupted by library breakage.

Table 9: Library selection matrix

| Library | Language / Runtime | 2FA support | Data coverage | Orders coverage | Docs depth | Maintenance | Recommended use cases |
|---|---|---|---|---|---|---|---|
| icastillejogomez/degiro‑api | Node.js (TypeScript) | Yes | Portfolio, funds, products, news | Create, execute, delete; historical orders | High | Active through 2022 | Backend order automation; typed integrations[^5][^6] |
| DegiroAPI (Python) | Python | Yes (implicit) | Search, info, real‑time/historical, stock lists | Multiple types; delete orders; time windows | Moderate | Last commit 2020 | Analytics + basic orders; research pipelines[^8] |
| degiroasync (Python) | Python | Yes | High‑level and low‑level data | Check/confirm via webapi; orders/history | High | Docs updated 2022–2023 | Async workflows; flexible control with webapi[^11][^12] |
| degiro‑connector (Python) | Python | Yes | Connection; real‑time tag | Not specified | Moderate | PyPI package | Simple integration with explicit 2FA[^13] |
| python‑degiro (PyPI) | Python | Unclear | Claims real‑time/historical | Not specified | Low | PyPI listing | Experimental usage; feature validation needed[^14] |

Table 10: Operational do’s and don’ts

| Practice | Do | Don’t |
|---|---|---|
| Deployment | Run backend‑only; isolate sessions | Attempt browser usage (CORS will block)[^4][^5][^6] |
| Request pacing | Cache, stagger, back‑off; retry judiciously | Burst polling; ignore rate‑limit signals[^8][^10] |
| Order handling | Validate parameters; pre‑trade checks; confirm via webapi if needed | Skip confirmations; assume stability[^11] |
| Secrets | Use vaulted storage; rotate credentials | Store in code or environment without protection |
| Monitoring | Alert on error rates and schema drift | Operate without visibility; ignore warnings |
| Fallbacks | Maintain manual workflows and circuit breakers | Depend solely on automation |

### Security Hardening

Security should be treated as a first‑class concern. Credentials and OTP secrets must be stored in a secure vault, not embedded in code or left in plaintext environment variables. Sessions should be scoped to least privilege and rotated periodically. Implement anomaly detection around authentication and order submission, and apply feature flags or kill switches to halt automation when the platform behavior changes unexpectedly.[^6]

## Appendix: DEGIRO Helpdesk and Trading Concepts

While unofficial clients focus on API‑like features, the official helpdesk describes order placement through the platform’s user interface—buy/sell buttons, order type selection, duration settings, and chart interactions. Understanding these concepts helps validate programmatic implementations against the documented UI workflow.[^30][^31][^32]

DEGIRO’s privacy statement and exchange rules frame broader obligations related to data handling and trading conduct. Teams should review these documents alongside the Client Agreement and Terms and Conditions when assessing compliance and risk.[^28][^29][^25][^27]

## Acknowledged Information Gaps

- DEGIRO’s internal API endpoints and headers are undocumented; behavior may change without notice.
- Official rate‑limits and throttling policies are not publicly specified.
- Current functionality across all libraries is uncertain given platform changes; some repositories are archived and others may have aged.
- Legal interpretations of terms depend on jurisdiction and specific use cases; consult counsel.
- Full order parameter schemas and validation rules may require deeper reverse‑engineering or fresh testing.

## References

[^1]: Does DEGIRO offer an API? https://www.degiro.com/uk/helpdesk/trading-platform/does-degiro-offer-api  
[^2]: Is There a DEGIRO Mobile App Available? https://www.degiro.com/uk/helpdesk/trading-platform/there-degiro-mobile-app-available  
[^3]: Can I place orders and track my portfolio on the DEGIRO mobile app? https://www.degiro.com/uk/helpdesk/trading-platform/can-i-place-orders-and-track-my-portfolio-degiro-mobile-app  
[^4]: pladaria/degiro: DEGIRO (unoficial) API - GitHub. https://github.com/pladaria/degiro  
[^5]: icastillejogomez/degiro-api - GitHub. https://github.com/icastillejogomez/degiro-api  
[^6]: degiro-api Documentation. https://icastillejogomez.github.io/degiro-api/  
[^7]: GitHub Topics: degiro-api. https://github.com/topics/degiro-api  
[^8]: lolokraus/DegiroAPI: An unofficial API for the trading platform Degiro - GitHub. https://github.com/lolokraus/DegiroAPI  
[^9]: python-degiro - PyPI. https://pypi.org/project/python-degiro/  
[^10]: (Unoffical) DegiroAPI not working anymore? - Reddit. https://www.reddit.com/r/DEGIRO/comments/ze8xz7/unoffical_degiroapi_not_working_anymore/  
[^11]: OhMajesticLama/degiroasync - GitHub. https://github.com/OhMajesticLama/degiroasync  
[^12]: degiroasync documentation - GitHub Pages. https://ohmajesticlama.github.io/degiroasync/index.html  
[^13]: degiro-connector - PyPI. https://pypi.org/project/degiro-connector/  
[^14]: python-degiro - PyPI. https://pypi.org/project/python-degiro/  
[^15]: degiro_api - Dart API docs - Pub.dev. https://pub.dev/documentation/degiro_api/latest/  
[^16]: Chavithra/degiro-connector - GitHub. https://github.com/Chavithra/degiro-connector  
[^17]: FKatenbrink/degiro-api - GitHub. https://github.com/FKatenbrink/degiro-api  
[^18]: Jorricks/python-degiro - GitHub. https://github.com/Jorricks/python-degiro  
[^19]: lucalaringe/degiro_portfolio_analytics - GitHub. https://github.com/lucalaringe/degiro_portfolio_analytics  
[^20]: marcopus/degiro-portfolio-rebalancer - GitHub. https://github.com/marcopus/degiro-portfolio-rebalancer  
[^21]: filipsuk/degiro-lambda-bot - GitHub. https://github.com/filipsuk/degiro-lambda-bot  
[^22]: degiro-patform · GitHub Topics. https://github.com/topics/degiro-patform  
[^23]: facoptere/screener - GitHub. https://github.com/facoptere/screener  
[^24]: DEGIRO | Online Brokerage - Reddit. https://www.reddit.com/r/DEGIRO/  
[^25]: Client Agreement - Investment Services Terms and Conditions (PDF). https://www.degiro.com/uk/data/pdf/uk/Client_Agreement_Investment_Services_Terms_and_Conditions.pdf  
[^26]: Client Agreement - Investment Services Terms and Conditions (PDF, redirect). https://www.degiro.com/uk/helpdesk/sites/co.uk/files/2021-07/Client_Agreement_Investment_Services_Terms_and_Conditions.pdf?redirect=true  
[^27]: Terms and Conditions - Degiro.com. https://www.degiro.com/uk/helpdesk/legal-documents/terms-and-conditions  
[^28]: DEGIRO's Privacy & Cookie Statement. https://www.degiro.com/uk/privacy-and-cookies  
[^29]: Exchange Rules and Regulations - Degiro.com. https://www.degiro.com/uk/helpdesk/documents/trading-and-exchanges/exchange-rules-and-regulations  
[^30]: How do I place an order? - Degiro.com. https://www.degiro.com/uk/helpdesk/orders/placing-order/how-do-i-place-order  
[^31]: Orders - Degiro.com. https://www.degiro.com/uk/helpdesk/orders  
[^32]: Trading Platform | Degiro. https://www.degiro.com/uk/helpdesk/trading-platform