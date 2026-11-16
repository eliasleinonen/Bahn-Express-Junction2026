# Junction 2026 — Spread Portfolio Dispatcher

Our team's entry for **Spread**, Bahn's real-time car transport dispatch optimization challenge at Junction 2026.

The system acts as an autonomous car transport dispatcher across Finland for a simulated one-week operational cycle. It dynamically prices delivery quotes for car dealerships and contracts freelance drivers to move the vehicles. The competition objective is to **maximize cumulative gross margin** while maintaining a strict **>= 90% on-time delivery rate**.

---

## System Architecture & Key Components

The final system is an integrated, event-driven portfolio dispatcher that models route capacity, market demand, and driver logistics holistically rather than treating orders as independent isolated cars.

```
                      ┌─────────────────────────────────────────┐
                      │             Spread API / World          │
                      └────────────────────┬────────────────────┘
                                           │  Events & Snapshot
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                               Portfolio Dispatch Engine                              │
│                                                                                      │
│   ┌────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────┐   │
│   │   Multilingual Note    │  │  Adaptive Quoting &     │  │  Dynamic Fee Ladder │   │
│   │    Constraint Parser   │  │ Bayesian Demand Model   │  │   & Late Recovery   │   │
│   │ (FI / SV / EN Rules)   │  │  (Win/Loss Residuals)   │  │   (8 Waking Steps)  │   │
│   └───────────┬────────────┘  └────────────┬────────────┘  └──────────┬──────────┘   │
│               │                            │                          │              │
│               ▼                            ▼                          ▼              │
│   ┌──────────────────────────────────────────────────────────────────────────────┐   │
│   │                 Route Chaining & Portfolio Resource Allocator                │   │
│   │        - Spatiotemporal Capacity Tracking (No Hard Global Car Cap)           │   │
│   │        - Feasibility & Economics of Multi-Car Chaining (Pair Limits)         │   │
│   │        - State Reconstruction & Graceful Stream Reconnection                 │   │
│   └──────────────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

### 1. Portfolio Dispatch Engine (`dispatch.py`, `dispatch_core.py`)
- **Event-Driven State Machine**: Listens to server events (`price_request`, `order_won`, `visit_failed`, `gig_expired`, `deadline_close`) and executes actions within the API rate limits.
- **Dynamic Resource Reservations**: Replaces rigid global car-count caps with local spatiotemporal capacity tracking (pickup locations, delivery deadlines, regional driver roster supply, and weather forecasts).
- **Crash Recovery & Replay**: Capable of rebuilding active reservations, dealer trust metrics, and market priors directly from snapshot history upon process restarts.

### 2. Adaptive Quoting & Bayesian Demand Learning
- **Empirical Market Prior**: Initializes around an empirical flip point (~1.36× list price) and conducts real-time Bayesian updates for dealer, lane, weekday, and delivery speed residuals based solely on observed wins and losses.
- **Selective Decline Pricing**: Automatically quotes 6.0× list price on infeasible deadlines, ambiguous notes, private pickups, or high-risk edge cases to decline without incurring unanswered-request penalties.
- **Marginal Contribution Search**: Evaluates candidate delivery options (`same_day`, `next_day`, `within_3_days`) in 0.01× increments to choose the combination maximizing expected net margin minus expected delay penalties.

### 3. Multilingual Note Constraint Parser (`note_constraints.py`)
- **Deterministic Rule Extraction**: Highly conservative NLP parser covering Finnish, Swedish, and English dealer notes.
- **Actionable Constraints**: Extracts site opening hours, earliest pickup times, delivery deadlines, transit plate requirements, manual gearbox needs, and seasonal tyre constraints (e.g. flagging summer tyres in winter/snow conditions).

### 4. Multi-Order Chaining & Route Economics (`pair_economics.py`)
- **Route Chaining**: Combines compatible single-car orders into multi-car driver gigs to save driver fees.
- **Safety Bounds**: Enforces strict driving-time limits (<= 5 hours), bridge driving limits (<= 60 minutes between dropoff and next pickup), and idle-time limits (<= 60 minutes).
- **Economic Feasibility**: Evaluates joint risk and requires a positive net margin gain before locking in paired routes.

### 5. Dynamic Driver Fee Ladder & Late Recovery
- **Waking-Hour Stepping**: Posts gigs at 0.90× standard reference fees and escalates across 8 daytime steps up to 1.60× as the safe pickup cutoff approaches.
- **Deadline Rescue Escalation**: Automatically activates steep rescue schedules when deadline warnings fire.
- **Late Recovery Ladder**: When an order has already missed its on-time window, the system switches to a low-cost recovery ladder, avoiding wasted speed premiums and preserving total profit.

### 6. Fleet Booking & Offline Verification (`fleet_booking_model.py`, `decision_dataset.py`, `offline_data_contract.py`)
- **Booking Prior**: Calibrates gig acceptance probabilities against warning slack and gig attributes.
- **Append-Only Decision Log**: Cryptographically hashes decision records and subsequent outcome labels to enable reproducible evaluation without future data leakage.

### 7. Local Monitoring & Human Dispatcher UI (`ui/`)
- Includes a lightweight local proxy server (`ui/server.py`) serving a full-screen replay dashboard and manual practice console without exposing API keys to the browser.

---

## Repository Structure

```
.
├── dispatch.py               # Main final portfolio dispatcher
├── dispatch_core.py          # Core runtime loop, event handling, and CLI runner
├── dispatch_v5.py            # Backward-compatibility alias for dispatch.py
├── dispatch_v3_core.py       # Backward-compatibility alias for dispatch_core.py
├── note_constraints.py       # Multilingual dealer note constraint extraction
├── pair_economics.py         # Multi-car gig pairing feasibility & economics
├── fleet_booking_model.py    # Offline driver booking probability models
├── decision_dataset.py       # Append-only decision logging with HMAC/SHA-256 checks
├── offline_data_contract.py  # Shared feature schema and hashing contract
├── market_choice_model.py    # Transparent market demand choice model
├── experiment_registry.py    # Experiment manifest and deterministic split hashing
│
├── test_dispatch.py          # Comprehensive test suite for portfolio dispatcher
├── test_note_constraints.py  # Tests for multilingual note parsing
├── test_pair_economics.py    # Tests for route pairing economics
├── test_shadow_logging.py    # Tests for decision and shadow logging
├── test_decision_dataset.py  # Tests for append-only dataset schema
├── test_offline_data_contract.py # Tests for data contracts and features
├── test_experiment_registry.py   # Tests for experiment registry
│
├── requirements.txt          # Python dependencies (spread, google-genai)
├── .env.example              # Environment variables template
├── docs/                     # Challenge handbook, rules, and API specifications
├── reference/starter.py      # Baseline starter code provided with the kit
├── data/models/              # Serialized fleet and market priors
└── ui/                       # Local web dashboard and human dispatcher UI
```

---

## Setup & Running

### 1. Requirements & Environment
Python 3.9+ is required.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure environment credentials
cp .env.example .env
# Edit .env and insert your SPREAD_KEY
set -a; source .env; set +a
```

### 2. Run Verification Tests
Verify all core modules and offline logic:

```sh
for test in test_*.py; do python "$test"; done
```

### 3. Run the Dispatcher

**Practice World:**
```sh
python dispatch.py --level 8 --seed test-run-1
```

**Official Practice Ranking Run:**
```sh
python dispatch.py --board --level 8
```

**Live Competition Final:**
```sh
python dispatch.py --final
```

### 4. Local Web UI Dashboard
Launch the local visualizer:

```sh
python ui/server.py
```
Then open `http://localhost:8765` in your browser.
