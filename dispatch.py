"""Portfolio dispatcher for Spread.

Makes acceptance and posting decisions as an integrated route portfolio instead of
independent cars behind a global car-count cap. Evaluates option pricing with
online Bayesian market residual updates, dynamic driver fee schedules, conservative
multilingual dealer-note parsing, and multi-order chain planning.
"""
from __future__ import annotations

import math
import os
import json
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Any

import dispatch_core as base
import fleet_booking_model as booking_model
import decision_dataset
import offline_data_contract
import note_constraints


DAY = 24 * 60
HOUR = 60


def sigmoid(value: float) -> float:
    if value >= 35:
        return 1.0
    if value <= -35:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


class PortfolioDesk(base.Desk):
    """A clean-room portfolio and multi-order-chain policy."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.roads: dict[tuple[str, str], tuple[float, int]] = {}
        self.place_coordinates: dict[str, tuple[float, float]] = {}
        try:
            road_map = self.world.client.map()
            self.roads = {
                (road["from"], road["to"]): (float(road["km"]), int(road["minutes"]))
                for road in road_map.get("roads", [])
            }
            self.place_coordinates = {
                place["id"]: (float(place["lat"]), float(place["lon"]))
                for place in road_map.get("places", [])
            }
        except Exception:
            pass

        # A level-8 list price deliberately leaves material room to the buyer.
        # The market prior is a *flip point*, not a price floor: every answer
        # moves it from the only evidence we are allowed to observe, our own
        # wins and losses.
        self.market_log = math.log(float(os.getenv("V5_INITIAL_FLIP", "1.36")))
        self.market_var = 0.10 ** 2
        self.market_width = float(os.getenv("V5_MARKET_WIDTH", "0.06"))
        self.dealer_market: dict[str, float] = defaultdict(float)
        self.lane_market: dict[tuple[str, str], float] = defaultdict(float)
        self.day_market: dict[int, float] = defaultdict(float)
        self.dealer_speed: dict[str, float] = defaultdict(float)
        self.dealer_trust: dict[str, int] = defaultdict(int)
        self.market_observations: dict[tuple[str, Any], int] = defaultdict(int)
        self.market_refreshed_through = -1
        self.quote_book: dict[str, dict[str, Any]] = {}
        self.option_plans: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
        self.note_facts: dict[str, note_constraints.NoteConstraints] = {}

        # Orders wait briefly so the route planner can join compatible cars.
        # A waiting order is still fully owned; it has simply not been posted.
        self.waiting: dict[str, int] = {}
        self.posted_at: dict[str, int] = {}
        self.forecasts: list[dict[str, Any]] = []
        self.reserved_late: dict[str, float] = {}
        # Reservations are probability-weighted while a quote is undecided and
        # become whole-car claims when we own an order.  They make a pickup
        # place/time collision expensive without imposing a global car cap.
        self.resource_reservations: dict[str, dict[str, Any]] = {}
        self.home_supply: Counter[str] = Counter()
        try:
            self.home_supply = Counter(profile["homePlaceId"] for profile in self.world.roster())
        except Exception:
            pass
        self.completed_orders = 0
        self.late_orders = 0
        self.reliability_bias = 0.0
        self.settled_orders: set[str] = set()
        self.pair_failed_orders: set[str] = set()
        self.ends_at: int | None = None
        # Once the on-time pickup has passed, there is no longer a reason to
        # buy speed.  These orders use the deliberately cheap late-delivery
        # recovery schedule when their expired gig is posted again.
        self.late_recovery_orders: set[str] = set()
        self.chain_posts = 0
        self.chain_cars = 0
        self.single_posts = 0
        # These counters are strictly observational.  They tell experiments
        # which gate removed viable-looking speed options without changing the
        # live decision path.
        self.quote_requests = 0
        self.options_seen = 0
        self.options_offered = 0
        self.declined_requests = 0
        self.rejected_options: dict[str, int] = defaultdict(int)

        # Legacy fee telemetry remains available for offline experiments.  The
        # live fee policy below is explicit: a published waking-hour ladder,
        # an on-time rescue, and a separate late-delivery recovery ladder.
        self.fee_scale = float(os.getenv("V5_OPENING_FEE_SCALE", "1.18"))
        self.fee_context_adjustment: dict[tuple[Any, ...], float] = defaultdict(float)
        self.gig_fee_state: dict[str, dict[str, Any]] = {}
        self.gig_views: dict[str, set[str]] = defaultdict(set)
        self.fleet_booking_model: dict[str, Any] | None = None
        self.fleet_predictions: list[float] = []
        self.decision_log_path = os.getenv("V5_DECISION_LOG")
        self.quote_decisions: dict[str, str] = {}
        self.gig_decisions: dict[str, str] = {}
        self.warning_labeled_gigs: set[str] = set()
        self.withdrawn_posts: dict[str, dict[str, Any]] = {}
        if os.getenv("V52_FLEET_MODEL", "0") == "1":
            try:
                with open(os.getenv("V52_FLEET_MODEL_PATH", "data/models/fleet_booking_v1.json")) as handle:
                    self.fleet_booking_model = json.load(handle)
            except (OSError, ValueError, KeyError):
                self.fleet_booking_model = None

    # Event and lifecycle --------------------------------------------------

    def handle(self, event: Any) -> None:
        kind = event["kind"]
        self._refresh_market_uncertainty(event.get("at", self.now))
        if kind == "forecast":
            self.forecasts.append({
                "from": event["from"], "to": event["to"], "placeIds": set(event["placeIds"]),
            })
        elif kind == "order_won":
            self._learn_quote(event["order"]["id"], event["order"]["option"], True)
            self._label_quote(event["order"]["id"], event.get("at", self.now), {"won": True, "option": event["order"]["option"]})
        elif kind == "order_lost":
            self._learn_quote(event["requestId"], None, False)
            self._label_quote(event["requestId"], event.get("at", self.now), {"won": False})
            self.reserved_late.pop(event["requestId"], None)
            self.resource_reservations.pop(event["requestId"], None)
        elif kind == "gig_viewed":
            self.gig_views[event["gigId"]].add(event["driverId"])
        elif kind == "gig_booked":
            self.fee_scale = max(1.02, self.fee_scale - 0.004)
            state = self.gig_fee_state.get(event["gigId"])
            order_ids = () if state is None else state["orderIds"]
            known = [self.orders[order_id] for order_id in order_ids if order_id in self.orders]
            early: bool | None = (all(event["at"] < order["lastPickupAt"] - 3 * HOUR for order in known)
                                 if known and len(known) == len(order_ids) else None)
            if self._v52_fee_model() and early is not None:
                self._learn_gig_fee(event["gigId"], -0.004 if early else 0.012)
            self._label_gig(event["gigId"], event["at"], {"booked": True, "bookedBeforeWarning": early})
        elif kind in ("gig_expired", "gig_dropped", "no_show"):
            if not self._v52_fee_model():
                self.fee_scale = min(1.60, self.fee_scale + 0.040)
        if kind == "gig_dropped":
            self._reopen_after_drop(event["gigId"], event.get("at", self.now))
        elif kind == "deadline_close" and self._v52_fee_model():
            for gig_id in self.gigs_of(event["orderId"]):
                state = self.gig_fee_state.get(gig_id)
                if state is not None and gig_id not in self.booked and not state.get("warningLabeled"):
                    self._learn_gig_fee(gig_id, 0.012)
                    state["warningLabeled"] = True
        if kind == "deadline_close":
            for gig_id in self.gigs_of(event["orderId"]):
                if gig_id not in self.booked and gig_id not in self.warning_labeled_gigs:
                    self._label_gig(gig_id, event["at"], {"reachedWarning": True,
                                                          "bookedBeforeWarning": False,
                                                          "warningAt": event["at"]})
                    self.warning_labeled_gigs.add(gig_id)

        super().handle(event)

        # The core updates deliveredAt while handling the event, so outcome
        # learning must happen afterwards rather than from stale state.
        if kind == "delivered":
            order = self.orders.get(event["orderId"])
            if (order is not None and order["id"] not in self.settled_orders
                    and all(at is not None for at in order["deliveredAt"])):
                predicted = self.reserved_late.pop(order["id"], 0.04)
                self.resource_reservations.pop(order["id"], None)
                late = any(at > order["deadline"] for at in order["deliveredAt"])
                self.completed_orders += 1
                self.late_orders += int(late)
                self.reliability_bias = self._bound(
                    self.reliability_bias + 0.012 * (float(late) - predicted)
                )
                dealer = self.requests.get(order["id"], {}).get("dealerId")
                if dealer is not None:
                    self.dealer_trust[dealer] = (
                        self.dealer_trust[dealer] // 2 if late
                        else min(25, self.dealer_trust[dealer] + 1)
                    )
                self.settled_orders.add(order["id"])

        if kind == "gig_expired":
            self.posts.pop(event["gigId"], None)
        if kind == "order_cancelled":
            self.waiting.pop(event["orderId"], None)
            self.reserved_late.pop(event["orderId"], None)
            self.resource_reservations.pop(event["orderId"], None)
            self._recover_cancelled_chain(event)
        self._transition_late_recovery()
        self._take_second_look()
        self._plan_waiting()

    def win(self, order: Any) -> None:
        """Own the order now; defer posting only long enough to plan a route."""
        self.pending_quotes.pop(order["id"], None)
        if order["id"] in self.requests and order["id"] not in self.orders:
            self.orders[order["id"]] = order
            request = self.requests[order["id"]]
            self.reserved_late.setdefault(
                order["id"], self._late_probability(request, self._won_option(order["id"]), 0.0)
            )
            self._reserve_resources(order["id"], request, 1.0, "owned")
            self.waiting[order["id"]] = self.now

    def repost(self, gig_id: str) -> None:
        """Split an expired chain or advance a daily opening-hours window safely."""
        action = self.posts.get(gig_id)
        if action is None or not any(
            self.leg_gigs.get(stop["orderId"], {}).get(stop["leg"]) == gig_id for stop in action["stops"]
        ):
            return
        members = self._action_orders(action)
        if len(members) <= 1:
            if members:
                order = self.orders.get(members[0])
                facts = self.note_facts.get(members[0], note_constraints.NoteConstraints())
                if order is not None and (facts.pickup_hours is not None or facts.delivery_hours is not None):
                    if self.now > order["lastPickupAt"]:
                        self.late_recovery_orders.add(order["id"])
                        self.urgent.discard(order["id"])
                    legs = [stop["leg"] for stop in action["stops"] if stop["kind"] == "pickup"]
                    stops = self._stops_with_facts(order, legs, base.Retry(), facts)
                    if any("window" in stop for stop in stops):
                        for stop in action["stops"]:
                            self.leg_gigs.get(stop["orderId"], {}).pop(stop["leg"], None)
                        self.outbox.append({
                            **action,
                            "stops": stops,
                            "schedule": self.fees_for(order, len(legs)),
                        })
                        return
            self.late_recovery_orders.update(members)
            super().repost(gig_id)
            return
        for stop in action["stops"]:
            self.leg_gigs.get(stop["orderId"], {}).pop(stop["leg"], None)
        self.posts.pop(gig_id, None)
        for order_id in members:
            order = self.orders.get(order_id)
            if order and base.undelivered(order):
                self.waiting[order_id] = self.now

    def change(self, order: Any) -> None:
        """Withdraw an open shared gig before its route assumptions change."""
        if order["id"] not in self.orders:
            return
        self.orders[order["id"]] = order
        if self._unplan_open_gigs(order["id"]) and base.undelivered(order):
            self.waiting[order["id"]] = self.now

    def raise_fees(self, order_id: str) -> None:
        """Raise every open gig carrying this order, including a shared chain."""
        order = self.orders.get(order_id)
        if order is None:
            return
        self.urgent.add(order_id)
        self._send_queued_posts_for(order_id)
        for gig_id in self.gigs_of(order_id):
            if gig_id in self.booked:
                continue
            action = self.posts.get(gig_id)
            if action is None:
                continue
            members = self._action_orders(action)
            if len(members) > 1:
                schedule = self._bundle_schedule(members, urgent=True)
            else:
                schedule = self.fees_for(order, base.pickups(action))
            action["schedule"] = schedule
            self.outbox.append({"type": "reprice", "gigId": gig_id, "schedule": schedule})

    def read_results(self, actions: list[Any], results: list[Any]) -> None:
        records = [self._log_decision(action) for action in actions]
        super().read_results(actions, results)
        for action, result, record_id in zip(actions, results, records):
            if action["type"] == "withdraw":
                old_action = self.withdrawn_posts.pop(action["gigId"], None)
                if result.get("ok") is not True and old_action is not None:
                    self.carry(old_action, action["gigId"])
            elif action["type"] == "post":
                if result.get("ok") is True:
                    self.posted_at[result["id"]] = self.now
                    members = self._action_orders(action)
                    self.gig_fee_state[result["id"]] = {
                        "contexts": tuple(self._fee_context(self.orders[order_id]) for order_id in members
                                          if order_id in self.orders),
                        "openingFee": action["schedule"][0]["feeCents"], "postedAt": self.now,
                        "orderIds": tuple(members), "repriced": False,
                    }
                    if record_id:
                        self.gig_decisions[result["id"]] = record_id
                else:
                    members = self._action_orders(action)
                    if len(members) > 1:
                        self.pair_failed_orders.update(members)
                    for order_id in members:
                        order = self.orders.get(order_id)
                        if order is None or not base.undelivered(order) or self.gigs_of(order_id):
                            continue
                        if len(members) > 1:
                            self.waiting[order_id] = self.now
                        elif any("window" in stop for stop in action.get("stops", [])):
                            self.late_recovery_orders.add(order_id)
                            self.urgent.discard(order_id)
                            self.waiting[order_id] = self.now
            elif action["type"] == "quote":
                if result.get("ok") is True and action["requestId"] in self.quote_book:
                    self._persist_quote(action["requestId"], self.quote_book[action["requestId"]])
                if record_id:
                    self.quote_decisions[action["requestId"]] = record_id

    def _quote_journal_path(self) -> Path | None:
        world_id = getattr(self.world, "id", None)
        if not world_id or os.getenv("V5_PERSIST_QUOTES", "1") != "1":
            return None
        return Path("data/runtime") / f"quotes_{world_id}.jsonl"

    def _persist_quote(self, request_id: str, record: dict[str, Any]) -> None:
        path = self._quote_journal_path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"requestId": request_id, "record": record}, separators=(",", ":")) + "\n")
                handle.flush()
        except OSError:
            pass

    def _load_persisted_quotes(self) -> dict[str, dict[str, Any]]:
        path = self._quote_journal_path()
        if path is None or not path.is_file():
            return {}
        loaded: dict[str, dict[str, Any]] = {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    record = item["record"]
                    if isinstance(record.get("lane"), list):
                        record["lane"] = tuple(record["lane"])
                    loaded[item["requestId"]] = record
        except (OSError, ValueError, KeyError, TypeError):
            return {}
        return loaded

    def _drain_outbox(self) -> list[Any]:
        """Drop stale reprice actions for gigs booked or expired in the same batch."""
        raw, self.outbox = self.outbox, []
        return [
            action for action in raw
            if action["type"] != "reprice"
            or (action["gigId"] in self.posts and action["gigId"] not in self.booked)
        ]

    def send(self) -> None:
        """Use the kit's token-aware sender, prioritizing quotes/posts over offers."""
        actions = self._drain_outbox()
        if not actions:
            return
        # Keep withdrawal/post relative order via Python's stable sort while
        # ensuring quotes go first and low-yield direct offers go last.
        actions = sorted(actions, key=lambda a: 0 if a["type"] == "quote" else (2 if a["type"] == "offer" else 1))
        self.read_results(actions, self.world.act(actions))

    def rejoin(self, snapshot: Any, history: list[Any]) -> None:
        """Restore full V5 priors BEFORE answering missed requests or posting."""
        self.ends_at = snapshot.get("endsAt")
        posted: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.booked = {gig["id"] for gig in snapshot["gigs"] if gig["status"] in ("booked", "working", "done")}
        self.leg_gigs = {}
        self.posts = {}
        for gig in snapshot["gigs"]:
            for stop in gig["stops"]:
                posted[stop["orderId"]].append(gig)
            if gig["status"] in ("open", "booked", "working"):
                self.carry({"type": "post", "stops": gig["stops"], "audience": gig["audience"], "schedule": gig["schedule"]}, gig["id"])
        self.requests = {event["request"]["id"]: event["request"] for event in history if event["kind"] == "price_request"}
        self.urgent = {event["orderId"] for event in history if event["kind"] == "deadline_close"}
        self.orders = {o["id"]: o for o in snapshot["orders"] if o["id"] in self.requests and o["status"] != "cancelled"}
        self.now = snapshot["now"]
        self.note_facts = {
            rid: note_constraints.parse_note(req.get("note", {}).get("text") if req.get("note") else None, req["arrivedAt"])
            for rid, req in self.requests.items()
        }
        for req in snapshot["openRequests"]:
            self.requests[req["id"]] = req
            self.note_facts[req["id"]] = note_constraints.parse_note(
                req.get("note", {}).get("text") if req.get("note") else None, req["arrivedAt"]
            )
        self.late_recovery_orders = {
            order["id"] for order in self.orders.values()
            if base.undelivered(order) and snapshot["now"] > order["lastPickupAt"]
        }
        self.urgent.difference_update(self.late_recovery_orders)
        self.posted_at = {gig["id"]: gig["postedAt"] for gig in snapshot["gigs"]}
        self.gig_fee_state = {
            gig["id"]: {
                "contexts": tuple(), "openingFee": gig["schedule"][0]["feeCents"],
                "postedAt": gig["postedAt"], "orderIds": tuple(
                    dict.fromkeys(stop["orderId"] for stop in gig["stops"])
                ), "repriced": False,
            }
            for gig in snapshot["gigs"] if gig["schedule"]
        }
        self._restore_strategy_state(snapshot, history)
        self._transition_late_recovery()
        for order in self.orders.values():
            if order["status"] == "waiting" and not self.gigs_of(order["id"]):
                self.post(order, base.undelivered(order))
        for event in history:
            order = self.orders.get(event.get("orderId", ""))
            if order is None:
                continue
            if event["kind"] == "visit_failed" and order["deliveredAt"][event["leg"]] is None:
                later = [g for g in posted.get(order["id"], []) if g["postedAt"] >= event["at"] and g["id"] != event["gigId"]]
                if not any(stop["leg"] == event["leg"] for gig in later for stop in gig["stops"]):
                    self.retry(order["id"], event["leg"], event["reason"])
        for request in snapshot["openRequests"]:
            self.quote(request)

    def _restore_strategy_state(self, snapshot: dict[str, Any], history: list[dict[str, Any]]) -> None:
        """Recreate safe priors from the server snapshot and immutable event log.

        Quotes themselves are not events, so their old prices cannot always be
        recovered after a hard process kill.  We replay the deterministic
        strategy against historical requests to rebuild its market posterior;
        current orders and gigs always come from the authoritative snapshot.
        """
        final_now = snapshot["now"]
        current_orders = self.orders
        current_requests = dict(self.requests)
        saved_facts = dict(self.note_facts)
        self.market_log = math.log(float(os.getenv("V5_INITIAL_FLIP", "1.36")))
        self.market_var = 0.10 ** 2
        self.dealer_market = defaultdict(float)
        self.lane_market = defaultdict(float)
        self.day_market = defaultdict(float)
        self.dealer_speed = defaultdict(float)
        self.dealer_trust = defaultdict(int)
        self.market_observations = defaultdict(int)
        self.market_refreshed_through = -1
        self.forecasts = []
        self.quote_book = {}
        self.pending_quotes = {}
        self.resource_reservations = {}
        self.reserved_late = {}
        self.option_plans = defaultdict(dict)
        self.note_facts = saved_facts
        self.outbox = []

        # Replaying demand must not let today's open fleet affect Monday's
        # historical admission decisions.  It is used only for quote learning.
        persisted_quotes = self._load_persisted_quotes()
        unquoted_losses: set[str] = set()
        if not persisted_quotes:
            # Stateless recovery fallback (no local quote journal on disk):
            # never count order_lost runs of >=10 consecutive losses with zero wins
            # or gig posts, which indicate the process was offline/not started yet.
            posted_times = sorted(
                gig["postedAt"] for gig in snapshot.get("gigs", []) if gig.get("postedAt") is not None
            )
            streak: list[str] = []
            last_activity_at = -1
            post_idx = 0
            for event in history:
                ev_at = event.get("at", 0)
                while post_idx < len(posted_times) and posted_times[post_idx] <= ev_at:
                    if posted_times[post_idx] > last_activity_at:
                        if len(streak) >= 10:
                            unquoted_losses.update(streak)
                        streak.clear()
                        last_activity_at = posted_times[post_idx]
                    post_idx += 1
                ekind = event["kind"]
                if ekind == "order_won":
                    if len(streak) >= 10:
                        unquoted_losses.update(streak)
                    streak.clear()
                    last_activity_at = max(last_activity_at, ev_at)
                elif ekind == "order_lost":
                    streak.append(event["requestId"])
            if len(streak) >= 10:
                unquoted_losses.update(streak)

        self.orders = {}
        for event in history:
            self.now = event.get("at", self.now)
            self._refresh_market_uncertainty(self.now)
            kind = event["kind"]
            if kind == "forecast":
                self.forecasts.append({
                    "from": event["from"], "to": event["to"], "placeIds": set(event["placeIds"]),
                })
            elif kind == "price_request":
                req = event["request"]
                if persisted_quotes:
                    if req["id"] in persisted_quotes:
                        self.quote_book[req["id"]] = dict(persisted_quotes[req["id"]])
                elif req["id"] not in unquoted_losses:
                    self.quote(req)
            elif kind == "order_won":
                order = event["order"]
                if order["id"] not in self.quote_book and order["id"] in self.requests:
                    self.quote(self.requests[order["id"]])
                self._learn_quote(order["id"], order["option"], True)
                self.reserved_late.pop(order["id"], None)
                self.resource_reservations.pop(order["id"], None)
            elif kind == "order_lost":
                rid = event["requestId"]
                if rid in unquoted_losses:
                    self.quote_book.pop(rid, None)
                else:
                    self._learn_quote(rid, None, False)
                self.reserved_late.pop(rid, None)
                self.resource_reservations.pop(rid, None)
        self.outbox = []
        self.withdrawn_posts = {}
        self.pending_quotes = {}
        self.quote_book = {}
        self.orders = current_orders
        self.requests.update(current_requests)
        self.note_facts = saved_facts
        self.now = final_now
        self._refresh_market_uncertainty(self.now)

        # Rebuild observable dealer trust and the active reliability/resource
        # reservations from the same event history and current snapshot.
        won: dict[str, dict[str, Any]] = {}
        delivered: dict[str, set[int]] = defaultdict(set)
        settled: set[str] = set()
        self.completed_orders = self.late_orders = 0
        self.dealer_trust = defaultdict(int)
        for event in history:
            if event["kind"] == "order_won":
                won[event["order"]["id"]] = event["order"]
            elif event["kind"] == "delivered":
                order = won.get(event["orderId"])
                if order is None:
                    continue
                delivered[order["id"]].add(event["leg"])
                if order["id"] in settled or len(delivered[order["id"]]) < len(order["legs"]):
                    continue
                late = event["at"] > order["deadline"]
                request = self.requests.get(order["id"])
                if request is not None:
                    dealer = request["dealerId"]
                    self.dealer_trust[dealer] = self.dealer_trust[dealer] // 2 if late else min(
                        25, self.dealer_trust[dealer] + 1
                    )
                self.completed_orders += 1
                self.late_orders += int(late)
                settled.add(order["id"])
        self.settled_orders = settled
        for order_id, order in self.orders.items():
            request = self.requests.get(order_id)
            if request is None or not base.undelivered(order):
                continue
            option = self._won_option(order_id)
            self.reserved_late[order_id] = self._late_probability(request, option, 0.0)
            self._reserve_resources(order_id, request, 1.0, "owned")

    def _log_decision(self, action: dict[str, Any]) -> str | None:
        if not self.decision_log_path or action["type"] not in {"quote", "post", "reprice", "withdraw"}:
            return None
        snapshot = {"openGigs": sum(gig_id not in self.booked for gig_id in self.posts),
                    "waiting": len(self.waiting), "ownedOrders": len(self.orders),
                    "lateBudget": sum(self.reserved_late.values()), "now": self.now}
        if action["type"] == "quote":
            snapshot["request"] = self.requests.get(action["requestId"], {})
        else:
            order_ids = self._action_orders(action) if action["type"] == "post" else []
            snapshot["orders"] = {order_id: self.orders[order_id] for order_id in order_ids
                                  if order_id in self.orders}
            snapshot["requests"] = {order_id: self.requests[order_id] for order_id in order_ids
                                    if order_id in self.requests}
        record = decision_dataset.make_decision_record(kind=action["type"], world_id=self.world.id,
            at=self.now, snapshot=snapshot, action=action,
            run={"policy": "dispatch_v5", "v5": {key: value for key, value in os.environ.items()
                 if key.startswith("V5_") or key.startswith("V52_")}})
        decision_dataset.append_record(self.decision_log_path, record)
        return record["recordId"]

    def _label_quote(self, request_id: str, at: int, labels: dict[str, Any]) -> None:
        decision_id = self.quote_decisions.pop(request_id, None)
        if decision_id and self.decision_log_path:
            decision_dataset.append_record(self.decision_log_path, decision_dataset.make_label_record(
                decision_id=decision_id, world_id=self.world.id, at=at, labels=labels))

    def _label_gig(self, gig_id: str, at: int, labels: dict[str, Any]) -> None:
        decision_id = self.gig_decisions.get(gig_id)
        if decision_id and self.decision_log_path:
            decision_dataset.append_record(self.decision_log_path, decision_dataset.make_label_record(
                decision_id=decision_id, world_id=self.world.id, at=at, labels=labels))

    def _fee_context(self, order: dict[str, Any]) -> tuple[Any, ...]:
        request = self.requests[order["id"]]
        leg = order["legs"][0]
        route = self._road(leg)[1]
        slack = max(0, order["lastPickupAt"] - self.now)
        return (leg["pickup"]["placeId"], self.now // (6 * HOUR), min(3, route // 120),
                min(3, slack // (6 * HOUR)), int(self._route_risk(request, self._won_option(order["id"])) > 0.6))

    def _context_fee_factor(self, order: dict[str, Any]) -> float:
        return math.exp(self.fee_context_adjustment[self._fee_context(order)]) if self._v52_fee_model() else 1.0

    @staticmethod
    def _v52_fee_model() -> bool:
        return os.getenv("V52_FEE_MODEL", "0") == "1"

    def _learn_gig_fee(self, gig_id: str, change: float) -> None:
        state = self.gig_fee_state.get(gig_id)
        if state is None:
            return
        for context in state["contexts"]:
            self.fee_context_adjustment[context] = max(-0.10, min(0.20,
                self.fee_context_adjustment[context] + change / max(1, len(state["contexts"]))))

    def _recover_cancelled_chain(self, event: dict[str, Any]) -> None:
        """Only repost other cars when the world withdrew their shared gig."""
        cancelled = event["orderId"]
        for record in event.get("gigs", []):
            if record["outcome"] != "withdrawn":
                continue
            action = self.posts.pop(record["gigId"], None)
            if action is None:
                continue
            for stop in action["stops"]:
                self.leg_gigs.get(stop["orderId"], {}).pop(stop["leg"], None)
            for order_id in self._action_orders(action):
                if order_id != cancelled and order_id in self.orders:
                    self.waiting[order_id] = self.now

    def _unplan_open_gigs(self, order_id: str) -> bool:
        self._send_queued_posts_for(order_id)
        gig_ids = self.gigs_of(order_id)
        if any(gig_id in self.booked for gig_id in gig_ids):
            return False
        recovered: set[str] = set()
        for gig_id in gig_ids:
            action = self.posts.get(gig_id)
            if action is None:
                continue
            recovered.update(self._action_orders(action))
            self.outbox.append({"type": "withdraw", "gigId": gig_id})
            self.withdrawn_posts[gig_id] = action
            self.posts.pop(gig_id, None)
            for stop in action["stops"]:
                self.leg_gigs.get(stop["orderId"], {}).pop(stop["leg"], None)
        for recovered_id in recovered:
            if recovered_id in self.orders and base.undelivered(self.orders[recovered_id]):
                self.waiting[recovered_id] = self.now
        return True

    def _send_queued_posts_for(self, order_id: str) -> None:
        if any(
            action["type"] == "post" and order_id in self._action_orders(action)
            for action in self.outbox
        ):
            self.send()

    @staticmethod
    def _action_orders(action: dict[str, Any]) -> list[str]:
        return list(dict.fromkeys(stop["orderId"] for stop in action["stops"]))

    # Quote portfolio ------------------------------------------------------

    def quote(self, request: Any) -> None:
        self.requests[request["id"]] = request
        self._refresh_market_uncertainty(self.now)
        self.quote_requests += 1
        self.options_seen += len(request["options"])
        self.pending_quotes = {
            request_id: item for request_id, item in self.pending_quotes.items() if item[1] >= self.now
        }

        facts = note_constraints.parse_note(
            request.get("note", {}).get("text") if request.get("note") else None,
            request["arrivedAt"],
        )
        self.note_facts[request["id"]] = facts
        prices: dict[str, int] = {}
        records: dict[str, dict[str, float]] = {}
        pressure = self._portfolio_pressure(request)
        terms: list[dict[str, Any]] = []
        offered_late: list[float] = []
        request_reason = self._request_decline_reason(request, facts)
        if request_reason is not None:
            self.rejected_options[request_reason] += len(request["options"])
        else:
            for option in request["options"]:
                plan, reason = self._on_time_plan(request, option, facts)
                if plan is None:
                    self.rejected_options[reason or "infeasible"] += 1
                    continue
                self.option_plans[request["id"]][option["kind"]] = plan
                risk = self._route_risk(request, option)
                candidate_late = self._late_probability(request, option, pressure, plan)
                # A dealer chooses one speed.  Admit and reserve each option
                # on its own risk, so a tight same-day deadline cannot price
                # out a safe next-day sale from the same request.
                admission = self._admission_reason(candidate_late, request, 0.50)
                if admission is not None:
                    self.rejected_options[admission] += 1
                    continue
                threshold, cost, minimum, ceiling, speed = self._option_terms(
                    request, option, risk, pressure, candidate_late
                )
                terms.append({"option": option, "risk": risk, "threshold": threshold,
                              "cost": cost, "minimum": minimum, "ceiling": ceiling,
                              "speed": speed})
                offered_late.append(candidate_late)

        if terms:
            # The simultaneous menu optimiser is retained for controlled
            # experiments, but the first paired seed regressed.  Keep the
            # corrected menu-level learning while pricing each speed with the
            # proven independent expected-margin rule by default.
            if os.getenv("V5_MENU_PRICE_OPTIMIZER", "0") == "1":
                prices, records = self._price_menu(terms)
            else:
                prices, records = self._price_independently(terms)

        # Quote high rather than omit a request: it avoids a goodwill loss and
        # does not add capacity risk when it loses.
        if not prices:
            self.declined_requests += 1
            slowest = max(request["options"], key=lambda option: option["deadline"])
            prices = {slowest["kind"]: max(1, slowest["listPriceCents"] * 6)}
        else:
            self.options_offered += len(prices)
            lane = self._lane(request)
            self.quote_book[request["id"]] = {
                "dealer": request["dealerId"], "lane": lane,
                "day": request["arrivedAt"] // DAY,
                "trust": math.log1p(0.01 * self.dealer_trust[request["dealerId"]]),
                "options": records,
            }
            choice_probability = sum(self._menu_probabilities(records).values())
            # One dealer can choose one speed, so reserve the maximum plausible
            # workload of the menu rather than adding every displayed option.
            self.reserved_late[request["id"]] = max(offered_late) * choice_probability
            self._reserve_resources(request["id"], request, choice_probability, "quoted")
        self.outbox.append({"type": "quote", "requestId": request["id"], "prices": prices})
        self.pending_quotes[request["id"]] = (len(request["legs"]), request["respondBy"])

    def _request_decline_reason(self, request: dict[str, Any], facts: note_constraints.NoteConstraints) -> str | None:
        """Reject work whose operational facts cannot be safely represented.

        A lost six-times-list quote is free.  An accepted job with an unknown
        private slot, premium-driver restriction, trade-in leg, or ambiguous
        note can damage every later quote through the delivery minimum.
        """
        if len(request["legs"]) != 1:
            return "trade_in"
        leg = request["legs"][0]
        if leg["car"]["premium"]:
            return "premium"
        if leg["pickup"]["privateAddress"] and os.getenv("V5_ALLOW_PRIVATE_PICKUP", "1") != "1":
            return "private_pickup"
        if facts.status == "ambiguous":
            return "ambiguous_note"
        return None

    @staticmethod
    def _window_with_facts(place: dict[str, Any], facts: note_constraints.NoteConstraints,
                           kind: str) -> tuple[int, int]:
        """Apply only facts the deterministic reader positively recognised."""
        start, end = place["window"]["from"], place["window"]["to"]
        if kind == "pickup":
            if facts.pickup_after is not None:
                start = max(start, facts.pickup_after)
            if facts.pickup_before is not None:
                end = min(end, facts.pickup_before)
        else:
            if facts.delivery_after is not None:
                start = max(start, facts.delivery_after)
            if facts.delivery_before is not None:
                end = min(end, facts.delivery_before)
        return start, end

    @staticmethod
    def _hours_for_day(hours: dict[str, Any] | None, day: int) -> tuple[int, int] | None:
        """Return a site's stated opening interval for a world day."""
        if hours is None:
            return None
        weekday = day % 7
        if weekday < 5 and weekday in hours.get("weekdays", []):
            return int(hours["opens"]), int(hours["closes"])
        saturday = hours.get("saturday")
        if weekday == 5 and saturday is not None:
            return int(saturday["opens"]), int(saturday["closes"])
        return None

    @staticmethod
    def _effective_hours(hours: dict[str, Any] | None,
                         override: tuple[int | None, int | None, bool] | None) -> dict[str, Any] | None:
        """Overlay parsed note opening hours onto a site's stated hours."""
        if override is None:
            return hours
        ov_opens, ov_closes, weekdays_only = override
        if hours is None:
            o, c = 0 if ov_opens is None else ov_opens, DAY if ov_closes is None else ov_closes
            return {
                "opens": o, "closes": c, "weekdays": [0, 1, 2, 3, 4],
                "saturday": None if weekdays_only else {"opens": o, "closes": c},
            }
        merged = dict(hours)
        if ov_opens is not None:
            merged["opens"] = max(int(hours["opens"]), ov_opens)
        if ov_closes is not None:
            merged["closes"] = min(int(hours["closes"]), ov_closes)
        if weekdays_only:
            merged["saturday"] = None
        elif hours.get("saturday") is not None:
            sat = dict(hours["saturday"])
            if ov_opens is not None:
                sat["opens"] = max(int(sat["opens"]), ov_opens)
            if ov_closes is not None:
                sat["closes"] = min(int(sat["closes"]), ov_closes)
            merged["saturday"] = sat if sat["opens"] < sat["closes"] else None
        return merged

    def _first_open_minute(self, place: dict[str, Any], start: int, end: int,
                           hours_override: tuple[int | None, int | None, bool] | None = None) -> int | None:
        """First usable minute in a stop window, including dealer opening hours."""
        if start > end:
            return None
        # Unit fixtures and old snapshots before the hours field was added
        # provide a window only; that window is already the best information
        # available and must remain usable after a restart.
        effective = self._effective_hours(place.get("hours"), hours_override)
        if (place.get("privateAddress") or effective is None) and hours_override is None:
            return start
        if effective is None:
            return start
        for day in range(start // DAY, end // DAY + 1):
            hours = self._hours_for_day(effective, day)
            if hours is None:
                continue
            opens, closes = day * DAY + hours[0], day * DAY + hours[1]
            candidate = max(start, opens)
            if candidate <= min(end, closes):
                return candidate
        return None

    def _last_open_minute(self, place: dict[str, Any], start: int, end: int,
                          hours_override: tuple[int | None, int | None, bool] | None = None) -> int | None:
        """Last usable minute in a stop window, including dealer opening hours."""
        if start > end:
            return None
        effective = self._effective_hours(place.get("hours"), hours_override)
        if (place.get("privateAddress") or effective is None) and hours_override is None:
            return end
        if effective is None:
            return end
        for day in range(end // DAY, start // DAY - 1, -1):
            hours = self._hours_for_day(effective, day)
            if hours is None:
                continue
            opens, closes = day * DAY + hours[0], day * DAY + hours[1]
            candidate = min(end, closes)
            if candidate >= max(start, opens):
                return candidate
        return None

    @staticmethod
    def _waking_minutes(start: int, end: int) -> int:
        """Minutes in the driver's 07:00–21:00 discovery period."""
        total = 0
        for day in range(start // DAY, end // DAY + 1):
            total += max(0, min(end, day * DAY + 21 * HOUR) - max(start, day * DAY + 7 * HOUR))
        return total

    @classmethod
    def _rewind_waking_minutes(cls, end: int, minutes: int) -> int:
        """Move backwards through the 07:00–21:00 waking clock."""
        at, remaining = end, minutes
        while remaining > 0:
            day = at // DAY
            waking_start, waking_end = day * DAY + 7 * HOUR, day * DAY + 21 * HOUR
            if at > waking_end:
                at = waking_end
            available = max(0, at - waking_start)
            if available >= remaining:
                return at - remaining
            remaining -= available
            at = day * DAY - 1
        return at

    @classmethod
    def _advance_waking_minutes(cls, start: int, minutes: int) -> int:
        """Move forwards through the 07:00–21:00 waking clock."""
        at, remaining = start, minutes
        while remaining > 0:
            day = at // DAY
            waking_start, waking_end = day * DAY + 7 * HOUR, day * DAY + 21 * HOUR
            if at < waking_start:
                at = waking_start
            if at >= waking_end:
                at = (day + 1) * DAY + 7 * HOUR
                continue
            available = waking_end - at
            if available >= remaining:
                return at + remaining
            remaining -= available
            at = (day + 1) * DAY + 7 * HOUR
        return at

    def _on_time_plan(self, request: dict[str, Any], option: dict[str, Any],
                      facts: note_constraints.NoteConstraints) -> tuple[dict[str, int] | None, str | None]:
        """Compute the conservative last on-time pickup before accepting work."""
        leg = request["legs"][0]
        pickup_start, pickup_end = self._window_with_facts(leg["pickup"], facts, "pickup")
        delivery_start, delivery_end = self._window_with_facts(leg["delivery"], facts, "delivery")
        delivery_end = min(delivery_end, option["deadline"])
        effective_start = max(pickup_start, self.now)
        earliest = self._first_open_minute(leg["pickup"], effective_start, pickup_end, facts.pickup_hours)
        latest_delivery = self._last_open_minute(leg["delivery"], delivery_start, delivery_end, facts.delivery_hours)
        if earliest is None or latest_delivery is None:
            return None, "closed_window"
        distance, drive = self._road(leg)
        car = leg["car"]
        if car["electric"] and car["chargePercent"] is not None and car["chargePercent"] < 30 and distance > 200:
            drive += 40
        if facts.with_plates:
            drive += 30
        if os.getenv("V5_SNOW_DELIV_RISK", "0") == "1":
            places = {leg["pickup"]["placeId"], leg["delivery"]["placeId"]}
            if any(f["from"] <= option["deadline"] and f["to"] >= self.now and (places & f["placeIds"])
                   for f in self.forecasts):
                drive = int(drive * float(os.getenv("V5_SNOW_DRIVE_MULT", "1.25")) + 0.5)
        # The 30-minute buffer is deliberately before the final pickup rather
        # than after it: actual drives and handovers regularly overrun the map.
        latest_pickup = self._last_open_minute(
            leg["pickup"], effective_start, min(pickup_end, latest_delivery - drive - 30), facts.pickup_hours
        )
        if latest_pickup is None or latest_pickup < earliest:
            return None, "route_window"
        if latest_pickup - earliest < 150:
            return None, "pickup_room"
        min_waking = int(os.getenv("V5_MIN_WAKING_MINUTES", "360"))
        if self._waking_minutes(self.now, latest_pickup) < min_waking:
            return None, "driver_discovery"
        return {"earliestPickup": earliest, "lastPickup": latest_pickup,
                "latestDelivery": latest_delivery}, None

    def _option_terms(self, request: dict[str, Any], option: dict[str, Any], risk: float,
                      pressure: float, late_probability: float | None = None) -> tuple[float, float, float, float, float]:
        """Return bounded economics for one delivery speed within a quote menu."""
        mode = os.getenv("V5_MARKET_MODE", "world")
        if mode in ("world", "kalman"):
            threshold = self.market_log
            speed = max(0.0, max(item["deadline"] for item in request["options"]) - option["deadline"]) / DAY
            kind_offsets = os.getenv("V5_KIND_OFFSETS", "")
            if kind_offsets:
                parts = [float(x) for x in kind_offsets.split(",")]
                idx = {"same_day": 0, "next_day": 1, "within_3_days": 2}.get(option["kind"])
                if idx is not None and idx < len(parts):
                    threshold += parts[idx]
        else:
            lane = self._lane(request)
            threshold = self.market_log + self.dealer_market[request["dealerId"]]
            threshold += self.lane_market[lane] + self.day_market[request["arrivedAt"] // DAY]
            speed = max(0.0, max(item["deadline"] for item in request["options"]) - option["deadline"]) / DAY
            threshold += speed * (0.04 + self.dealer_speed[request["dealerId"]])
        threshold += math.log1p(0.01 * self.dealer_trust[request["dealerId"]])
        if self.now < DAY:
            threshold += float(os.getenv("V5_MONDAY_QUOTE_SHIFT", "0.0"))
        cost = self._transport_cost(
            request, option, pressure,
            late_probability if late_probability is not None else self._late_probability(request, option, pressure),
        )
        listed = max(1, option["listPriceCents"])
        # Search the entire requested 0.80–3.00× list interval.  Cost belongs
        # in the objective; treating it as a quote floor hides the genuinely
        # optimal price whenever the market happens to be cheap that morning.
        minimum, ceiling = listed * 0.80, listed * 3.00
        return threshold, cost, minimum, ceiling, speed

    def _safe_slack_discount(self, request: dict[str, Any], option: dict[str, Any], risk: float) -> float:
        """Discount only abundant-slack work; never use price to buy risky volume.

        This is deliberately opt-in while we measure it.  A global lower quote
        floor produced an unranked week, so the discount excludes every common
        source of operational uncertainty before it reaches the price floor.
        """
        maximum = max(0.0, min(0.15, float(os.getenv("V5_SAFE_PRICE_DISCOUNT", "0"))))
        if maximum <= 0.0 or risk > 0.35 or len(request["legs"]) != 1 or request.get("note"):
            return 0.0
        leg = request["legs"][0]
        car = leg["car"]
        if (leg["pickup"]["privateAddress"] or leg["delivery"]["privateAddress"]
                or car["premium"] or car["electric"]):
            return 0.0
        places = {leg["pickup"]["placeId"], leg["delivery"]["placeId"]}
        if any(forecast["from"] <= option["deadline"] and forecast["to"] >= self.now
               and places & forecast["placeIds"] for forecast in self.forecasts):
            return 0.0
        route_hours = self._road(leg)[1] / HOUR
        slack_hours = (option["deadline"] - self.now) / HOUR - route_hours
        # Full discount starts only once there is a full day of surplus room;
        # 12–24h receives a linear partial discount.
        slack_factor = max(0.0, min(1.0, (slack_hours - 12.0) / 12.0))
        return maximum * slack_factor

    def _menu_probabilities(self, records: dict[str, dict[str, float]]) -> dict[str, float]:
        """Choice probabilities for the offered speeds plus the outside option."""
        weights = {
            kind: math.exp(max(-30.0, min(30.0, (
                item["threshold"] - math.log(item["price"] / item["list"])
            ) / self.market_width)))
            for kind, item in records.items()
        }
        denominator = 1.0 + sum(weights.values())  # competitor / no-buy utility is zero
        return {kind: weight / denominator for kind, weight in weights.items()}

    def _price_menu(self, terms: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, dict[str, float]]]:
        """Choose a coherent price menu via exact Dinkelbach MNL fractional optimization."""
        best_records: dict[str, dict[str, float]] = {}
        value = 0.0
        for _ in range(8):
            records: dict[str, dict[str, float]] = {}
            for item in terms:
                option = item["option"]
                listed = max(1, int(option["listPriceCents"]))
                best_price = max(1, int(listed * 0.80 + 0.5))
                best_gain = float("-inf")
                for tick in range(221):
                    price = max(1, int(listed * (0.80 + 0.01 * tick) + 0.5))
                    weight = math.exp(max(-30.0, min(30.0, (
                        item["threshold"] - math.log(price / listed)
                    ) / self.market_width)))
                    gain = weight * (price - item["cost"] - value)
                    if gain > best_gain:
                        best_gain, best_price = gain, price
                records[option["kind"]] = {
                    "price": float(best_price), "threshold": item["threshold"],
                    "cost": item["cost"], "list": float(listed),
                    "speed": item["speed"], "risk": item["risk"],
                }
            probabilities = self._menu_probabilities(records)
            next_value = sum(probabilities[kind] * (item["price"] - item["cost"]) for kind, item in records.items())
            best_records = records
            if abs(next_value - value) < 1e-4:
                break
            value = max(0.0, next_value)
        return ({kind: int(item["price"]) for kind, item in best_records.items()}, best_records)

    def _price_independently(self, terms: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, dict[str, float]]]:
        """The qualified one-speed expected-margin rule, represented as a menu record."""
        records: dict[str, dict[str, float]] = {}
        for item in terms:
            option = item["option"]
            listed = max(1, option["listPriceCents"])
            best_price, best_value = int(listed * 0.80 + 0.5), float("-inf")
            for tick in range(221):
                price = max(1, int(listed * (0.80 + 0.01 * tick) + 0.5))
                accepted = sigmoid((item["threshold"] - math.log(price / listed)) / self.market_width)
                value = accepted * (price - item["cost"])
                if value > best_value:
                    best_price, best_value = price, value
            records[option["kind"]] = {
                "price": float(best_price), "threshold": item["threshold"],
                "cost": item["cost"], "list": float(listed),
                "speed": item["speed"], "risk": item["risk"],
            }
        return ({kind: int(item["price"]) for kind, item in records.items()}, records)

    def _learn_quote(self, request_id: str, won_option: str | None, accepted: bool) -> None:
        record = self.quote_book.pop(request_id, None)
        if record is None:
            return
        records = record["options"]
        mode = os.getenv("V5_MARKET_MODE", "world")
        if mode == "kalman":
            trust = record.get("trust", 0.0)
            if not accepted and os.getenv("V5_KALMAN_MULTIMENU", "0") == "1":
                p = max(1e-4, min(1.0 - 1e-4, sum(self._menu_probabilities(records).values())))
            else:
                if accepted and won_option in records:
                    item = records[won_option]
                    x = math.log(item["price"] / item["list"]) - trust
                else:
                    x = min(math.log(item["price"] / item["list"]) - trust for item in records.values())
                p = sigmoid((self.market_log - x) / self.market_width)
            info = max(0.05, p * (1.0 - p)) / (self.market_width * self.market_width)
            self.market_var = 1.0 / (1.0 / max(1e-6, self.market_var) + info)
            self.market_log = max(
                math.log(0.85),
                min(math.log(2.10), self.market_log + self.market_var * (float(accepted) - p) / self.market_width),
            )
            return
        probabilities = self._menu_probabilities(records)
        accepted_probability = sum(probabilities.values())
        error = float(accepted) - accepted_probability
        if mode == "world":
            rate = float(os.getenv("V5_WORLD_MARKET_RATE", "0.16"))
            count = self.market_observations[("global", "all")]
            self.market_observations[("global", "all")] = count + 1
            self.market_log = max(-0.20, min(0.75, self.market_log + rate / math.sqrt(8.0 + count) * error))
            return
        self.market_log = self._adjust_market("global", "all", self.market_log, error, 0.055, hi=0.75)
        self.dealer_market[record["dealer"]] = self._adjust_market(
            "dealer", record["dealer"], self.dealer_market[record["dealer"]], error, 0.025
        )
        self.lane_market[record["lane"]] = self._adjust_market(
            "lane", record["lane"], self.lane_market[record["lane"]], error, 0.035
        )
        self.day_market[record["day"]] = self._adjust_market(
            "day", record["day"], self.day_market[record["day"]], error, 0.060
        )
        if accepted and won_option in records:
            speed_error = sum(
                ((1.0 if kind == won_option else 0.0) - probability) * records[kind]["speed"]
                for kind, probability in probabilities.items()
            )
            dealer = record["dealer"]
            self.dealer_speed[dealer] = self._bound(self.dealer_speed[dealer] + 0.020 * speed_error)

    def _adjust_market(self, scope: str, key: Any, current: float, error: float, rate: float,
                       lo: float = -0.40, hi: float = 0.40) -> float:
        """Make local beliefs move slowly until their observations are earned."""
        observation = (scope, key)
        count = self.market_observations[observation]
        self.market_observations[observation] = count + 1
        return max(lo, min(hi, current + rate / math.sqrt(8.0 + count) * error))

    @staticmethod
    def _bound(value: float) -> float:
        return max(-0.40, min(0.40, value))

    def _refresh_market_uncertainty(self, at: int) -> None:
        """Forget confidence, not price information, at market regime changes."""
        checkpoints: list[int] = []
        for day in range(0, at // DAY + 1):
            checkpoints.extend((day * DAY, day * DAY + 8 * HOUR, day * DAY + 18 * HOUR))
        latest = max((point for point in checkpoints if point <= at), default=-1)
        if latest <= self.market_refreshed_through:
            return
        self.market_refreshed_through = latest
        # The competitor moves at the start/end of the working day.  Retaining
        # the level avoids a Monday-style cold start; relaxing uncertainty lets
        # the next dozen outcomes track regime shifts immediately.
        max_std = float(os.getenv("V5_KALMAN_MAX_STD", "0.14"))
        jump_std = float(os.getenv("V5_KALMAN_JUMP", "0.09"))
        self.market_var = min(max_std ** 2, self.market_var + jump_std ** 2)
        for key, count in list(self.market_observations.items()):
            self.market_observations[key] = max(0, count // 2)

    # Capacity and chain planning ----------------------------------------

    def _portfolio_pressure(self, request: dict[str, Any]) -> float:
        """Expected load at the constrained pickup/delivery resources."""
        profile = self._resource_profile(request)
        loads = self._resource_load()
        return max((loads.get(bucket, 0.0) / self._resource_capacity(bucket[0])
                    for bucket in profile), default=0.0)

    def _resource_profile(self, request: dict[str, Any]) -> dict[tuple[str, int], float]:
        """Small, explicit approximation of the driver time/place resources a job uses."""
        profile: dict[tuple[str, int], float] = defaultdict(float)
        for leg in request["legs"]:
            route_minutes = self._road(leg)[1]
            workload = 1.0 + route_minutes / 360.0
            pickup = leg["pickup"]
            delivery = leg["delivery"]
            pickup_bucket = pickup["window"]["from"] // (3 * HOUR)
            delivery_bucket = delivery["window"]["from"] // (3 * HOUR)
            profile[(pickup["placeId"], pickup_bucket)] += workload
            # Arriving for a pickup consumes some preceding availability; after
            # delivery there is a smaller return/continuation opportunity cost.
            profile[(pickup["placeId"], pickup_bucket - 1)] += 0.50 * workload
            profile[(delivery["placeId"], delivery_bucket)] += 0.25 * workload
        return dict(profile)

    def _resource_load(self) -> dict[tuple[str, int], float]:
        loads: dict[tuple[str, int], float] = defaultdict(float)
        for reservation in self.resource_reservations.values():
            for bucket, weight in reservation["profile"].items():
                loads[bucket] += reservation["claim"] * weight
        return dict(loads)

    def _resource_capacity(self, place_id: str) -> float:
        """A shrunken home-density prior; availability remains deliberately unknown."""
        base_capacity = float(os.getenv("V5_RESOURCE_BUCKET_CAP", "16"))
        return max(6.0, min(24.0, base_capacity + 0.50 * self.home_supply[place_id]))

    def _reserve_resources(self, request_id: str, request: dict[str, Any], probability: float,
                           phase: str) -> None:
        self.resource_reservations[request_id] = {
            "phase": phase, "claim": max(0.0, min(1.0, probability)),
            "profile": self._resource_profile(request),
        }

    def _marginal_resource_pressure(self, request: dict[str, Any], probability: float) -> float:
        profile, loads = self._resource_profile(request), self._resource_load()
        return max(((loads.get(bucket, 0.0) + probability * weight) / self._resource_capacity(bucket[0])
                    for bucket, weight in profile.items()), default=0.0)

    def _unassigned_orders(self) -> int:
        """Orders we own that still have no booked driver."""
        total = 0
        for order_id, order in self.orders.items():
            if not base.undelivered(order):
                continue
            gig_ids = self.gigs_of(order_id)
            if not gig_ids or not any(gig_id in self.booked for gig_id in gig_ids):
                total += 1
        return total

    def _nearby_driver_count(self, place_id: str) -> int:
        """Public roster density around a stop, using the road map as distance."""
        if not self.home_supply:
            return 0
        count = 0
        for home, drivers in self.home_supply.items():
            road = self.roads.get((home, place_id))
            if home == place_id or (road is not None and road[0] <= 60.0):
                count += drivers
        return count

    def _late_probability(self, request: dict[str, Any], option: dict[str, Any] | None,
                          local_pressure: float, plan: dict[str, int] | None = None) -> float:
        """Logistic late model fitted on level-8 revealed weeks, with rule fallback."""
        if option is None:
            return 0.45
        leg = request["legs"][0]
        if plan is None:
            plan = self.option_plans.get(request["id"], {}).get(option["kind"])
        pickup = leg["pickup"]["placeId"]
        _, drive = self._road(leg)
        last_pickup = (plan or {}).get("lastPickup", option["deadline"] - drive - 30)
        if os.getenv("V5_SNOW_DELIV_RISK", "0") == "1":
            places = {pickup, leg["delivery"]["placeId"]}
            snow = any(
                forecast["from"] <= option["deadline"] and forecast["to"] >= self.now and (places & forecast["placeIds"])
                for forecast in self.forecasts
            )
        else:
            snow = any(
                forecast["from"] <= last_pickup and forecast["to"] >= self.now and pickup in forecast["placeIds"]
                for forecast in self.forecasts
            )
        waking_h = self._waking_minutes(self.now, last_pickup) / HOUR
        if self.home_supply:
            min_near60 = min(
                self._nearby_driver_count(pickup),
                self._nearby_driver_count(leg["delivery"]["placeId"]),
            )
            low_near60 = max(0.0, 1.0 - math.log1p(min_near60) / math.log1p(30.0))
        else:
            min_near60 = 99
            low_near60 = 0.0
        unassigned = self._unassigned_orders()
        late_model = os.getenv("V5_LATE_MODEL", "logreg")
        if late_model == "logreg_v3":
            z = (
                -5.6528
                + 0.6663 * float(leg["delivery"]["privateAddress"])
                + 0.7395 * float(snow)
                + 2.6083 * max(0.0, 1.0 - waking_h / 18.0)
                + 1.0681 * low_near60
                + 1.7060 * min(2.0, drive / 300.0)
                + 0.4698 * min(2.0, max(0, unassigned - 15) / 25.0)
                + 0.0627 * float(option["kind"] == "same_day")
            )
            return max(0.005, min(0.65, sigmoid(z)))
        if late_model == "logreg_v2":
            z = (
                -5.0878
                + 0.0428 * float(leg["delivery"]["privateAddress"])
                + 0.4777 * float(snow)
                + 1.7703 * max(0.0, 1.0 - waking_h / 18.0)
                + 0.8499 * low_near60
                + 1.1109 * min(2.0, drive / 300.0)
                + 0.3679 * min(2.0, max(0, unassigned - 15) / 25.0)
                + 0.1356 * float(option["kind"] == "same_day")
            )
            return max(0.005, min(0.65, sigmoid(z)))
        if late_model == "logreg":
            low_bat = float(leg["car"]["electric"] and (leg["car"]["chargePercent"] or 100) < 30)
            z = (
                -3.6574
                + 0.2691 * float(leg["delivery"]["privateAddress"])
                + 0.4268 * float(snow)
                + 1.2422 * max(0.0, 1.0 - waking_h / 18.0)
                + 0.8215 * low_near60
                + 1.0118 * min(2.0, drive / 300.0)
                + 0.3774 * low_bat
                + 0.20 * min(2.0, max(0, unassigned - 15) / 25.0)
            )
            return max(0.01, min(0.65, sigmoid(z)))
        signs = 0
        if leg["delivery"]["privateAddress"]:
            signs += 1
        if snow:
            signs += 1
        if waking_h < 6.0:
            signs += 1
        if self.home_supply and min_near60 < 4:
            signs += 1
        if unassigned > 15 or local_pressure > 1.0:
            signs += 1
        base_probability = float(os.getenv("V5_BASE_LATE_PROBABILITY", "0.05"))
        learned_adjustment = self.reliability_bias if os.getenv("V5_LEARN_LATE_RESIDUAL", "0") == "1" else 0.0
        return max(0.005, min(0.45, base_probability * (2 ** signs) + learned_adjustment))

    def _admission_reason(self, candidate_late: float, request: dict[str, Any],
                          win_probability: float) -> str | None:
        """Name the first hard gate that rejects an option, without mutating state."""
        existing = sum(self.reserved_late.values())
        denominator = self.completed_orders + len(self.reserved_late) + 1
        expected_share = (self.late_orders + existing + candidate_late * win_probability) / denominator
        target = float(os.getenv("V5_LATE_TARGET", "1.0"))
        local_limit = float(os.getenv("V5_RESOURCE_PRESSURE_LIMIT", "100.0"))
        if candidate_late >= 0.45:
            return "late_risk"
        if expected_share > target:
            return "late_budget"
        if self._marginal_resource_pressure(request, win_probability) > local_limit:
            return "resource_pressure"
        return None

    def _admit(self, request_id: str, candidate_late: float, request: dict[str, Any],
               win_probability: float) -> bool:
        """Compatibility wrapper for callers that need only the binary gate."""
        return self._admission_reason(candidate_late, request, win_probability) is None

    def _connection_value(self, request: dict[str, Any]) -> float:
        """Value a current continuation opportunity without assuming it will win."""
        leg = request["legs"][0]
        best = 0.0
        for order_id in self.waiting:
            other = self.orders.get(order_id)
            if other is None or not self._chainable(other):
                continue
            other_leg = other["legs"][0]
            bridge = self.roads.get((other_leg["delivery"]["placeId"], leg["pickup"]["placeId"]))
            if bridge and bridge[0] <= 60:
                best = max(best, 1.0 - bridge[0] / 60.0)
        return best

    def _take_second_look(self) -> None:
        """Return an unbooked, stale single gig to the pool for one re-plan."""
        pair_patience = int(os.getenv("V5_PAIR_PATIENCE_MINUTES", "45"))
        for gig_id, action in list(self.posts.items()):
            members = self._action_orders(action)
            if len(members) <= 1 or gig_id in self.booked:
                continue
            if self.now - self.posted_at.get(gig_id, self.now) < pair_patience:
                continue
            # A pair had its chance.  Do not churn it back into the exact same
            # pair: return both cars to independent, flat-fee posting.
            self.pair_failed_orders.update(members)
            self._unplan_open_gigs(members[0])

        wait = int(os.getenv("V5_SECOND_LOOK_MINUTES", "0"))
        if wait <= 0:
            return
        for order_id, order in list(self.orders.items()):
            if order_id in self.waiting or order_id in self.urgent:
                continue
            gig_ids = self.gigs_of(order_id)
            if len(gig_ids) != 1 or gig_ids[0] in self.booked:
                continue
            gig_id = gig_ids[0]
            action = self.posts.get(gig_id)
            if action is None or len(self._action_orders(action)) != 1:
                continue
            if self._v52_fee_model():
                state = self.gig_fee_state.get(gig_id)
                if (state is not None and not state["repriced"]
                        and self.now - state["postedAt"] >= 6 * HOUR
                        and len(self.gig_views[gig_id]) >= 3):
                    state["repriced"] = True
                    self.outbox.append({"type": "reprice", "gigId": gig_id,
                                        "schedule": [{"at": self.now,
                                                      "feeCents": int(state["openingFee"] * 1.12 + 0.5)}]})
                # In V5.2 low-evidence singles wait for expiry or the deadline
                # rescue; they are not churned through a blind withdraw/repost.
                continue
            if self.now - self.posted_at.get(gig_id, self.now) < wait:
                continue
            self._unplan_open_gigs(order_id)

    def _plan_waiting(self) -> None:
        used: set[str] = set()
        if os.getenv("V5_ENABLE_CHAINING", "1") == "1":
            candidates = [
                order_id for order_id in self.waiting
                if order_id in self.orders and order_id not in self.pair_failed_orders
                and self._chainable(self.orders[order_id])
            ]

            # A fresh order may unlock an already-open single gig.  Withdraw that
            # open gig first, then post the one safe pair in the same action batch.
            for fresh_id in candidates:
                if fresh_id in used:
                    continue
                best: list[str] | None = None
                best_merit = float("-inf")
                for open_id in self._open_single_orders():
                    if open_id == fresh_id or open_id in used:
                        continue
                    for sequence in ([open_id, fresh_id], [fresh_id, open_id]):
                        merit = self._chain_merit(sequence)
                        if merit > best_merit:
                            best, best_merit = sequence, merit
                if best is not None and self._unplan_open_gigs(next(order_id for order_id in best if order_id != fresh_id)):
                    self._post_chain(best)
                    used.update(best)

            for order_id in candidates:
                if order_id in used:
                    continue
                chain = [order_id]
                while len(chain) < int(os.getenv("V5_CHAIN_MAX_CARS", "2")):
                    additions = [
                        candidate for candidate in candidates
                        if candidate not in used and candidate not in chain
                        and self._chain_plan(chain + [candidate]) is not None
                    ]
                    if not additions:
                        break
                    chain.append(max(additions, key=lambda candidate: self._chain_merit(chain + [candidate])))
                if len(chain) > 1:
                    self._post_chain(chain)
                    used.update(chain)

        hold = int(os.getenv("V5_POOL_MINUTES", "0"))
        for order_id, queued_at in list(self.waiting.items()):
            order = self.orders.get(order_id)
            if order is None:
                self.waiting.pop(order_id, None)
            elif not base.undelivered(order):
                self.waiting.pop(order_id, None)
            elif order_id not in used and self.now >= queued_at + hold:
                self.post(order, base.undelivered(order))
                self.waiting.pop(order_id, None)

    def _open_single_orders(self) -> list[str]:
        available: list[str] = []
        for order_id, order in self.orders.items():
            if order_id in self.waiting or order_id in self.urgent or not self._chainable(order):
                continue
            gig_ids = self.gigs_of(order_id)
            if len(gig_ids) != 1 or gig_ids[0] in self.booked:
                continue
            action = self.posts.get(gig_ids[0])
            if action is not None and self._action_orders(action) == [order_id]:
                available.append(order_id)
        return available

    def _chainable(self, order: dict[str, Any]) -> bool:
        if len(order["legs"]) != 1 or not base.undelivered(order):
            return False
        request = self.requests.get(order["id"])
        if request is None:
            return False
        if request.get("note") is not None:
            if os.getenv("V5_CHAIN_ALLOW_HARMLESS_NOTES", "0") != "1":
                return False
            facts = self.note_facts.get(order["id"])
            if facts is None or facts.has_fact or facts.manual_licence or facts.with_plates:
                return False
        leg = order["legs"][0]
        if leg["pickup"]["privateAddress"] or leg["delivery"]["privateAddress"] or leg["car"]["premium"]:
            return False
        if leg["car"]["electric"]:
            min_soc = os.getenv("V5_CHAIN_ELECTRIC_MIN_SOC", "")
            if not min_soc:
                return False
            if (leg["car"]["chargePercent"] or 0) < int(min_soc):
                return False
        return True

    def _chain_plan(self, order_ids: list[str]) -> tuple[int, float] | None:
        """Return (finish minute, bridge km) when an ordered chain is feasible."""
        at = self.now + int(os.getenv("V5_CHAIN_BUFFER_MINUTES", "150"))
        driving = 0
        bridges = 0.0
        previous: dict[str, Any] | None = None
        for order_id in order_ids:
            order = self.orders[order_id]
            leg = order["legs"][0]
            if previous is not None:
                bridge = self.roads.get((previous["delivery"]["placeId"], leg["pickup"]["placeId"]))
                if bridge is None:
                    return None
                max_bridge = os.getenv("V5_CHAIN_MAX_BRIDGE_KM", "30")
                if max_bridge and bridge[0] > float(max_bridge):
                    return None
                at += bridge[1]
                driving += bridge[1]
                bridges += bridge[0]
            at = max(at, leg["pickup"]["window"]["from"])
            if at > order["lastPickupAt"]:
                return None
            road = self._road(leg)
            at = max(at + road[1], leg["delivery"]["window"]["from"])
            driving += road[1]
            if at > order["deadline"]:
                return None
            previous = leg
        if len(order_ids) > 1 and driving > 12 * HOUR:
            return None
        return at, bridges

    def _chain_merit(self, order_ids: list[str]) -> float:
        plan = self._chain_plan(order_ids)
        if plan is None:
            return float("-inf")
        references = sum(self.requests[order_id]["referenceFeeCents"] for order_id in order_ids)
        risk = sum(self._route_risk(self.requests[order_id], self._won_option(order_id)) for order_id in order_ids)
        return references - plan[1] * 80 - risk * 2_500

    def _post_chain(self, order_ids: list[str]) -> None:
        if self._chain_plan(order_ids) is None:
            return
        stops: list[dict[str, Any]] = []
        for order_id in order_ids:
            order = self.orders[order_id]
            stops.extend(base.stops_for(order, [0], None, base.Retry()))
        stops = base.without_windows(stops, self.now)
        self.outbox.append({
            "type": "post", "stops": stops,
            "audience": {"experienced": False, "manualLicence": False},
            "schedule": self._bundle_schedule(order_ids, urgent=False),
        })
        self.chain_posts += 1
        self.chain_cars += len(order_ids)
        for order_id in order_ids:
            self.waiting.pop(order_id, None)

    # Cost and fee model ---------------------------------------------------

    def post(self, order: Any, legs: list[int], retry: Any = base.Retry()) -> None:
        """Post one safe gig, applying only note facts we understood."""
        if not legs or (self.ends_at is not None and self.now >= self.ends_at):
            return
        self.single_posts += len(legs)
        request = self.requests[order["id"]]
        facts = self.note_facts.get(order["id"], note_constraints.NoteConstraints())
        audience = {
            "experienced": any(leg["car"]["premium"] for leg in order["legs"]),
            "manualLicence": bool(retry.manual_licence or facts.manual_licence),
        }
        stops = self._stops_with_facts(order, legs, retry, facts)
        self.outbox.append({"type": "post", "stops": stops, "audience": audience,
                            "schedule": self.fees_for(order, len(legs))})
        if len(legs) == 1:
            return
        # Keep the kit's trade-in fallback intact for any order that was won
        # before a restart restored the current strict eligibility policy.
        self.send()
        if not self.gigs_of(order["id"]):
            for leg in legs:
                self.post(order, [leg], retry)

    def _next_open_interval(self, place: dict[str, Any], start: int, end: int,
                            hours_override: tuple[int | None, int | None, bool] | None,
                            min_at: int) -> tuple[int, int] | None:
        """Next single-day [opens, closes] interval when note hours restrict a site."""
        if start > end or hours_override is None:
            return None
        effective = self._effective_hours(place.get("hours"), hours_override)
        if effective is None:
            return None
        for day in range(max(start, min_at) // DAY, end // DAY + 1):
            hours = self._hours_for_day(effective, day)
            if hours is None:
                continue
            opens = max(start, day * DAY + hours[0])
            closes = min(end, day * DAY + hours[1])
            if closes > max(opens, min_at):
                return opens, closes
        return None

    def _stops_with_facts(self, order: dict[str, Any], legs: list[int], retry: Any,
                          facts: note_constraints.NoteConstraints) -> list[dict[str, Any]]:
        stops = base.stops_for(order, legs, None, retry)
        is_late = order["id"] in self.late_recovery_orders or self.now > order["lastPickupAt"]
        for stop in stops:
            leg = order["legs"][stop["leg"]]
            kind = stop["kind"]
            place = leg[kind]
            start, end = self._window_with_facts(place, facts, kind)
            hours_override = facts.pickup_hours if kind == "pickup" else facts.delivery_hours
            if kind == "pickup":
                start = max(start, retry.pickup_from)
                if facts.with_plates:
                    stop["withPlates"] = True
            horizon = max(end, self.now + 14 * DAY) if is_late else end
            day_interval = (
                self._next_open_interval(place, start, horizon, hours_override, self.now)
                if hours_override is not None else None
            )
            if day_interval is not None:
                start, end = day_interval
            elif hours_override is not None:
                first_open = self._first_open_minute(place, start, horizon, hours_override)
                last_open = self._last_open_minute(place, start, horizon, hours_override)
                if first_open is not None and last_open is not None and first_open <= last_open:
                    start, end = max(start, first_open), min(horizon, last_open)
            base_window = place["window"]
            narrowed = {"from": start, "to": max(start, end)}
            if narrowed != base_window:
                stop["window"] = narrowed
            if is_late and "window" in stop and hours_override is None:
                if kind == "delivery" or stop["window"]["from"] <= self.now:
                    stop.pop("window", None)
        return base.without_windows(stops, self.now)

    def _transition_late_recovery(self) -> None:
        """Once a car can no longer be on time, switch it to windowless patient fees."""
        if self.ends_at is not None and self.now >= self.ends_at:
            return
        for order_id, order in list(self.orders.items()):
            if order_id in self.late_recovery_orders or not base.undelivered(order):
                continue
            if self.now <= order["lastPickupAt"]:
                continue
            self.late_recovery_orders.add(order_id)
            self.urgent.discard(order_id)
            open_gigs = [gid for gid in self.gigs_of(order_id) if gid not in self.booked and gid in self.posts]
            if not open_gigs:
                continue
            if self._unplan_open_gigs(order_id):
                self.waiting.pop(order_id, None)
                self.post(order, base.undelivered(order))

    def _effective_reference_fee(self, order: dict[str, Any]) -> float:
        """Scale referenceFeeCents if order_changed moved pickup or delivery."""
        request = self.requests[order["id"]]
        orig_ref = float(request["referenceFeeCents"])
        if len(order["legs"]) != len(request["legs"]):
            return orig_ref
        changed = any(
            ol["pickup"]["placeId"] != rl["pickup"]["placeId"]
            or ol["delivery"]["placeId"] != rl["delivery"]["placeId"]
            for ol, rl in zip(order["legs"], request["legs"])
        )
        if not changed:
            return orig_ref
        def est_cents(legs: list[dict[str, Any]]) -> float:
            return sum(1632.0 + 25.66 * self._road(leg)[0] for leg in legs)
        orig_est = max(1.0, est_cents(request["legs"]))
        return max(orig_ref, orig_ref * (est_cents(order["legs"]) / orig_est))

    def fees_for(self, order: Any, cars: int) -> list[dict[str, int]]:
        typical = self._effective_reference_fee(order) * cars / max(1, len(order["legs"]))
        if order["id"] in self.late_recovery_orders:
            return self._late_recovery_schedule(typical)
        if order["id"] in self.urgent:
            return self._deadline_schedule(typical)
        private_pickup = any(leg["pickup"]["privateAddress"] for leg in order["legs"])
        private_mult = float(os.getenv("V5_PRIVATE_PICKUP_FEE_MULT", "1.10")) if private_pickup else 1.0
        return self._normal_fee_schedule(typical * private_mult, order["lastPickupAt"])

    def _normal_fee_schedule(self, typical: float, last_pickup: int) -> list[dict[str, int]]:
        """Opening multiplier now, hold 30 waking minutes, then rise to cap before pickup."""
        cap = float(os.getenv("V5_RAMP_CAP", "1.60"))
        num_steps = int(os.getenv("V5_RAMP_STEPS", "8"))
        power = float(os.getenv("V5_RAMP_POWER", "1.0"))
        opening_mult = float(os.getenv("V5_OPENING_MULT", "0.90"))
        high_slack_mult = os.getenv("V5_HIGH_SLACK_OPENING_MULT", "")
        if high_slack_mult and self._waking_minutes(self.now, last_pickup) >= 12 * HOUR:
            opening_mult = float(high_slack_mult)
        opening = max(1, int(opening_mult * typical + 0.5))
        schedule = [{"at": self.now, "feeCents": opening}]
        ramp_start = self._advance_waking_minutes(self.now, 30)
        ramp_end = self._rewind_waking_minutes(last_pickup, 200)
        waking_span = self._waking_minutes(ramp_start, ramp_end)
        if ramp_end <= ramp_start or waking_span <= 0:
            if last_pickup > ramp_start > self.now:
                schedule.append({"at": ramp_start, "feeCents": max(1, int(cap * typical + 0.5))})
            return schedule
        steps = min(num_steps, waking_span)
        step1_floor = os.getenv("V5_STEP1_FLOOR_MULT", "")
        for step in range(1, steps + 1):
            at = self._advance_waking_minutes(ramp_start, max(1, (waking_span * step + steps - 1) // steps))
            if at <= schedule[-1]["at"] or at > ramp_end:
                continue
            progress = (step / steps) ** power
            multiplier = opening_mult + (cap - opening_mult) * progress
            if step == 1 and step1_floor:
                multiplier = max(multiplier, float(step1_floor))
            schedule.append({"at": at, "feeCents": max(1, int(typical * multiplier + 0.5))})
        return schedule

    def _deadline_schedule(self, typical: float) -> list[dict[str, int]]:
        """On deadline_close: jump to 1.6×, then +0.25× every 20 minutes up to 3.0×."""
        return [
            {"at": self.now + 20 * step,
             "feeCents": max(1, int(typical * min(3.00, 1.60 + 0.25 * step) + 0.5))}
            for step in range(7)
        ]

    def _late_recovery_schedule(self, typical: float) -> list[dict[str, int]]:
        """Once late, restart at 1.05× (or configured mult) and add 0.10× every 2 waking hours up to 3.0×."""
        restart = float(os.getenv("V5_LATE_RESTART_MULT", "1.05"))
        return [
            {"at": self._advance_waking_minutes(self.now, 120 * step),
             "feeCents": max(1, int(typical * min(3.00, restart + 0.10 * step) + 0.5))}
            for step in range(20)
        ]

    def _booking_before_warning(self, order: dict[str, Any]) -> float | None:
        if self.fleet_booking_model is None:
            return None
        leg = order["legs"][0]
        features = offline_data_contract.post_time_features(
            last_pickup_at=order["lastPickupAt"], decision_at=self.now,
            pickup_private=leg["pickup"]["privateAddress"], delivery_private=leg["delivery"]["privateAddress"],
            premium=leg["car"]["premium"], electric=leg["car"]["electric"],
            legs=len(order["legs"]),
        )
        return booking_model.predict(self.fleet_booking_model, features)

    def _bundle_schedule(self, order_ids: list[str], urgent: bool) -> list[dict[str, int]]:
        references = sum(self.requests[order_id]["referenceFeeCents"] for order_id in order_ids)
        latest_pickup = min(self.orders[order_id]["lastPickupAt"] for order_id in order_ids)
        if urgent:
            return self._deadline_schedule(references)
        bundle_min = os.getenv("V5_BUNDLE_FEE_MIN", "0.75")
        if bundle_min:
            lo = float(bundle_min)
            hi = float(os.getenv("V5_BUNDLE_FEE_MAX", "0.90"))
            schedule = [{"at": self.now, "feeCents": max(1, int(lo * references + 0.5))}]
            ramp_start = self._advance_waking_minutes(self.now, 30)
            patience = int(os.getenv("V5_PAIR_PATIENCE_MINUTES", "45"))
            ramp_end = min(self._rewind_waking_minutes(latest_pickup, 200), self.now + patience)
            waking_span = self._waking_minutes(ramp_start, ramp_end)
            if ramp_end <= ramp_start or waking_span <= 0:
                if ramp_start > self.now:
                    schedule.append({"at": ramp_start, "feeCents": max(1, int(hi * references + 0.5))})
                return schedule
            steps = min(4, waking_span)
            for step in range(1, steps + 1):
                at = self._advance_waking_minutes(ramp_start, max(1, (waking_span * step + steps - 1) // steps))
                if at <= schedule[-1]["at"] or at > ramp_end:
                    continue
                mult = lo + (hi - lo) * (step / steps)
                schedule.append({"at": at, "feeCents": max(1, int(references * mult + 0.5))})
            return schedule
        return self._normal_fee_schedule(references, latest_pickup)

    def _reopen_after_drop(self, gig_id: str, at: int) -> None:
        """A drop resets driver expectations at 110% of the visible fee."""
        action = self.posts.get(gig_id)
        if action is None:
            return
        schedule = action.get("schedule", [])
        visible = next((step["feeCents"] for step in reversed(schedule) if step["at"] <= at), None)
        if visible is None:
            return
        reopen_fee = max(1, int(visible * 1.10 + 0.5))
        new_schedule = [{"at": at, "feeCents": reopen_fee}]
        for step in schedule:
            if step["at"] > at:
                new_schedule.append({"at": step["at"], "feeCents": max(reopen_fee, step["feeCents"])})
        action["schedule"] = new_schedule
        self.outbox.append({"type": "reprice", "gigId": gig_id, "schedule": new_schedule})

    def _backlog_fee_factor(self) -> float:
        """Pay a credible opening premium when the live driver market is busy."""
        open_gigs = sum(gig_id not in self.booked for gig_id in self.posts)
        target = max(1.0, float(os.getenv("V5_OPEN_GIG_TARGET", "30")))
        maximum = float(os.getenv("V5_BACKLOG_FEE_UPLIFT", "0.22"))
        return 1.0 + maximum * min(1.0, open_gigs / target)

    def _transport_cost(self, request: dict[str, Any], option: dict[str, Any], pressure: float,
                        late_probability: float) -> float:
        """Quote-time expected cost in cents under the level-8 policy."""
        typical = float(request["referenceFeeCents"])
        leg = request["legs"][0]
        facts = self.note_facts.get(request["id"], note_constraints.NoteConstraints())
        base_mult = float(os.getenv("V5_BASE_FEE_MULT", "1.25"))
        mon_mult = os.getenv("V5_MONDAY_BASE_FEE_MULT", "")
        if mon_mult and self.now < DAY and late_probability < 0.08:
            base_mult = float(mon_mult)
        cost = base_mult * typical
        if option["kind"] == "same_day":
            cost += 0.10 * typical
        if leg["delivery"]["privateAddress"]:
            cost += 0.15 * typical
        if leg["pickup"]["privateAddress"]:
            cost += float(os.getenv("V5_PRIVATE_PICKUP_COST_MULT", "0.10")) * typical
        if facts.has_fact:
            cost += 0.05 * typical
        # €0.75 for each unassigned order after the first 15 represents the
        # extra fee needed to keep the next marginal car from becoming a
        # deadline rescue.  It is intentionally a cost, not a hard car cap.
        backlog_rate = float(os.getenv("V5_BACKLOG_COST", "115"))
        free_orders = 15
        if os.getenv("V5_FLEET_BACKLOG", "0") == "1" and self.home_supply:
            fleet_ratio = 800.0 / max(400.0, float(sum(self.home_supply.values())))
            free_orders = max(8, int(round(15.0 / fleet_ratio)))
            backlog_rate *= fleet_ratio ** 1.5
        cost += backlog_rate * max(0, self._unassigned_orders() - free_orders)
        cost += late_probability * (6_000 + 0.56 * option["listPriceCents"])
        return cost

    def _context_fee_factor_for_request(self, request: dict[str, Any]) -> float:
        """Quote-side approximation of the reusable post-time fee context."""
        if not self._v52_fee_model():
            return 1.0
        leg = request["legs"][0]
        route = self._road(leg)[1]
        slack = max(0, max(option["deadline"] for option in request["options"]) - self.now)
        context = (leg["pickup"]["placeId"], self.now // (6 * HOUR), min(3, route // 120),
                   min(3, slack // (6 * HOUR)), 0)
        return math.exp(self.fee_context_adjustment[context])

    def _route_risk(self, request: dict[str, Any], option: dict[str, Any] | None) -> float:
        if option is None:
            return 3.0
        longest = max((self._road(leg)[1] for leg in request["legs"]), default=0)
        hours_left = max(0.0, (option["deadline"] - self.now) / HOUR)
        score = max(0.0, (longest / HOUR + 7.0 - hours_left) / 4.0)
        score += max(0.0, (longest / HOUR - 4.0) / 3.0)
        score += 0.5 * any(leg["pickup"]["privateAddress"] or leg["delivery"]["privateAddress"] for leg in request["legs"])
        score += 0.4 * (len(request["legs"]) > 1)
        places = {place["placeId"] for leg in request["legs"] for place in (leg["pickup"], leg["delivery"])}
        if any(forecast["from"] <= option["deadline"] and forecast["to"] >= self.now and places & forecast["placeIds"] for forecast in self.forecasts):
            score += 0.55
        return score

    def _won_option(self, order_id: str) -> dict[str, Any] | None:
        order = self.orders.get(order_id)
        request = self.requests.get(order_id)
        if order is None or request is None:
            return None
        return next((option for option in request["options"] if option["kind"] == order["option"]), None)

    @staticmethod
    def _lane(request: dict[str, Any]) -> tuple[str, str]:
        leg = request["legs"][0]
        return leg["pickup"]["placeId"], leg["delivery"]["placeId"]

    def _road(self, leg: dict[str, Any]) -> tuple[float, int]:
        return self.roads.get((leg["pickup"]["placeId"], leg["delivery"]["placeId"]), (0.0, 0))


def _next_planner_wake(desk: PortfolioDesk, upper: int) -> int:
    """Wake the practice loop for pool, late-recovery, and second-look transitions."""
    due = [upper]
    hold = int(os.getenv("V5_POOL_MINUTES", "0"))
    if hold > 0:
        due.extend(queued_at + hold for queued_at in desk.waiting.values())
    pair_patience = int(os.getenv("V5_PAIR_PATIENCE_MINUTES", "45"))
    second_look = int(os.getenv("V5_SECOND_LOOK_MINUTES", "0"))
    for gig_id, action in desk.posts.items():
        posted = desk.posted_at.get(gig_id)
        if posted is None or gig_id in desk.booked:
            continue
        is_chain = len(desk._action_orders(action)) > 1
        interval = pair_patience if is_chain else second_look
        if interval > 0:
            due.append(posted + interval)
    for order_id, order in desk.orders.items():
        if order_id not in desk.late_recovery_orders and base.undelivered(order):
            if any(gid not in desk.booked and gid in desk.posts for gid in desk.gigs_of(order_id)):
                due.append(order["lastPickupAt"] + 1)
    future = [minute for minute in due if minute > desk.now]
    return min(future, default=upper)


def play_practice(world: Any, level: int, quote_multiplier: float = 1.0,
                  fee_multiplier: float = 1.0, max_exposure: int | None = 8,
                  option_policy: str = "exclude_same_day") -> None:
    """Practice loop that also wakes when V5's planner clock is due."""
    snapshot = world.snapshot()
    desk = PortfolioDesk(world, level, snapshot["now"], quote_multiplier, fee_multiplier, max_exposure, option_policy)
    ends_at = snapshot["endsAt"]
    desk.ends_at = ends_at
    while desk.now < ends_at:
        day, midnight = desk.now // DAY, (desk.now // DAY + 1) * DAY
        until = _next_planner_wake(desk, min(ends_at, midnight))
        actions = desk._drain_outbox()
        step = world.advance(until=base.WAKE_ON, max=until, actions=actions)
        desk.read_results(actions, step["results"])
        desk.now = step["now"]
        for event in step["events"]:
            desk.handle(event)
        # An advance can finish at a planned timer without an event.
        desk._transition_late_recovery()
        desk._take_second_look()
        desk._plan_waiting()
        if desk.now // DAY > day:
            base.report(base.WEEKDAYS[day % 7], world.score())
    print("V5_TELEMETRY " + json.dumps({
        "chainPosts": desk.chain_posts,
        "chainCars": desk.chain_cars,
        "singlePosts": desk.single_posts,
        "completedOrders": desk.completed_orders,
        "lateOrders": desk.late_orders,
        "quoteRequests": desk.quote_requests,
        "optionsSeen": desk.options_seen,
        "optionsOffered": desk.options_offered,
        "declinedRequests": desk.declined_requests,
        "rejectedOptions": dict(sorted(desk.rejected_options.items())),
    }, separators=(",", ":")), flush=True)


def play_final(world: Any, quote_multiplier: float = 1.0, fee_multiplier: float = 1.0,
               max_exposure: int | None = 8, option_policy: str = "exclude_same_day") -> None:
    """Live-clock V5 loop with planner wakes and restart reconstruction.

    The kit client enforces the final's action bucket (20 actions per second
    after it has observed the server's rate headers).  We therefore keep this
    loop small: quote/post decisions are made immediately; timer-driven fee
    changes run on idle stream ticks without a polling burst.
    """
    history = world.events()
    snapshot = world.snapshot()
    if snapshot["now"] >= snapshot["endsAt"]:
        return
    level = next((item["level"] for item in world.client.worlds() if item["worldId"] == world.id), 8)
    desk = PortfolioDesk(world, level, snapshot["now"], quote_multiplier, fee_multiplier,
                          max_exposure, option_policy)
    desk.rejoin(snapshot, history)
    desk.send()
    if desk.waiting:
        desk._plan_waiting()
        desk.send()
    after = history[-1]["seq"] if history else -1
    for event in world.stream(after=after, idle=1.0):
        previous_day = desk.now // DAY
        desk.now = world.client.final()["now"] if event is None else max(desk.now, event["at"])
        if event is not None:
            desk.handle(event)
        # The scheduler has meaningful work even when the event stream is
        # quiet: an opening hold, fee rise, pairing patience, or late recovery.
        desk._transition_late_recovery()
        desk._take_second_look()
        desk._plan_waiting()
        desk.send()
        if desk.waiting:
            desk._plan_waiting()
            desk.send()
        if desk.now // DAY > previous_day:
            base.report(base.WEEKDAYS[previous_day % 7], world.score())
        if desk.now >= snapshot["endsAt"]:
            return


base.Desk = PortfolioDesk
base.play_practice = play_practice
base.play_final = play_final

if __name__ == "__main__":
    base.main()
