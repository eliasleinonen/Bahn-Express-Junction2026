"""Offline checks for V5's portfolio quote and route-chain planner."""
import math
import os
from types import SimpleNamespace

import dispatch
import note_constraints


def desk() -> dispatch.PortfolioDesk:
    roads = [
        {"from": "A", "to": "B", "km": 120, "minutes": 110},
        {"from": "B", "to": "C", "km": 18, "minutes": 30},
        {"from": "C", "to": "D", "km": 90, "minutes": 100},
    ]
    world = SimpleNamespace(client=SimpleNamespace(map=lambda: {"roads": roads}))
    return dispatch.PortfolioDesk(world, 8, 8 * 60, 1.60, 0.80, 28, "exclude_same_day")


def request(order_id: str, pickup: str, delivery: str, *, pickup_from: int,
            delivery_from: int, deadline: int) -> dict:
    latest_deadline = deadline + 24 * 60
    leg = {
        "pickup": {"placeId": pickup, "privateAddress": False, "window": {"from": pickup_from, "to": latest_deadline}},
        "delivery": {"placeId": delivery, "privateAddress": False, "window": {"from": delivery_from, "to": latest_deadline}},
        "car": {"premium": False, "electric": False, "chargePercent": None},
    }
    return {
        "id": order_id, "dealerId": "dealer", "arrivedAt": 8 * 60, "respondBy": 8 * 60 + 15,
        "referenceFeeCents": 10_000, "note": None, "legs": [leg],
        "options": [
            {"kind": "same_day", "deadline": deadline, "listPriceCents": 25_000},
            {"kind": "next_day", "deadline": latest_deadline, "listPriceCents": 21_000},
        ],
    }


def order(source: dict, *, last_pickup: int) -> dict:
    return {
        "id": source["id"], "option": "next_day", "priceCents": 34_000,
        "deadline": source["options"][1]["deadline"], "lastPickupAt": last_pickup,
        "legs": source["legs"], "deliveredAt": [None],
    }


def main() -> None:
    policy = desk()
    quoted = request("quote", "A", "B", pickup_from=600, delivery_from=600, deadline=1_200)
    policy.quote(quoted)
    # V5 intentionally prices every viable option; the portfolio model, rather
    # than a blanket same-day exclusion, decides which one has a risk premium.
    assert set(policy.outbox[-1]["prices"]) == {"same_day", "next_day"}

    # The new pricer searches the stated 0.80–3.00× list range.  Cost appears
    # in expected margin, not as an artificial price floor.
    safe_option, tight_option = quoted["options"][1], quoted["options"][0]
    plain = desk()
    safe_terms = plain._option_terms(quoted, safe_option, plain._route_risk(quoted, safe_option), 0.0)
    tight_terms = plain._option_terms(quoted, tight_option, plain._route_risk(quoted, tight_option), 0.0)
    assert safe_terms[2:4] == (safe_option["listPriceCents"] * 0.80, safe_option["listPriceCents"] * 3.00)
    assert tight_terms[2:4] == (tight_option["listPriceCents"] * 0.80, tight_option["listPriceCents"] * 3.00)

    # A tight same-day option must not make us decline the safe next-day sale.
    # The same-day option has no viable pickup window, whereas the slower
    # option clears the explicit last-pickup and daytime-discovery gates.
    split = request("split", "A", "B", pickup_from=510, delivery_from=510, deadline=600)
    policy.outbox.clear()
    policy.quote(split)
    assert set(policy.outbox[-1]["prices"]) == {"next_day"}
    assert policy.outbox[-1]["prices"]["next_day"] < split["options"][1]["listPriceCents"] * 5
    assert policy.rejected_options["route_window"] == 1
    assert policy.quote_requests == 2 and policy.options_seen == 4 and policy.options_offered == 3

    # Strict admission never leaves a request unanswered.  Trade-ins,
    # premium cars, and ambiguous notes get the harmless six-times-list
    # decline quote before they can spend reliability budget.
    unsafe_cases = {
        "trade": {"legs": [quoted["legs"][0], quoted["legs"][0]]},
        "premium": {"legs": [{**quoted["legs"][0], "car": {**quoted["legs"][0]["car"], "premium": True}}]},
        "note": {"note": {"text": "pickup actually at 12"}},
    }
    for case_id, changes in unsafe_cases.items():
        candidate = {**quoted, "id": case_id, **changes}
        policy.quote(candidate)
        quoted_decline = policy.outbox[-1]["prices"]
        assert list(quoted_decline.values()) == [candidate["options"][1]["listPriceCents"] * 6]
    assert policy.rejected_options["trade_in"] >= 2
    assert policy.rejected_options["premium"] >= 2
    assert policy.rejected_options["ambiguous_note"] >= 2

    # Private pickups are admitted with +0.10x transport cost and 1.10x posting fee.
    priv_req = {**quoted, "id": "priv-ok", "legs": [{**quoted["legs"][0], "pickup": {**quoted["legs"][0]["pickup"], "privateAddress": True}}]}
    policy.quote(priv_req)
    assert "priv-ok" in policy.quote_book
    priv_ord = order(priv_req, last_pickup=3 * 24 * 60 + 12 * 60)
    policy.orders["priv-ok"] = priv_ord
    assert policy.fees_for(priv_ord, 1)[0]["feeCents"] == int(0.90 * priv_req["referenceFeeCents"] * 1.10 + 0.5)

    # Restart recovery rebuilds the market posterior from the event stream;
    # no stale quote, reservation, or outbox action is required to continue.
    restarted = desk()
    restarted.rejoin(
        {"now": 500, "gigs": [], "orders": [], "openRequests": []},
        [
            {"kind": "price_request", "at": 480, "request": quoted},
            {"kind": "order_lost", "at": 495, "requestId": "quote"},
        ],
    )
    assert restarted.market_log < __import__("math").log(1.36)
    assert not restarted.outbox and not restarted.resource_reservations

    # The normal fee path holds 0.9×, takes eight even waking-time steps to
    # 1.6×, and switches to the documented rescue/recovery schedules.
    normal = policy._normal_fee_schedule(10_000, 3 * 24 * 60 + 12 * 60)
    assert len(normal) == 9 and normal[0]["feeCents"] == 9_000 and normal[-1]["feeCents"] == 16_000
    assert normal == sorted(normal, key=lambda step: step["at"])
    assert [step["feeCents"] for step in policy._deadline_schedule(10_000)] == [
        16_000, 18_500, 21_000, 23_500, 26_000, 28_500, 30_000
    ]
    late_sched = [step["feeCents"] for step in policy._late_recovery_schedule(10_000)]
    assert late_sched[0] == 10_500 and late_sched[1] == 11_500 and late_sched[-1] == 29_500
    assert len(late_sched) == 20

    # The telemetry keeps the late-risk and resource gates separate.  Relax the
    # late-risk threshold only for this isolated check, then saturate the local
    # resource so the rejection is attributed to congestion instead.
    previous_target = os.environ.get("V5_LATE_TARGET")
    previous_pressure = os.environ.get("V5_RESOURCE_PRESSURE_LIMIT")
    os.environ["V5_LATE_TARGET"] = "0.40"
    os.environ["V5_RESOURCE_PRESSURE_LIMIT"] = "1.0"
    try:
        congested = desk()
        congested_request = request("congested", "A", "B", pickup_from=600, delivery_from=600, deadline=1_200)
        bucket = next(iter(congested._resource_profile(congested_request)))
        congested.resource_reservations["busy"] = {"claim": 1.0, "profile": {bucket: 100.0}}
        congested.quote(congested_request)
        assert congested.rejected_options["resource_pressure"] == 2
        assert congested.declined_requests == 1
    finally:
        if previous_target is None:
            os.environ.pop("V5_LATE_TARGET", None)
        else:
            os.environ["V5_LATE_TARGET"] = previous_target
        if previous_pressure is None:
            os.environ.pop("V5_RESOURCE_PRESSURE_LIMIT", None)
        else:
            os.environ["V5_RESOURCE_PRESSURE_LIMIT"] = previous_pressure
    # A menu loss is evidence that the competitor won the entire menu, not
    # evidence that whichever option happened to be first in the request lost.
    market_before = policy.market_log
    policy.handle({"kind": "order_lost", "requestId": "quote"})
    assert policy.market_log < market_before and "quote" not in policy.quote_book

    # Selecting the fast option teaches speed preference separately from the
    # market threshold in split-residual mode.  This only moves after an actual selection.
    os.environ["V5_MARKET_MODE"] = "split"
    try:
        speed_request = request("speed", "A", "B", pickup_from=600, delivery_from=600, deadline=1_200)
        policy.quote(speed_request)
        quoted_claim = policy.resource_reservations["speed"]["claim"]
        assert 0.0 < quoted_claim < 1.0
        speed_before = policy.dealer_speed["dealer"]
        speed_order = order(speed_request, last_pickup=700)
        speed_order["option"] = "same_day"
        speed_order["deadline"] = speed_request["options"][0]["deadline"]
        policy.handle({"kind": "order_won", "order": speed_order})
        assert policy.dealer_speed["dealer"] > speed_before
        assert policy.resource_reservations["speed"]["claim"] == 1.0
        policy.waiting.pop("speed", None)
        policy.orders.pop("speed", None)
    finally:
        os.environ.pop("V5_MARKET_MODE", None)

    first_request = request("one", "A", "B", pickup_from=600, delivery_from=720, deadline=1_400)
    second_request = request("two", "C", "D", pickup_from=700, delivery_from=700, deadline=1_400)
    # Local resource claims are weighted by route time and reserve the pickup,
    # its arrival bucket, and the delivery continuation—not a global car count.
    profile = policy._resource_profile(first_request)
    assert len(profile) == 3 and max(profile.values()) > 1.0
    first, second = order(first_request, last_pickup=700), order(second_request, last_pickup=900)
    policy.requests.update({"one": first_request, "two": second_request})
    policy.orders.update({"one": first, "two": second})
    policy.waiting.update({"one": policy.now, "two": policy.now})
    os.environ["V5_ENABLE_CHAINING"] = "1"
    try:
        policy._plan_waiting()
    finally:
        os.environ.pop("V5_ENABLE_CHAINING", None)
    chain = policy.outbox[-1]
    assert chain["type"] == "post"
    assert [stop["orderId"] for stop in chain["stops"] if stop["kind"] == "pickup"] == ["one", "two"]
    assert chain["schedule"][0]["feeCents"] < chain["schedule"][-1]["feeCents"]

    # A late delivered event can leave a stale pool entry; it must never turn
    # into an invalid zero-stop post.
    policy.outbox.clear()
    policy.orders["one"]["deliveredAt"] = [policy.now]
    policy.waiting["one"] = policy.now - 30
    policy._plan_waiting()
    assert not policy.outbox

    # Delivery evidence updates V5's fleet reliability belief only after the
    # core has marked the final leg delivered.
    delivered = order(first_request, last_pickup=700)
    delivered["deadline"] = policy.now - 1
    policy.orders["done"] = {**delivered, "id": "done"}
    policy.requests["done"] = {**first_request, "id": "done"}
    policy.reserved_late["done"] = 0.03
    policy.handle({"kind": "delivered", "orderId": "done", "leg": 0, "at": policy.now})
    assert policy.completed_orders == 1 and policy.late_orders == 1

    # Trust is observable state, separate from the residual demand learner:
    # late work halves it, then each fully on-time order earns one point.
    policy.dealer_trust["dealer"] = 8
    late = order(first_request, last_pickup=700)
    late["id"], late["deadline"] = "trust-late", policy.now - 1
    policy.orders["trust-late"] = late
    policy.requests["trust-late"] = {**first_request, "id": "trust-late"}
    policy.handle({"kind": "delivered", "orderId": "trust-late", "leg": 0, "at": policy.now})
    assert policy.dealer_trust["dealer"] == 4
    on_time = order(first_request, last_pickup=700)
    on_time["id"], on_time["deadline"] = "trust-on-time", policy.now + 1
    policy.orders["trust-on-time"] = on_time
    policy.requests["trust-on-time"] = {**first_request, "id": "trust-on-time"}
    policy.handle({"kind": "delivered", "orderId": "trust-on-time", "leg": 0, "at": policy.now})
    assert policy.dealer_trust["dealer"] == 5

    # Dropping a gig raises the visible fee 10% while keeping future steps non-decreasing.
    drop_desk = desk()
    drop_desk.posts["g1"] = {
        "type": "post", "stops": [],
        "schedule": [{"at": 400, "feeCents": 10_000}, {"at": 600, "feeCents": 10_500}, {"at": 700, "feeCents": 13_000}],
    }
    drop_desk._reopen_after_drop("g1", 500)
    reprice = drop_desk.outbox[-1]
    assert reprice["type"] == "reprice" and reprice["schedule"] == [
        {"at": 500, "feeCents": 11_000}, {"at": 600, "feeCents": 11_000}, {"at": 700, "feeCents": 13_000},
    ]

    # Passing lastPickupAt automatically moves an unbooked open gig onto the
    # windowless 1.05x..3.0x late recovery schedule.
    late_desk = desk()
    req_late = request("late-auto", "A", "B", pickup_from=400, delivery_from=400, deadline=1_600)
    ord_late = order(req_late, last_pickup=550)
    late_desk.requests["late-auto"] = req_late
    late_desk.orders["late-auto"] = ord_late
    late_desk.note_facts["late-auto"] = note_constraints.parse_note("Manual gearbox", 0)
    assert late_desk.note_facts["late-auto"].has_fact
    cost_with_fact = late_desk._transport_cost(req_late, req_late["options"][1], 0.0, 0.0)
    late_desk.note_facts["late-auto"] = note_constraints.NoteConstraints()
    cost_plain = late_desk._transport_cost(req_late, req_late["options"][1], 0.0, 0.0)
    assert abs((cost_with_fact - cost_plain) - 0.05 * req_late["referenceFeeCents"]) < 1e-6
    late_desk.posts["g-open"] = {
        "type": "post",
        "stops": [{"orderId": "late-auto", "leg": 0, "kind": "pickup", "window": {"from": 400, "to": 600}},
                  {"orderId": "late-auto", "leg": 0, "kind": "delivery", "window": {"from": 400, "to": 1600}}],
        "schedule": [{"at": 500, "feeCents": 16_000}],
    }
    late_desk.leg_gigs["late-auto"] = {0: "g-open"}
    late_desk.now = 551
    late_desk._transition_late_recovery()
    assert "late-auto" in late_desk.late_recovery_orders
    assert [a["type"] for a in late_desk.outbox] == ["withdraw", "post"]
    reposted = late_desk.outbox[1]
    assert all("window" not in s for s in reposted["stops"])
    assert reposted["schedule"][0]["feeCents"] == int(1.05 * req_late["referenceFeeCents"] + 0.5)

    # Rejoin restores market learning BEFORE answering missed openRequests.
    hist_req = request("lost-prior", "A", "B", pickup_from=600, delivery_from=600, deadline=2_000)
    open_req = request("missed-open", "A", "B", pickup_from=600, delivery_from=600, deadline=2_000)
    baseline_open = desk()
    baseline_open.now = 520
    baseline_open.quote(open_req)
    rejoined_open = desk()
    rejoined_open.rejoin(
        {"now": 520, "gigs": [], "orders": [], "openRequests": [open_req]},
        [{"kind": "price_request", "at": 480, "request": hist_req},
         {"kind": "order_lost", "at": 495, "requestId": "lost-prior"}],
    )
    assert rejoined_open.outbox[-1]["type"] == "quote"
    assert rejoined_open.outbox[-1]["prices"]["next_day"] < baseline_open.outbox[-1]["prices"]["next_day"]

    # Delivery opening hours from dealer note tighten last_open_minute.
    hrs_req = request("hrs", "A", "B", pickup_from=500, delivery_from=500, deadline=1_500)
    hrs_facts = note_constraints.parse_note("Delivery only 8-14", 0)
    plan_same, reason_same = policy._on_time_plan(hrs_req, hrs_req["options"][0], hrs_facts)
    plan_next, _ = policy._on_time_plan(hrs_req, hrs_req["options"][1], hrs_facts)
    assert plan_same is None and reason_same == "driver_discovery"
    assert plan_next is not None and plan_next["latestDelivery"] == 1440 + 14 * 60

    # order_changed moving a delivery to a farther city scales referenceFeeCents up,
    # and _drain_outbox drops reprices for already-booked gigs.
    changed_desk = desk()
    changed_desk.roads[("A", "D")] = (240.0, 220)
    short_req = request("moved", "B", "C", pickup_from=600, delivery_from=600, deadline=2_000)
    short_req["referenceFeeCents"] = 2_000
    changed_desk.requests["moved"] = short_req
    unchanged_order = order(short_req, last_pickup=900)
    assert changed_desk._effective_reference_fee(unchanged_order) == 2_000.0
    longer_order = {
        **unchanged_order,
        "legs": [{
            **unchanged_order["legs"][0],
            "pickup": {**unchanged_order["legs"][0]["pickup"], "placeId": "A"},
            "delivery": {**unchanged_order["legs"][0]["delivery"], "placeId": "D"},
        }],
    }
    scaled_fee = changed_desk._effective_reference_fee(longer_order)
    assert scaled_fee > 7_000.0
    changed_desk.posts["g-open"] = {"type": "post", "stops": [], "schedule": [{"at": 480, "feeCents": 2_000}]}
    changed_desk.posts["g-booked"] = {"type": "post", "stops": [], "schedule": [{"at": 480, "feeCents": 2_000}]}
    changed_desk.booked.add("g-booked")
    changed_desk.outbox = [
        {"type": "reprice", "gigId": "g-open", "schedule": [{"at": 480, "feeCents": 2_500}]},
        {"type": "reprice", "gigId": "g-booked", "schedule": [{"at": 480, "feeCents": 2_500}]},
    ]
    assert [a["gigId"] for a in changed_desk._drain_outbox()] == ["g-open"]

    # Restart trap: never count unquoted order_lost during an outage or late start.
    outage_history = []
    for idx in range(15):
        req_miss = request(f"miss-{idx}", "A", "B", pickup_from=600 + idx, delivery_from=600 + idx, deadline=2_000 + idx)
        outage_history.append({"kind": "price_request", "at": 480 + idx, "request": req_miss})
        outage_history.append({"kind": "order_lost", "at": 495 + idx, "requestId": f"miss-{idx}"})
    stateless_rejoin = desk()
    stateless_rejoin.rejoin({"now": 520, "gigs": [], "orders": [], "openRequests": []}, outage_history)
    assert abs(stateless_rejoin.market_log - __import__("math").log(1.36)) < 1e-9

    # Persisted quote journal: only requests confirmed ok=True are replayed as lost quotes.
    from pathlib import Path
    journal_id = "test-outage-journal"
    journal_path = Path("data/runtime") / f"quotes_{journal_id}.jsonl"
    if journal_path.exists():
        journal_path.unlink()
    try:
        live_desk = desk()
        live_desk.world.id = journal_id
        quoted_req = request("real-q", "A", "B", pickup_from=600, delivery_from=600, deadline=2_000)
        unquoted_req = request("unquoted-q", "A", "B", pickup_from=600, delivery_from=600, deadline=2_000)
        live_desk.quote(quoted_req)
        live_desk.read_results([{"type": "quote", "requestId": "real-q", "prices": {}}], [{"ok": True, "id": "real-q"}])
        rejoined_journal = desk()
        rejoined_journal.world.id = journal_id
        rejoined_journal.rejoin(
            {"now": 530, "gigs": [], "orders": [], "openRequests": []},
            [
                {"kind": "price_request", "at": 480, "request": quoted_req},
                {"kind": "price_request", "at": 485, "request": unquoted_req},
                {"kind": "order_lost", "at": 500, "requestId": "unquoted-q"},
            ],
        )
        assert abs(rejoined_journal.market_log - __import__("math").log(1.36)) < 1e-9
    finally:
        if journal_path.exists():
            journal_path.unlink()

    # Failed post / failed withdraw recovery keeps orders from being stranded in waiting.
    fail_desk = desk()
    req_a = request("fail-a", "A", "B", pickup_from=600, delivery_from=600, deadline=1_500)
    req_b = request("fail-b", "A", "B", pickup_from=600, delivery_from=600, deadline=1_500)
    ord_a = order(req_a, last_pickup=1_200)
    ord_b = order(req_b, last_pickup=1_200)
    fail_desk.requests["fail-a"] = req_a
    fail_desk.requests["fail-b"] = req_b
    fail_desk.orders["fail-a"] = ord_a
    fail_desk.orders["fail-b"] = ord_b
    fail_desk.now = 700
    bundle_post = {
        "type": "post",
        "stops": [
            {"orderId": "fail-a", "leg": 0, "kind": "pickup", "window": {"from": 600, "to": 1_200}},
            {"orderId": "fail-b", "leg": 0, "kind": "pickup", "window": {"from": 600, "to": 1_200}},
            {"orderId": "fail-a", "leg": 0, "kind": "delivery"},
            {"orderId": "fail-b", "leg": 0, "kind": "delivery"},
        ],
        "schedule": [{"at": 700, "feeCents": 2_500}],
    }
    fail_desk.read_results([bundle_post], [{"ok": False, "error": "too_late"}])
    assert {"fail-a", "fail-b"}.issubset(fail_desk.pair_failed_orders)
    assert "fail-a" in fail_desk.waiting and "fail-b" in fail_desk.waiting

    # Note-hours gig expiry before lastPickupAt still falls through to windowless late recovery.
    exp_desk = desk()
    req_hrs_exp = request("hrs-exp", "A", "B", pickup_from=1860, delivery_from=1860, deadline=2640)
    req_hrs_exp["note"] = {"text": "Receiving Mon-Fri 10:00 - 17:00 only.", "language": "en"}
    ord_hrs_exp = order(req_hrs_exp, last_pickup=2519)
    ord_hrs_exp["legs"][0]["pickup"]["window"]["to"] = 2640
    ord_hrs_exp["legs"][0]["delivery"]["window"]["to"] = 2640
    exp_desk.requests["hrs-exp"] = req_hrs_exp
    exp_desk.orders["hrs-exp"] = ord_hrs_exp
    exp_desk.note_facts["hrs-exp"] = note_constraints.parse_note(req_hrs_exp["note"]["text"], 0)
    exp_desk.now = 1860
    exp_desk.post(ord_hrs_exp, [0])
    posted_exp = exp_desk.outbox.pop(0)
    exp_desk.carry(posted_exp, "g-exp")
    exp_desk.now = 2461
    exp_desk.repost("g-exp")
    assert "hrs-exp" in exp_desk.late_recovery_orders
    assert len(exp_desk.outbox) == 1 and exp_desk.outbox[0]["type"] == "post"
    assert all("window" not in s for s in exp_desk.outbox[0]["stops"])

    windowed_post = {
        "type": "post",
        "stops": [
            {"orderId": "fail-a", "leg": 0, "kind": "pickup", "window": {"from": 600, "to": 1_200}},
            {"orderId": "fail-a", "leg": 0, "kind": "delivery"},
        ],
        "schedule": [{"at": 700, "feeCents": 2_000}],
    }
    fail_desk.waiting.clear()
    fail_desk.read_results([windowed_post], [{"ok": False, "error": "bad_window"}])
    assert "fail-a" in fail_desk.late_recovery_orders and "fail-a" in fail_desk.waiting

    fail_desk.carry(windowed_post, "g-keep")
    assert fail_desk._unplan_open_gigs("fail-a") is True
    assert "g-keep" not in fail_desk.posts
    withdraw_action = [a for a in fail_desk.outbox if a["type"] == "withdraw"][0]
    fail_desk.outbox.clear()
    fail_desk.read_results([withdraw_action], [{"ok": False, "error": "already_booked"}])
    assert "g-keep" in fail_desk.posts
    assert fail_desk.gigs_of("fail-a") == ["g-keep"]

    # Daily note pickup hours keep single-day [opens, closes] across reposts and late recovery.
    from note_constraints import parse_note
    hours_desk = desk()
    hours_req = request("hrs-1", "A", "B", pickup_from=687, delivery_from=687, deadline=3_500)
    hours_req["legs"][0]["pickup"]["window"] = {"from": 687, "to": 3_000}
    hours_req["legs"][0]["pickup"]["hours"] = {"opens": 480, "closes": 1140, "weekdays": [0, 1, 2, 3, 4], "saturday": None}
    hours_req["note"] = {"text": "auto valims, nouto 10.00-15.00", "language": "fi"}
    hours_desk.requests["hrs-1"] = hours_req
    hours_desk.note_facts["hrs-1"] = parse_note(hours_req["note"]["text"], 687)
    hours_ord = order(hours_req, last_pickup=2_500)
    hours_desk.orders["hrs-1"] = hours_ord
    hours_desk.now = 687
    hours_desk.post(hours_ord, [0])
    day0_post = hours_desk.outbox.pop()
    assert day0_post["stops"][0]["window"] == {"from": 687, "to": 900}
    hours_desk.carry(day0_post, "g-hrs-0")
    hours_desk.now = 900
    hours_desk.repost("g-hrs-0")
    day1_post = hours_desk.outbox.pop()
    assert "hrs-1" not in hours_desk.late_recovery_orders
    assert day1_post["stops"][0]["window"] == {"from": 1440 + 600, "to": 1440 + 900}
    hours_desk.late_recovery_orders.add("hrs-1")
    hours_desk.now = 2 * 1440 + 480
    hours_desk.post(hours_ord, [0])
    late_post = hours_desk.outbox.pop()
    assert late_post["stops"][0]["window"] == {"from": 2 * 1440 + 600, "to": 2 * 1440 + 900}

    # Verify Dinkelbach menu pricing, opening mult hook, and harmless-note chaining flag.
    os.environ["V5_OPENING_MULT"] = "0.82"
    try:
        sched = hours_desk._normal_fee_schedule(10_000.0, 2 * 1440 + 1000)
        assert sched[0]["feeCents"] == 8_200
    finally:
        os.environ.pop("V5_OPENING_MULT", None)

    menu_terms = [
        {"option": {"kind": "same_day", "listPriceCents": 8_000}, "risk": 0.1,
         "threshold": math.log(1.36), "cost": 7_500.0, "speed": 1.0},
        {"option": {"kind": "next_day", "listPriceCents": 7_000}, "risk": 0.0,
         "threshold": math.log(1.36), "cost": 5_000.0, "speed": 0.0},
    ]
    menu_prices, menu_recs = hours_desk._price_menu(menu_terms)
    assert set(menu_prices) == {"same_day", "next_day"} and len(menu_recs) == 2

    note_chain_req = request("nc-1", "A", "B", pickup_from=0, delivery_from=0, deadline=1000)
    note_chain_req["note"] = {"text": "Ref 12345", "language": "en"}
    hours_desk.requests["nc-1"] = note_chain_req
    hours_desk.note_facts["nc-1"] = parse_note("Ref 12345", 0)
    nc_ord = order(note_chain_req, last_pickup=900)
    assert hours_desk._chainable(nc_ord) is False
    os.environ["V5_CHAIN_ALLOW_HARMLESS_NOTES"] = "1"
    try:
        assert hours_desk._chainable(nc_ord) is True
    finally:
        os.environ.pop("V5_CHAIN_ALLOW_HARMLESS_NOTES", None)

    # Verify snow delivery risk, fleet-scaled backlog, and multi-menu kalman flags.
    snow_desk = desk()
    snow_desk.forecasts.append({"from": 480, "to": 1_500, "placeIds": {"B"}})
    snow_req = request("sn-1", "A", "B", pickup_from=480, delivery_from=480, deadline=1_200)
    plan_off, _ = snow_desk._on_time_plan(snow_req, snow_req["options"][0], note_constraints.NoteConstraints())
    late_off = snow_desk._late_probability(snow_req, snow_req["options"][0], 0.0, plan_off)
    os.environ["V5_SNOW_DELIV_RISK"] = "1"
    try:
        plan_on, _ = snow_desk._on_time_plan(snow_req, snow_req["options"][0], note_constraints.NoteConstraints())
        late_on = snow_desk._late_probability(snow_req, snow_req["options"][0], 0.0, plan_on)
        assert plan_on is not None and plan_off is not None and plan_on["lastPickup"] < plan_off["lastPickup"]
        assert late_on > late_off
    finally:
        os.environ.pop("V5_SNOW_DELIV_RISK", None)

    snow_desk.home_supply = {"A": 300, "B": 300}  # 600 drivers (< 800)
    snow_desk.orders = {f"u-{i}": order(snow_req, last_pickup=900) | {"id": f"u-{i}"} for i in range(18)}
    snow_desk.requests.update({f"u-{i}": snow_req for i in range(18)})
    cost_fixed = snow_desk._transport_cost(snow_req, snow_req["options"][0], 0.0, 0.05)
    os.environ["V5_FLEET_BACKLOG"] = "1"
    try:
        cost_scaled = snow_desk._transport_cost(snow_req, snow_req["options"][0], 0.0, 0.05)
        assert cost_scaled > cost_fixed
    finally:
        os.environ.pop("V5_FLEET_BACKLOG", None)
    print("V5 offline checks passed")


if __name__ == "__main__":
    main()
