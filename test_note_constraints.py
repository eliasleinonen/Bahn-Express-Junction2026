"""Focused checks for conservative structured dealer-note parsing."""
from note_constraints import parse_note


def main() -> None:
    ready = parse_note("Auto valmis klo 13.30, nouto aikaisintaan klo 14", 2 * 1440 + 500)
    assert ready.pickup_after == 2 * 1440 + 14 * 60 and ready.status == "safe_timing" and ready.chainable
    delivery = parse_note("Deliver by 16:00", 0)
    assert delivery.delivery_before == 16 * 60 and delivery.status == "safe_timing"
    equipment = parse_note("Manuaalivaihteisto, autossa ei kilpiä", 0)
    assert equipment.manual_licence and equipment.with_plates and not equipment.chainable
    tyres = parse_note("Auto vielä kesärenkailla", 0)
    assert tyres.summer_tyres and not tyres.chainable
    unclear = parse_note("Nouto klo 12, ei kun klo 14", 0)
    assert unclear.status == "ambiguous" and not unclear.chainable

    corrected_time = parse_note("Nouto klo 12 jälkeen, ei kun klo 14", 0)
    assert corrected_time.pickup_after == 14 * 60 and corrected_time.status == "safe_timing"

    corrected_day = parse_note(
        "vastaanottava liike auki klo 9 - 21. toimitus perille viim. sunnuntaina klo 14, ei kun maanantaina!",
        6371,
    )
    assert corrected_day.status == "safe_timing"
    assert corrected_day.delivery_hours == (540, 1260, False)
    assert corrected_day.delivery_before == 10920

    swedish_hours = parse_note("Mottagaren öppet vardagar 7-19, leverans senast onsdag kl 16", 0)
    assert swedish_hours.delivery_hours == (420, 1140, True)
    assert swedish_hours.delivery_before == 2 * 1440 + 16 * 60

    neg_day = parse_note("Nouto klo 15 jälkeen, ei huomenna.", 500)
    assert neg_day.pickup_after == 15 * 60 and neg_day.status == "safe_timing"

    word_half = parse_note("Auto valmis huomenna puoli kolmelta", 500)
    assert word_half.pickup_after == 1440 + 14 * 60 + 30

    spare_boot = parse_note("4 kesärengasta takakontissa, menee mukana", 0)
    assert not spare_boot.summer_tyres and not spare_boot.has_fact

    both_tyres = parse_note("Auto vielä kesärenkailla, kesärenkaat vanteineen takakontissa", 0)
    assert both_tyres.summer_tyres and both_tyres.has_fact

    plates_typo = parse_note("Rekisterikilvet irotettu", 0)
    assert plates_typo.with_plates and plates_typo.has_fact

    manual_typo = parse_note("Huom manuaail", 0)
    assert manual_typo.manual_licence and manual_typo.has_fact

    en_dash_wd = parse_note("Pickup Mon–Fri 8am - 5pm only", 0)
    assert en_dash_wd.pickup_hours == (480, 1020, True)

    typo_arkisin = parse_note("Huom aukioloajat akrisin klo 10-14", 0)
    assert typo_arkisin.pickup_hours == (600, 840, True)

    typo_huomnena = parse_note("Noudettavissa aikaisintaan huomnena klo 13.", 0)
    assert typo_huomnena.pickup_after == 1440 + 780

    typo_arisin = parse_note("Nouto arisin 10 - 15.30.", 0)
    assert typo_arisin.pickup_hours == (600, 930, True)

    typo_valims_tommorrow = parse_note("Auto valims tommorrow klo 10", 0)
    assert typo_valims_tommorrow.status == "safe_timing" and typo_valims_tommorrow.pickup_after == 1440 + 600

    neg_after_close = parse_note("täällä ei ketään puoli viiden jälkeen", 0)
    assert neg_after_close.status == "safe_timing" and neg_after_close.pickup_hours == (None, 990, False)

    test_final_notes = [
        ("Noudettavissa aikaiintaan klo 15", "safe_timing"),
        ("Moikka. Kohteen aukioloajat ma-pe klo 7.00–20.00. Liike menee kiinni koolmelta", "safe_timing"),
        ("Moikka. Myyjä paikalla huomenna klo 8.00-11.00. Oltava perillä viimeistään keskiviikkona klo 20.00.", "safe_timing"),
        ("mottagning till kl 15, referens 95190, mvh försäljningen.", "safe_timing"),
        ("Mottagaren har öppet vardagar kl 8-21. Klar först i morgon kl 7, nej förlåt, kl 8", "safe_timing"),
        ("Auton voi hakea ma-pe 9 - 19. Auto valmistuu huomenna kahdeltatoista.", "safe_timing"),
        ("hi there, open 9:00 - 18:00", "safe_timing"),
        ("Nouto ennen puolta neljää. Kiitos.", "safe_timing"),
        ("Klar, kan hämtas direkt.", "harmless"),
    ]
    for txt, expected_status in test_final_notes:
        parsed = parse_note(txt, 600)
        assert parsed.status == expected_status, (txt, parsed)
    ma_pe = parse_note("Auton voi hakea ma-pe 9 - 19.", 600)
    assert ma_pe.pickup_hours == (540, 1140, True) and ma_pe.pickup_after is None

    harmless = parse_note("Kiitos! Yksi auto, soitathan ennen saapumista.", 0)
    assert harmless.status == "harmless" and not harmless.has_fact

    no_pickup_before = parse_note("Payment clears tmrw 11:30 am, no pickup before that. Second set of wheels in the back", 5000)
    assert no_pickup_before.status == "safe_timing" and no_pickup_before.pickup_after == 4 * 1440 + 690 and no_pickup_before.pickup_before is None
    print("note constraint checks passed")


if __name__ == "__main__":
    main()
