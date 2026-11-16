"""Conservative deterministic constraints from dealer notes.

Parses explicit Finnish, Swedish, and English operational facts (days, clock
times, word hours, ranges, corrections, negations, plates, manual gearbox,
summer tyres) and marks unrecognized or contradictory timing notes ambiguous so
callers can decline safely.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re


HOUR, DAY = 60, 24 * 60

# Equipment & vehicle flags -------------------------------------------------
_MANUAL = re.compile(
    r"ma+nua+i*l|mnuaali|manula|manaul|maunal|mauaali|k[äa]si?vaiht\w*|ma+nuell?|maneull|stick\s*shift"
)
_MANUAL_NEGATED = re.compile(
    r"(?:automaat|automatic|automat)[^.;!\n]{0,30}(?:ei|not|inte)\s+(?:a\s+)?ma+nua"
    r"|(?:ei|not|inte)\s+(?:a\s+)?ma+nua"
)

_PLATES = re.compile(
    r"siirtok+ilv|koek+ilv|provskylt|prvoskylt|trade\s*pl+ates?|transfer\s*pl+ates?"
    r"|pl+ates?\s*removed|no\s*pl+ates?|without\s*pl+ates?|unregi?ster|urnegister"
    r"|ei\s*(?:ole\s*)?(?:rekisteri)?kilp|ei\s*rekkar|ilman\s*(?:kilp|kili)"
    r"|kilv[et]*\s*(?:puut+u+|pois|irr?o?t+)|(?:puut+u+|pois|irr?o?t+)\w*\s*(?:rekisteri)?kilv"
    r"|poistettu\s*re[ek]+ist|re[ek]+ister\w*\s*(?:poist|irr?o?t)"
    r"|saknar\s*skylt|skylt\w*\s*bort|inga\s*skylt"
)

_SUMMER_CAR = re.compile(
    r"kes[äa]ren\w*\s*alla|kes[äa]kum\w*\s+alla|ke[äa]s?kum\w*\s+alla|kes[äa]rekailla|kes[äa]rnkaat\s*alla|kesr[äa]enkailla|ke[äa]renkailla|kess[äa]renkailla|kesrenkaat\s*alla"
    r"|viel[äa]\s*(?:kes[äa]ren|kess[äa]ren|kesr[äa]en|ke[äa]ren|kesren)"
    r"|sum+er\s*ty?res?\s*(?:still\s*)?on|still\s*on\s*summ|summers\s*still\s*on"
    r"|talvirenk\w*\s*ei\s*(?:viel[äa]\s*)?vaihd|winter\s*ty?res?\s*not\s*fitted"
    r"|sommardäck\w*\s*på|inga\s*vinterdäck|ei\s*talvirenk|no\s*winters"
)
_TYRES_IN_BOOT = re.compile(
    r"(?:kesären|kesäreng|summer\s*ty?res?|sommardäck|renkaat|rengassarja|wheels|tyres|tires)"
    r"[^.;!\n]{0,35}(?:kontissa|takakont|tavaratil|takapenk|kyydissä|mukaan|boot|trunk|back|bakluck|följer)"
    r"|(?:lisäksi|toinen\s*sarja|second\s*set|extra\s*set|spare\s*wheels|4\s*(?:kesären|rengasta))"
)

# Harmless boilerplate removed before timing checks --------------------------
_HARMLESS_NOISE = re.compile(
    r"(?:"
    r"(?:soita|ilmoita|call|ring)[^.;!\n]{0,60}(?:ennen|before|befoe|beofre|innan|asap|läht|saapum|perillä|tulet|arrive|arrival|pickup|nouto)"
    r"|(?:meidän\s*)?(?:viit+e(?:nro)?|var+aus|ti?ll?ausn(?:umero|ro)|booking\s*ref|order\s*no|ordernr|referens|ref|vår\s*ref)\s*[:#]?\s*\d+"
    r"|(?:koodi|code|kod|ruutu|ruudussa|paikka|plats|bay)\s*[:#]?\s*\d+"
    r"|[24]\s*(?:sarjaa|kesärengasta|rengasta|wheels|tyres|tires)"
    r"|(?:ready\s*,?\s*collect\s*right\s*away|ready\s*now|auto\s+(?:on\s+)?(?:valmi+s|valmmis|valims)\s+noudett?a?vaksi|(?:valmi+s|valmmis|valims)\s*,?\s*(?:nouto|voi\s+hakea)\s*heti|auto\s+(?:valmi+s|valmmis|valims)\s*,\s*nouto\s+heti\s+kun\s+sopii|valmiina\s*pihalla|bilen\s*står\s*klar|klar\s*,?\s*kan\s*hämtas\s*direk+t)"
    r"|(?:avaimet|avain|keys?|nyckel)\s+\w+"
    r"|(?:yksi|one|ett)\s+(?:siirto|auto|car|transfer|leg|bil)"
    r"|(?:haettava|nouto)\s+ennen\s+sitä"
    r"|(?:,\s*)?(?:ei|not|inte)\s+(?:huomenna|tomorrow|tmrw|i\s+morgon)\b"
    r")",
    re.IGNORECASE,
)

# Day references (relative to arrived_at) -----------------------------------
_WEEKDAYS = [
    (re.compile(r"\b(?:ma+na+nt\w*|måndag\w*|mondy|monday|mån|mon|ma\.?)\b"), 0),
    (re.compile(r"\b(?:tiist\w*|tisdag\w*|tuesday|tis|tue|ti\.?)\b"), 1),
    (re.compile(r"\b(?:keskiv\w*|kesiv\w*|onsdag\w*|wednesday|ons|wed|ke\.?)\b"), 2),
    (re.compile(r"\b(?:torst\w*|torsdag\w*|thu[rs]*day|tors|thurs|thu|to\.?)\b"), 3),
    (re.compile(r"\b(?:p[er]*jant\w*|fredag\w*|friday|fre|fri|pe\.?)\b"), 4),
    (re.compile(r"\b(?:lauant\w*|lördag\w*|saturday|lör|sat|la\.?)\b"), 5),
    (re.compile(r"\b(?:sunnunt\w*|söndag\w*|sunday|sön|sun|su\.?)\b"), 6),
]

_WORD_HOURS: dict[str, int] = {
    # Finnish (nominative + inflections)
    "yksi": 13, "yhtä": 13, "yhden": 13, "yhdeltä": 13, "yhteen": 13, "yhdestä": 13,
    "kaksi": 14, "kahta": 14, "kahden": 14, "kahdelta": 14, "kahteen": 14, "kahdesta": 14,
    "kolme": 15, "kolmea": 15, "kolmen": 15, "kolmelta": 15, "kolmeen": 15, "kolmesta": 15,
    "koolme": 15, "koolmen": 15, "koolmelta": 15, "koolmeen": 15,
    "neljä": 16, "neljää": 16, "neljän": 16, "neljältä": 16, "neljään": 16, "neljästä": 16,
    "viisi": 17, "viittä": 17, "viiden": 17, "viideltä": 17, "viiteen": 17, "viidestä": 17,
    "seitsemän": 7, "seitsemää": 7, "seitsemältä": 7, "seitsemään": 7, "seitsemästä": 7,
    "kahdeksan": 8, "kahdeksaa": 8, "kahdeksalta": 8, "kahdeksaan": 8, "kahdeksasta": 8,
    "yhdeksän": 9, "yhdeksää": 9, "yhdeksältä": 9, "yhdeksään": 9, "yhdeksästä": 9,
    "kymmenen": 10, "kymmentä": 10, "kymmeneltä": 10, "kymmeneen": 10, "kymmenestä": 10,
    "yksitoista": 11, "yhdentoista": 11, "yhdeltätoista": 11, "yhteentoista": 11, "yhdestätoista": 11,
    "kaksitoista": 12, "kahdentoista": 12, "kahdeltatoista": 12, "kahteentoista": 12, "kahdestatoista": 12,
    # Swedish
    "ett": 13, "två": 14, "tre": 15, "fyra": 16, "fem": 17,
    "sju": 7, "åtta": 8, "nio": 9, "tio": 10, "elva": 11, "tolv": 12,
    # English
    "one": 13, "two": 14, "three": 15, "four": 16, "five": 17,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_WORD_NUMS_RAW: dict[str, int] = {
    "yksi": 1, "yhtä": 1, "yhden": 1, "yhdeltä": 1, "yhteen": 1, "yhdestä": 1,
    "kaksi": 2, "kahta": 2, "kahden": 2, "kahdelta": 2, "kahteen": 2, "kahdesta": 2,
    "kolme": 3, "kolmea": 3, "kolmen": 3, "kolmelta": 3, "kolmeen": 3, "kolmesta": 3,
    "neljä": 4, "neljää": 4, "neljän": 4, "neljältä": 4, "neljään": 4, "neljästä": 4,
    "viisi": 5, "viittä": 5, "viiden": 5, "viideltä": 5, "viiteen": 5, "viidestä": 5,
    "seitsemän": 7, "seitsemää": 7, "seitsemältä": 7, "seitsemään": 7, "seitsemästä": 7,
    "kahdeksan": 8, "kahdeksaa": 8, "kahdeksalta": 8, "kahdeksaan": 8, "kahdeksasta": 8,
    "yhdeksän": 9, "yhdeksää": 9, "yhdeksältä": 9, "yhdeksään": 9, "yhdeksästä": 9,
    "kymmenen": 10, "kymmentä": 10, "kymmeneltä": 10, "kymmeneen": 10, "kymmenestä": 10,
    "yksitoista": 11, "yhtätoista": 11, "yhdentoista": 11, "yhdeltätoista": 11, "yhteentoista": 11, "yhdestätoista": 11,
    "kaksitoista": 12, "kahtatoista": 12, "kahdentoista": 12, "kahdeltatoista": 12, "kahteentoista": 12, "kahdestatoista": 12,
    "ett": 1, "två": 2, "tre": 3, "fyra": 4, "fem": 5,
    "sju": 7, "åtta": 8, "nio": 9, "tio": 10, "elva": 11, "tolv": 12,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_HALF_PREFIX = re.compile(r"\b(?:puol(?:i|ta)|halv|half(?:\s+past)?)\s+([a-zåäö]+|\d{1,2})\b")
_WORD_TIME_RE = re.compile(
    r"(?:\b(?:klo|kl\.?|kello|klockan|at)\s+)?\b("
    + "|".join(k for k in _WORD_HOURS if k not in {"yksi", "kaksi", "kolme", "neljä", "viisi", "ett", "två", "tre", "fyra", "fem", "one", "two", "three", "four", "five"})
    + r"|noon|keskipäivä\w*)\b"
    r"|\b(?:klo|kl\.?|kello|klockan|at)\s+(yksi|kaksi|kolme|neljä|viisi|ett|två|tre|fyra|fem|one|two|three|four|five)\b"
)

_RANGE_RE = re.compile(
    r"(?:(?:klo|kl\.?|kello)\s*)?(\d{1,2})(?:[.:](\d{2}))?\s*(am|pm)?(?::(?:st[aä]|lt[aä]))?\s*[-–]\s*(\d{1,2})(?:[.:](\d{2}))?\s*(am|pm)?(?::(?:[aeä]{2}n))?"
    r"|\b(\d{1,2}):(?:st[aä])\s+(\d{1,2}):(?:[aeä]{2}n)\b"
)
_CLOCK_RE = re.compile(
    r"(?:(?:klo|kl\.?|kello|klockan)\s*)?(\d{1,2})(?:[.:](\d{2}))?\s*(am|pm)?\b"
)
_CORRECTION_SPLIT = re.compile(
    r"(?:,\s*(?:ei\s+kun|eikun|korjaus|actually|instead|sorry\s*,?\s*(?:i\s+mean)?|menar|utan|nej\s*,?\s*förlåt|förlåt)|"
    r"\.\s*(?:korjaus|actually|sorry|ei\s+kun|eikun|nej\s*,?\s*förlåt|förlåt)|"
    r"\b(?:korjaus|ei\s+kun|eikun|correction)\s*[:,-]?)",
    re.IGNORECASE,
)

_AFTER = re.compile(
    r"j[aä]+l+k+e+n|al+ka+e?n|akaen|alkean|alken|alkan|ai+kai+s?i+(?:nt|tn)+aa?n|ei\s+ennen|not\s+(?:before|befoe)|earliest"
    r"|no\s+(?:pickup|collection|delivery|haku|nouto|hämt\w*)\s+be[of]*re|ei\s+(?:noutoa?|hakua?|toimitust?a?)\s+ennen|inte\s+(?:hämt\w+|lever\w+)\s+före"
    r"|valmi+s|valmmis|valims|valmistuu|ready|klar|efter|tidigast|inte\s+före|\bförst\b|framåt|maksettu|maksun|paid|clears?|betalning"
    r"|luovutus|handover|released|from|vasta|\bafter\b"
)
_NEG_AFTER_CLOSE = re.compile(
    r"(?:ei\s+(?:ketään|kukaan|paikalla|enää\s+\w+)|ketään\s+ei|no\s*one|nobody|ingen(?:\s+(?:här|på\s+plats))?)\b.*?(?:j[aä]+l+k+e+n|after|efter)"
    r"|(?:\d{1,2}[.:]\d{2}|klo\s*\d+)\s*j[aä]+l+k+e+n\s+ei\s+(?:enää|ketään)"
)
_BEFORE = re.compile(
    r"ennen|m[e]?n+s?es+[äa]|vii+m\w*|la+t+e?st|\bby\b|be[of]*re|until|no\s+later\s+than|s[en]+ast|före|\btill\b|asti|saakka"
    r"|sul+ke+|kiinni|cl+os\w*|stäng\w*|oltava\s+silloin"
)
_DELIVERY = re.compile(
    r"toimit\w*|perill[äa]|peirl+\w*|perille|vastaanot\w*|kohde|kohtee\w*|deliver\w*|de[st]+\w*ation|d[es]+tin\w*|ar+iv\w*|receiv\w*"
    r"|m+o?t+ag\w*|lever\w*|framme|asiakas\s+(?:tulee|odottaa|noutaa)|customer\s+picks\s+it\s+up|oltava\s+(?:perill|peirl|silloin)"
)
_PICKUP = re.compile(
    r"nouto|noudet\w*|nouda|haettava|hakea|haku|pickup|collect\w*|clolect|hämt\w*|liike|auki|avoinna|öppet|\bopen\b|ready|valmi?s|valims|klar|maksettu|luovutus|handover|myyj[äa](?:\s+paikalla)?|seller|säljare"
)
_OPENING_CUE = re.compile(r"au?k?i(?:olo\w*)?|avoinna|öppet|\bopen(?:ing(?:\s*hours)?)?\b|arkipäivisin|a[rk]+i+sin|vardagar|weekday|ma\s*[-–—]\s*pe|m[åa]n\s*[-–—]\s*fre|mon\s*[-–—]\s*fri|only|vain|vastaanotto|receiving|m+o?t+ag|sulkee|kiinni|cl+os\w*|stäng\w*|myyj[äa]\s+paikalla")
_TIMING_RESIDUAL = re.compile(
    r"\b(?:valmi?s|valmmis|valims|valmistuu|ready|klar|after|before|befoe|jälkeen|ennen|aikai+s?i+ntaan"
    r"|vii+m\w*|m[e]?n+es+s[äa]|s[en]+ast|tidigast|efter|före|\btill\b|\bförst\b|klo|kl\.?|huo?m\w*n+a|ylihuo?\w*n+a"
    r"|t?o?m+o?r+o?w|t?o?mr?o?r+o?w|tmrw|moor?g[eo]?n|övermorgon|auk?i|öppet|sulkee|kiinni|cl+os\w*|stäng\w*|maksettu|betalning|noon)\b"
)


@dataclass(frozen=True)
class NoteConstraints:
    pickup_after: int | None = None
    pickup_before: int | None = None
    delivery_after: int | None = None
    delivery_before: int | None = None
    pickup_hours: tuple[int | None, int | None, bool] | None = None
    delivery_hours: tuple[int | None, int | None, bool] | None = None
    manual_licence: bool = False
    with_plates: bool = False
    summer_tyres: bool = False
    status: str = "harmless"  # harmless, safe_timing, ambiguous
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def chainable(self) -> bool:
        return self.status in {"harmless", "safe_timing"} and not (
            self.manual_licence or self.with_plates or self.summer_tyres
        )

    @property
    def has_fact(self) -> bool:
        return (
            self.status != "harmless"
            or self.manual_licence
            or self.with_plates
            or self.summer_tyres
            or self.pickup_after is not None
            or self.pickup_before is not None
            or self.delivery_after is not None
            or self.delivery_before is not None
            or self.pickup_hours is not None
            or self.delivery_hours is not None
        )


def _day_offset(clause: str, arrived_day: int) -> int | None:
    if re.search(r"\b(?:ylihuo?\w*n+a|i\s+övermorgon|övermorgon|day\s+after\s+tomorrow)\b", clause):
        return 2
    if re.search(r"\b(?:h[ou]*[eo]*m+n?[eo]*n+a[n]?|i\s+mo?ro?g[eo]?n|imo+rg[eo]?n|t?o?m+o?r+o?w|t?o?mr?o?r+o?w|tmrw|toorrow)\b", clause):
        return 1
    if re.search(r"\b(?:tänään|idag|today)\b", clause):
        return 0
    arr_wd = arrived_day % 7
    for pattern, wd in _WEEKDAYS:
        if pattern.search(clause):
            offset = (wd - arr_wd) % 7
            return offset if offset >= 2 else offset + 7
    return None


def _clock_minutes(hour: int, minute: int, ampm: str | None) -> int | None:
    if ampm == "pm" and hour < 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    if hour > 24 or minute > 59 or (hour == 24 and minute > 0):
        return None
    return hour * HOUR + minute


def _strip_parens_negations(text: str) -> str:
    # Remove parenthetical negations like "(ei puoli kolmelta)" or "(not 2pm)"
    return re.sub(r"\([^)]*\b(?:ei|not|inte)\b[^)]*\)", " ", text)


def _extract_single_time(clause: str) -> int | None:
    """Extract a single time-of-day (in minutes from midnight) from a clause."""
    # Check "half three" / "puoli kolme" / "halv tre"
    half_match = _HALF_PREFIX.search(clause)
    if half_match:
        tok = half_match.group(1)
        if tok.isdigit():
            h = int(tok)
            if 1 <= h <= 6:
                h += 12
            return (h - 1) * HOUR + 30 if 1 <= h <= 23 else None
        if tok in _WORD_NUMS_RAW:
            h = _WORD_NUMS_RAW[tok]
            # "puoli kolme" = 14:30, "puoli kymmeneltä" = 09:30, "puoli kahdeltatoista" = 11:30
            if 1 <= h <= 6:
                h += 12
            return (h - 1) * HOUR + 30
    # Check digits (explicit clock token preferred)
    candidates: list[tuple[int, int]] = []
    for m in _CLOCK_RE.finditer(clause):
        raw = m.group(0)
        h, mins, ampm = int(m.group(1)), int(m.group(2) or 0), m.group(3)
        # Skip bare integers without klo/kl/:/./am/pm if implausible as hour
        has_marker = bool(re.search(r"klo|kl|kello|klockan|[:.]|am|pm", raw))
        if not has_marker and (h < 6 or h > 22):
            continue
        val = _clock_minutes(h, mins, ampm)
        if val is not None:
            candidates.append((m.start(), val))
    if candidates:
        # Check for inline "ei klo X" negation before candidate
        filtered = []
        for pos, val in candidates:
            before = clause[max(0, pos - 12):pos]
            if re.search(r"(?<!ei\sennen\s)\b(?:ei|not|inte)\s+(?:klo\s+|kl\.?\s*)?$", before) and not re.search(r"(?:ei\s+ennen|not\s+before|inte\s+före)\s*$", before):
                continue
            filtered.append(val)
        if filtered:
            return filtered[-1]
    word_match = _WORD_TIME_RE.search(clause)
    if word_match:
        tok = word_match.group(1) or word_match.group(2)
        if tok in ("noon", "keskipäivä") or tok.startswith("keskipäivä"):
            return 12 * HOUR
        if tok in _WORD_HOURS:
            return _WORD_HOURS[tok] * HOUR
    return None


def _apply_hours(current: tuple[int | None, int | None, bool] | None,
                 opens: int | None, closes: int | None, weekdays_only: bool) -> tuple[int | None, int | None, bool]:
    if current is None:
        return (opens, closes, weekdays_only)
    cur_o, cur_c, cur_w = current
    new_o = opens if cur_o is None else (max(cur_o, opens) if opens is not None else cur_o)
    new_c = closes if cur_c is None else (min(cur_c, closes) if closes is not None else cur_c)
    return (new_o, new_c, cur_w or weekdays_only)


def parse_note(text: str | None, arrived_at: int) -> NoteConstraints:
    """Parse explicit requirements without guessing a target or correction."""
    if not text or not text.strip():
        return NoteConstraints()
    lowered = text.lower().strip()
    reasons: list[str] = []

    manual = bool(_MANUAL.search(lowered)) and not bool(_MANUAL_NEGATED.search(lowered))
    plates = bool(_PLATES.search(lowered))
    summer = bool(_SUMMER_CAR.search(lowered)) and not (
        bool(_TYRES_IN_BOOT.search(lowered))
        and not re.search(
            r"kes[äa]renk\w*lla|kes[äa]kum\w*\s+alla|ke[äa]s?kum\w*\s+alla|viel[äa]\s*(?:kes|sum)|not\s*fitted|ei\s*talvirenk|no\s*winters|(?:sum+er|sommardäck)\s+(?:ty?res?\s+)?(?:still\s+)?(?:on|på)",
            lowered,
        )
    )
    if manual:
        reasons.append("manual")
    if plates:
        reasons.append("plates")
    if summer:
        reasons.append("summer_tyres")

    # Protect common abbreviations before noise/sentence splitting
    prepped = re.sub(r"\b(viim|kl|klo|esim|n|t|ma|ti|ke|to|pe|la|su)\.", r"\1", lowered)
    prepped = re.sub(r"\b(klo|kl|kello|klockan)\s*(\d{2})(\d{2})\b", r"\1 \2:\3", prepped)
    cleaned = _strip_parens_negations(_HARMLESS_NOISE.sub(" ", prepped))
    arrived_day = arrived_at // DAY

    values: dict[str, int | None] = {
        "pickup_after": None, "pickup_before": None,
        "delivery_after": None, "delivery_before": None,
    }
    pickup_hours: tuple[int | None, int | None, bool] | None = None
    delivery_hours: tuple[int | None, int | None, bool] | None = None
    found_timing = False
    ambiguous = False

    # Split into sentence-level clauses (preserving decimal dots like 13.30 followed by digits)
    clauses = [c.strip() for c in re.split(r"\.(?!\d)|[;!\n]", cleaned) if c.strip()]
    last_point_key: str | None = None
    last_point_tod: int | None = None
    last_point_day: int = 0
    for raw_clause in clauses:
        # Handle sentence-level correction ("Correction: Friday." / "Korjaus: maanantaina." / "Korjaus: klo 15.30.")
        if re.match(r"^(?:korjaus|correction|actually|ei\s+kun|eikun|sorry(?:\s*,?\s*i\s+mean)?)\b", raw_clause):
            new_day = _day_offset(raw_clause, arrived_day)
            new_tod = _extract_single_time(raw_clause)
            if last_point_key is not None and (new_day is not None or new_tod is not None):
                tod_use = new_tod if new_tod is not None else last_point_tod
                day_use = new_day if new_day is not None else last_point_day
                if tod_use is not None:
                    values[last_point_key] = (arrived_day + day_use) * DAY + tod_use
                    last_point_tod, last_point_day = tod_use, day_use
                    continue
        # Handle corrections within a clause by taking the final corrected segment,
        # while retaining direction/target context from the earlier segment if omitted.
        parts = _CORRECTION_SPLIT.split(raw_clause)
        if len(parts) > 1:
            prior_ctx = parts[0]
            clause = parts[-1].strip()
            # Check for an inline day-only correction ("... sunnuntaina klo 14, ei kun maanantaina!")
            if clause and _extract_single_time(clause) is None:
                new_day = _day_offset(clause, arrived_day)
                prior_tod = _extract_single_time(prior_ctx)
                if new_day is not None and prior_tod is not None:
                    stripped_prior = re.sub(
                        r"\b(?:tänään|idag|today|huo?m\w*n+a|ylihuo?\w*n+a|ma+na+nt\w*|tiist\w*|keskiv\w*|torst\w*|p[er]*jant\w*|lauant\w*|sunnunt\w*|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
                        "",
                        prior_ctx,
                    )
                    clause = f"{stripped_prior} {clause}".strip()
        else:
            prior_ctx = ""
            clause = raw_clause
        if not clause:
            continue

        # 1. Check opening-hours range (e.g. "9-16", "klo 10.00-14.30", "9:stä 16:een", "8am-8pm")
        range_matches = list(_RANGE_RE.finditer(clause))
        if range_matches:
            is_home_only = False
            for rm in range_matches:
                if rm.group(1) is not None:
                    h1, m1, ap1 = int(rm.group(1)), int(rm.group(2) or 0), rm.group(3)
                    h2, m2, ap2 = int(rm.group(4)), int(rm.group(5) or 0), rm.group(6)
                else:
                    h1, m1, ap1, h2, m2, ap2 = int(rm.group(7)), 0, None, int(rm.group(8)), 0, None
                opens, closes = _clock_minutes(h1, m1, ap1), _clock_minutes(h2, m2, ap2)
                if opens is None or closes is None or opens >= closes:
                    ambiguous = True
                    reasons.append("invalid_range")
                    continue
                weekdays_only = bool(re.search(r"arkipäivisin|a[rk]+i+sin|vardagar|ma\s*[-–—]\s*pe|m[åa]n\s*[-–—]\s*fre|mon\s*[-–—]\s*fri|weekday", clause))
                range_day = None if weekdays_only else _day_offset(clause, arrived_day)
                if _DELIVERY.search(clause) and not re.search(r"\b(?:pickup|nouto|haku)\b", clause):
                    if range_day is not None and not re.search(r"\b(?:only|vain|bara)\b", clause):
                        values["delivery_after"] = max(values["delivery_after"] or 0, (arrived_day + range_day) * DAY + opens)
                        values["delivery_before"] = min(values["delivery_before"], (arrived_day + range_day) * DAY + closes) if values["delivery_before"] is not None else (arrived_day + range_day) * DAY + closes
                    else:
                        delivery_hours = _apply_hours(delivery_hours, opens, closes, weekdays_only)
                elif _PICKUP.search(clause) or _OPENING_CUE.search(clause):
                    if range_day is not None and not re.search(r"\b(?:only|vain|bara)\b", clause):
                        values["pickup_after"] = max(values["pickup_after"] or 0, (arrived_day + range_day) * DAY + opens)
                        values["pickup_before"] = min(values["pickup_before"], (arrived_day + range_day) * DAY + closes) if values["pickup_before"] is not None else (arrived_day + range_day) * DAY + closes
                    else:
                        pickup_hours = _apply_hours(pickup_hours, opens, closes, weekdays_only)
                elif re.search(r"kotona|home|hemma", clause):
                    # Private seller window already in request
                    is_home_only = True
                else:
                    ambiguous = True
                    reasons.append("unknown_range_target")
                found_timing = True
            # Remove ranges before checking for additional point-in-time constraints
            clause_rem = _RANGE_RE.sub(" ", clause).strip()
            if range_matches and len(range_matches) == 1 and not weekdays_only and _day_offset(clause, arrived_day) is not None and _extract_single_time(clause_rem) is None and not re.search(r"\b(?:only|vain|bara)\b", clause):
                continue
            if is_home_only or (not _TIMING_RESIDUAL.search(clause_rem) and not re.search(r"\d", clause_rem)):
                continue
            clause = clause_rem

        # 2. Check point-in-time constraints in the clause
        # Split on comma only if >= 2 comma-parts each contain a clock/word time
        comma_parts = [s.strip() for s in clause.split(",") if s.strip()]
        subclauses = comma_parts if (
            len(comma_parts) > 1 and sum(_extract_single_time(s) is not None for s in comma_parts) >= 2
        ) else [clause]

        for sub in subclauses:
            if len(subclauses) > 1 and re.match(r"^(?:ei|not|inte)\s+", sub):
                continue
            tod = _extract_single_time(sub)
            if tod is None:
                if _TIMING_RESIDUAL.search(sub) and not re.search(r"valmiina|ready\b", sub):
                    # Check if it's purely a closing/opening phrase without a clock
                    if not re.search(r"auki|öppet|weekday|a[rk]+i+sin|vardagar", sub):
                        ambiguous = True
                        reasons.append("unparsed_timing")
                continue

            day_off = _day_offset(sub, arrived_day)
            if day_off is None and prior_ctx:
                day_off = _day_offset(prior_ctx, arrived_day)
            if day_off is None:
                day_off = 0
            minute = (arrived_day + day_off) * DAY + tod

            # Explicit closing time ("kohde sulkee klo 16.30", "liike kiinni klo 16.30", "täällä ei ketään puoli viiden jälkeen")
            if (re.search(r"sul+ke+|kiinni|cl+os\w*|stäng\w*", sub) and not _AFTER.search(sub)) or _NEG_AFTER_CLOSE.search(sub):
                if _DELIVERY.search(sub):
                    delivery_hours = _apply_hours(delivery_hours, None, tod, False)
                else:
                    pickup_hours = _apply_hours(pickup_hours, None, tod, False)
                found_timing = True
                continue

            direction = "after" if _AFTER.search(sub) else "before" if _BEFORE.search(sub) else None
            if direction is None and prior_ctx:
                direction = "after" if _AFTER.search(prior_ctx) else "before" if _BEFORE.search(prior_ctx) else None
            if direction is None:
                # Phrases like "Auto valmistuu torstaina yhdeltä" already match _AFTER via valmistuu
                ambiguous = True
                reasons.append("missing_direction")
                continue

            ctx_for_target = f"{prior_ctx} {sub}" if prior_ctx else sub
            is_delivery = bool(_DELIVERY.search(ctx_for_target)) and not re.search(r"\b(?:pickup|nouto|haettava|ready|valmi?s|valims)\b", sub)
            target = "delivery" if is_delivery else "pickup"
            key = f"{target}_{direction}"
            old = values[key]
            values[key] = max(old, minute) if direction == "after" and old is not None else (
                min(old, minute) if old is not None else minute
            )
            last_point_key, last_point_tod, last_point_day = key, tod, day_off
            found_timing = True

    if values["pickup_after"] and values["pickup_before"] and values["pickup_after"] > values["pickup_before"]:
        # A day-less "nouto ennen 19" alongside a future ready time is a daily pickup closing hour.
        tod_before = values["pickup_before"] % DAY
        if values["pickup_before"] < arrived_at + DAY:
            pickup_hours = _apply_hours(pickup_hours, None, tod_before, False)
            values["pickup_before"] = None
        else:
            ambiguous = True
            reasons.append("empty_pickup_window")
    if values["delivery_after"] and values["delivery_before"] and values["delivery_after"] > values["delivery_before"]:
        ambiguous = True
        reasons.append("empty_delivery_window")

    status = "ambiguous" if ambiguous else "safe_timing" if found_timing else "harmless"
    return NoteConstraints(
        **values,
        pickup_hours=pickup_hours,
        delivery_hours=delivery_hours,
        manual_licence=manual,
        with_plates=plates,
        summer_tyres=summer,
        status=status,
        reasons=tuple(dict.fromkeys(reasons)),
    )
