"""The prompt templates shared by every run mode.

The model sees: today's date (the forecast moment), the event and question, the
resolution rules, the scheduled close date and the news block. It never sees the
market price, best bid or best ask (in the default template), the depth number, the
book flags, the raw order book, or anything from the analysis-only tables.

Run modes differ only in the news block:
  normal        -> the stored news text (or the no-news line if none was captured)
  memory_probe  -> the no-news line, always
(There is no separate calibration mode: the normal pass covers every row once and the
calibration split lives in the database.)

Two templates:
  default     -> price-blind: the main pass and the memory probe never show the market
                 price, best bid or best ask
  with_price  -> the same text plus the current price, best bid and best ask block;
                 kept in the code but not used in the study
"""
import hashlib
import re
from typing import Dict

MODES = ("normal", "memory_probe")

NO_NEWS_LINE = "No news is available."

TEMPLATE = """You are forecasting the outcome of a prediction-market question.

Today's date and time (UTC): {today}
Everything below is what is known as of today. Do not use or assume knowledge of anything that happened after today.

Event: {event_title}
Question: {question}{contract_line}

Resolution rules (word for word from the exchange):
{description}

Scheduled market close: {market_end_date}

Current market for the Yes contract (pays $1 if the answer is Yes, $0 if not):
- Current price: {price_yes}
- Best bid (highest price someone is offering to pay for Yes): {best_bid}
- Best ask (lowest price someone is offering to sell Yes for): {best_ask}

News:
{news_block}

Give your own probability that this question resolves Yes. Reason briefly, then end your reply with one line in exactly this form:
PROBABILITY: <number between 0 and 1>"""

STRICT_SUFFIX = """

Your previous reply did not contain a usable probability. You must answer with a number even if you are uncertain; refusing is not an option. Reply with exactly one line and nothing else:
PROBABILITY: <decimal number between 0 and 1, for example 0.35>"""

_PRICE_BLOCK = """Current market for the Yes contract (pays $1 if the answer is Yes, $0 if not):
- Current price: {price_yes}
- Best bid (highest price someone is offering to pay for Yes): {best_bid}
- Best ask (lowest price someone is offering to sell Yes for): {best_ask}

"""
TEMPLATE_WITH_PRICE = TEMPLATE
assert _PRICE_BLOCK in TEMPLATE_WITH_PRICE
TEMPLATE = TEMPLATE_WITH_PRICE.replace(_PRICE_BLOCK, "")
for _ph in ("{price_yes}", "{best_bid}", "{best_ask}"):
    assert _ph in TEMPLATE_WITH_PRICE and _ph not in TEMPLATE, _ph
assert "\n\n\n" not in TEMPLATE, "double blank line left where the price block was"

TEMPLATES = {"default": TEMPLATE, "with_price": TEMPLATE_WITH_PRICE}


_STRIP_PREFIXES = ("additional clarification", "clarification", "addendum")


def clean_description(description: str):
    """Drop any paragraph whose first line starts (case-insensitive) with
    "Clarification", "Additional clarification" or "Addendum" AND that carries
    clarification text inside the same paragraph (more text after the heading on that
    line, or further lines). A bare heading line on its own stays, because what follows
    it in later paragraphs is ordinary rules text. Exchanges append such paragraphs
    after listing, sometimes after the outcome.
    Returns (kept text, [stripped paragraphs])."""
    paragraphs = re.split(r"\n\s*\n", (description or "").strip())
    kept, stripped = [], []
    for para in paragraphs:
        para = para.strip()
        lines = para.splitlines()
        first = lines[0].strip() if lines else ""
        m = re.match(r"(additional clarification|clarification|addendum)[a-z()]*\s*[:\-–]?\s*(.*)$", first, re.I)
        has_body = bool(m) and (bool(m.group(2).strip()) or len(lines) > 1)
        if m and has_body:
            stripped.append(para)
        else:
            kept.append(para)
    return "\n\n".join(kept), stripped


def current_price(row: Dict):
    """The order-book midpoint when there is one, else the stored price (Window A's
    stored price can be days old; B and C store price = mid, so nothing changes there)."""
    return row["mid_yes"] if row.get("mid_yes") is not None else row["price_yes"]


def _num(x) -> str:
    if x is None:
        return "none"
    s = ("%.4f" % float(x)).rstrip("0").rstrip(".")
    return s if s else "0"


def news_block(row: Dict, mode: str) -> str:
    if mode == "memory_probe":
        return NO_NEWS_LINE
    if not row["news_available"] or not (row["news_text"] or "").strip():
        return NO_NEWS_LINE
    return row["news_text"].strip()


def build_prompt_info(row: Dict, mode: str, strict: bool = False, template: str = "default"):
    """Returns (prompt text, info) where info records what was changed on the way in:
    description_paragraphs_stripped and the stripped text."""
    if mode not in MODES:
        raise ValueError("unknown mode %r" % mode)
    description, stripped = clean_description(row["description"])
    detail = (row.get("question_detail") or "").strip()
    text = TEMPLATES[template].format(
        today=row["forecast_ts"],
        event_title=row["event_title"] or "",
        question=row["question"] or "",
        contract_line=("\nContract: " + detail) if detail else "",
        description=description or "(none given)",
        market_end_date=row["market_end_date"] or "not stated",
        price_yes=_num(current_price(row)),
        best_bid=_num(row["best_bid_yes"]),
        best_ask=_num(row["best_ask_yes"]),
        news_block=news_block(row, mode),
    )
    if strict:
        text += STRICT_SUFFIX
    return text, {"description_paragraphs_stripped": len(stripped), "description_stripped_text": stripped,
                  "price_shown_from": "mid_yes" if row.get("mid_yes") is not None else "price_yes"}


def build_prompt(row: Dict, mode: str, strict: bool = False, template: str = "default") -> str:
    return build_prompt_info(row, mode, strict, template)[0]


def template_hash(template: str = "default") -> str:
    return hashlib.sha256((TEMPLATES[template] + STRICT_SUFFIX).encode("utf-8")).hexdigest()[:16]
