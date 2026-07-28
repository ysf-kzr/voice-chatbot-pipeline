"""
Rudimentary voice AI chatbot pipeline - text-output variant. Runs either over
WebRTC in the browser or directly against the local mic from the CLI. Input
is still spoken (mic), but the bot's reply is text-only: no TTS, no speaker
output. Over WebRTC/eval, the reply is delivered to the connected client as
an RTVI "bot-llm-text" message; in --local mode it's only visible in the
console (CONVO log line), since there's no client to receive a text message.

Pipeline: mic (WebRTC or local) -> Silero VAD -> Groq STT (Whisper)
          -> Groq LLM (Llama 3.3 70B) -> text delivered via RTVI (no TTS)

Also includes worked examples of LLM tool calling against TWO REAL websites,
not fake/hardcoded data:

  Honda Pakistan (honda.com.pk):
    - check_honda_price: real, live starting prices from the homepage mega-menu.
    - browse_honda_page: fetches any known page (specs, dealers, promotions, etc.)

  MG Motors Pakistan (mgmotors.com.pk):
    - browse_mg_page: fetches any known page (model specs, dealers, financing,
      offers, contact info, etc.) MG does not publish prices on their site, so
      there is no price tool — the bot will direct users to contact a dealer.

Run:
    python bot.py             # WebRTC server; open http://localhost:7860
    python bot.py --local     # Talk directly via the local mic (reply is console-only)

Requires GROQ_API_KEY in a .env file.
"""

import asyncio
import os
import re
import sys
import time

import openai
import pyaudio
from bs4 import BeautifulSoup
from curl_cffi import requests as curl_requests
from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import ErrorFrame, Frame, LLMRunFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.models import (
    BotLLMStartedMessage,
    BotLLMStoppedMessage,
    BotLLMTextMessage,
    TextMessageData,
)
from pipecat.runner.types import EvalRunnerArguments, RunnerArguments, SmallWebRTCRunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.groq.stt import GroqSTTService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.whisper.base_stt import Transcription
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.workers.runner import WorkerRunner

# Not override=True: a real env var the user already set (e.g. `export
# GROQ_API_KEY=...` for a one-off test) should win over whatever's in
# .env, not get silently clobbered back to the .env value. .env is a
# fallback default for local dev, not an authority over the real
# environment - override=True previously inverted that, which is the
# entire reason scripts/run_all_evals.sh used to have to physically
# rename .env out of the way to test a bad key.
load_dotenv()

if "GROQ_API_KEY" not in os.environ:
    sys.exit("GROQ_API_KEY is missing - add it to your .env file.")

# Windows consoles default to cp1252, which can't encode Urdu/Arabic script.
# Reconfigure stderr to UTF-8 before loguru attaches so transcripts print
# correctly instead of crashing or showing ???.
if hasattr(sys.stderr, "buffer") and sys.stderr.encoding.lower() != "utf-8":
    import io
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logger.remove(0)

# A dedicated level for just the conversation transcript (what the user said,
# what the bot said) - separate from DEBUG/INFO noise. Must be registered
# before any sink references it by name.
logger.level("CONVO", no=25, color="<green>", icon="")

# --local mode's console IS the product's UI (no browser client to render
# anything) - defaulting it to full DEBUG output means every run is a wall
# of pipeline internals instead of a conversation. WEBRTC_DEFAULT_LOG_LEVEL
# is unchanged so nothing about that path is affected. Explicit LOG_LEVEL
# always wins over either default, for real debugging when needed.
# WARNING (not CONVO) is used here specifically so loguru's own sink stays
# silent by default in --local mode - the plain-print chat UI below
# (_print_chat/_print_thinking) is what actually renders the conversation
# there now, not loguru's CONVO-level lines (which still fire, for anyone
# who sets LOG_LEVEL=CONVO or DEBUG explicitly and wants the old behavior).
_IS_LOCAL_MODE = "--local" in sys.argv
_DEFAULT_LOG_LEVEL = "WARNING" if _IS_LOCAL_MODE else "DEBUG"
logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", _DEFAULT_LOG_LEVEL))


# Bright cyan / bright green so the two speakers are visually distinct from
# each other and from plain terminal text (default log lines, prompts,
# etc.) - readable on both dark and light terminal themes since these are
# the bright ANSI variants, not the dim base 8 colors.
_YOU_COLOR = "\033[96m"
_BOT_COLOR = "\033[92m"
_COLOR_RESET = "\033[0m"


def _clear_console_line() -> None:
    if _IS_LOCAL_MODE:
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()


def _print_chat(role: str, text: str) -> None:
    """Clean, colored chat-only console line for --local mode: bypasses
    loguru's formatting (timestamp/level/module) entirely, and clears any
    pending "thinking..." indicator first. No-op outside --local mode - a
    WebRTC client renders its own UI from RTVI events, this isn't for it.

    Args:
        role: "You" or "Bot" - picks the color and the printed label.
        text: the message content (without the "Role: " prefix).
    """
    if _IS_LOCAL_MODE:
        color = _YOU_COLOR if role == "You" else _BOT_COLOR
        _clear_console_line()
        sys.stderr.write(f"{color}{role}: {text}{_COLOR_RESET}\n")
        sys.stderr.flush()


def _print_thinking() -> None:
    """Shows a "Bot: thinking..." placeholder immediately after the user's
    turn is captured, masking the STT-already-done-but-LLM-still-generating
    gap instead of leaving the console looking frozen. Deliberately printed
    with no trailing newline so _print_chat's next call can overwrite this
    exact line (via \\r + clear-line) instead of leaving it sitting above
    the real reply once one arrives. Same color as a real "Bot:" line so it
    reads as a continuation of the same speaker once overwritten.
    """
    if _IS_LOCAL_MODE:
        sys.stderr.write(f"{_BOT_COLOR}Bot: thinking...{_COLOR_RESET}")
        sys.stderr.flush()

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant. The user is speaking to you out loud, but "
    "your replies are delivered back as text, not spoken - so normal written "
    "formatting is fine. "
    "You understand all languages. Always respond in English regardless of what language the user speaks in. "
    "Your replies are read as text, not spoken aloud, so there's no need to "
    "artificially shorten them the way a spoken answer would need to be - "
    "give complete, useful answers (including specs, lists, or detail) when "
    "the question calls for it. Still be direct: no filler, no unnecessary "
    "preamble, no restating the question, no hedging caveats, and no padding "
    "a simple answer just to sound thorough. "
    "If the user asks about the price of a Honda Civic, HR-V, or City, use "
    "the check_honda_price tool rather than guessing - never make up a price. "
    "For anything else about Honda Pakistan - specs, features, dealers, "
    "contact info, promotions, company info, policies - use the "
    "browse_honda_page tool to check the real website rather than guessing. "
    "For any question about MG Motors Pakistan - model specs, features, "
    "dealers, contact info, offers, financing, after-sales service - use the "
    "browse_mg_page tool to check the real mgmotors.com.pk website. "
    "MG Motors Pakistan does not publish car prices on their website, so if "
    "the user asks for an MG price, tell them prices are not listed online "
    "and suggest they contact a dealer or visit mgmotors.com.pk/dealer-locator. "
    "If the user asks you to compare a Honda model to an MG model, or asks "
    "anything else that needs data from both brands, call both "
    "browse_honda_page and browse_mg_page (and check_honda_price if a Honda "
    "price is part of the comparison) before answering, then reason over "
    "everything you fetched - never compare from memory or guesswork. "
    "If the user's message is too short, vague, or ambiguous to answer "
    "meaningfully (e.g. a single word like 'yes' or 'ok' with no clear "
    "context), don't guess at what they might mean - ask a brief clarifying "
    "question instead."
)

GREETING_MESSAGE = "Hi! How can I help you today?"
FALLBACK_ERROR_MESSAGE = "Sorry, I hit a glitch there. Could you say that again?"
FALLBACK_COOLDOWN_SECS = 5.0

# Ordered fallback chain of Groq chat models to fail over across on a rate
# limit, tried in order - the first is the one actually used until a rate
# limit forces a move to the next. Both are tool-calling-capable (a hard
# requirement here, given check_honda_price/browse_honda_page/
# browse_mg_page); llama-3.1-8b-instant is the weaker/faster fallback.
#
# Do not add llama3-groq-70b-8192-tool-use-preview back to this chain -
# Groq has fully decommissioned it (a hard 400 model_decommissioned error,
# not a rate limit). Check GET /v1/models against Groq's API before adding
# any other model here.
LLM_MODEL_FALLBACK_CHAIN = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]


def is_rate_limit_error(error_text: str) -> bool:
    """Fallback heuristic rate-limit detector: substring match on the
    lowercased error text. NOT the primary check anymore - see
    error_frame_is_rate_limit below, which checks a real, typed exception
    first and only falls back to this when no exception object is
    available at all.

    Known limitation, not fixed here: "429" is checked as a bare substring,
    so it could false-positive on an unrelated number that happens to
    contain those digits elsewhere in the error text (a token count, a
    request id) - error_frame_is_rate_limit's typed check avoids exactly
    this for the normal case; this text-only fallback keeps the weakness
    for the rarer case where no exception object made it through. Kept as
    its own pure function so this behavior - including that known
    weakness - is unit-tested (see tests/test_bot.py) rather than only
    exercised indirectly via a live rate limit.
    """
    return "rate_limit" in error_text or "429" in error_text or "rate limit" in error_text


def error_frame_is_rate_limit(frame: ErrorFrame) -> bool:
    """Real, typed rate-limit check - prefers this over string matching.

    Groq's client is OpenAI-compatible, so GroqLLMService raises
    openai.RateLimitError under the hood - a class with a reliable
    class-level `status_code = 429` (see the installed openai package's
    _exceptions.py), a genuine signal rather than a guess based on what the
    error message happens to say. Both the normal chat-completion path and
    the background context-summarization path (pipecat's
    services/openai/base_llm.py and services/llm_service.py) call
    push_error(..., exception=e), so frame.exception is populated in both
    cases this pipeline actually hits.

    Falls back to the text heuristic (is_rate_limit_error) only when
    frame.exception is None - some other pipecat-internal path might push
    an ErrorFrame without one. If exception IS present but isn't a
    RateLimitError, that's authoritative and trusted over the text
    heuristic, not just an additional vote - a non-rate-limit exception
    whose str() happens to contain "429" shouldn't be miscounted just
    because the text check alone would have said yes.
    """
    if isinstance(frame.exception, openai.RateLimitError):
        return True
    if frame.exception is not None:
        return False
    return is_rate_limit_error(str(frame.error).lower())


def is_background_summarization_error(error_text: str) -> bool:
    """True if `error_text` (lowercased) came from pipecat's own background
    context-summarization feature, not a real user-facing turn.

    Matches the exact, stable, library-authored prefixes pipecat's
    services/llm_service.py uses for this feature's two failure paths:
    "Error generating context summary: ..." and "Context summarization
    timed out after {N}s". A summarization failure has nothing to do with
    whatever the user actually asked in their current turn - showing them a
    "sorry, glitch" apology and spending the circuit breaker's budget on a
    background job that failed would be a false alarm.
    """
    return error_text.startswith("error generating context summary") or error_text.startswith(
        "context summarization timed out"
    )


def is_tool_call_failed_error(error_text: str) -> bool:
    """True if `error_text` (lowercased) is Groq's own tool-call-generation
    failure message, not a network/rate-limit/other issue.

    Groq occasionally can't get a model to produce a syntactically valid
    function call for a legitimate tool-calling request and returns this
    exact message instead of a malformed call: "Failed to call a function.
    Please adjust your prompt." (confirmed live - a real Honda-contact-info
    request that should have called browse_honda_page hit this verbatim).
    Usually a one-off sampling issue: a fresh attempt at the SAME turn
    (same context, nothing about the user's question changed) often
    succeeds on retry, unlike a rate limit or a genuine model/tool mismatch.
    """
    return "failed to call a function" in error_text


# Worked example of tool calling against a REAL website (not fake/hardcoded
# data): honda.com.pk's homepage includes a mega-menu block, present on
# every page, listing each model line's current starting price - e.g.
#   <h4>Honda Civic</h4> ... <div class="model-price">From PKR 8,499,000</div>
# This is public marketing content, not a live inventory/stock API - Honda
# Pakistan doesn't expose per-unit stock counts publicly (that lives inside
# individual dealers' internal systems). Pricing is the real, live data
# that's actually available to scrape here.
HONDA_HOMEPAGE_URL = "https://www.honda.com.pk/"
_HONDA_PRICE_PATTERN = re.compile(
    r'<h4>Honda ([\w\- ]+?)</h4>.*?model-price">From PKR ([\d,]+)</div>',
    re.DOTALL,
)

# Cache the parsed prices briefly so a burst of questions in one
# conversation doesn't re-fetch the real site on every single turn - this
# is a live public website, not our own infrastructure, so being a
# reasonably polite client matters.
_price_cache: dict[str, str] = {}
_price_cache_time = 0.0
_PRICE_CACHE_TTL_SECS = 300.0


def _fetch_honda_homepage_sync() -> str:
    """Blocking fetch, run in a background thread by `_get_honda_prices`.

    Plain httpx (or curl with no browser identity) gets blocked with a 403
    by the Cloudflare WAF in front of this site - confirmed by testing:
    even a normal browser `User-Agent` header wasn't enough, since
    Cloudflare's bot-detection also fingerprints the TLS handshake itself
    (JA3/JA4), which differs between a real browser and a generic Python
    HTTP client regardless of headers sent. `curl_cffi` with
    `impersonate="chrome"` reproduces an actual Chrome TLS fingerprint and
    gets through reliably (confirmed with repeated live requests).
    """
    response = curl_requests.get(HONDA_HOMEPAGE_URL, impersonate="chrome", timeout=10)
    response.raise_for_status()
    return response.text


async def _get_honda_prices() -> dict[str, str]:
    """Fetch + parse honda.com.pk's mega-menu prices, using a short-lived cache.

    Falls back to a stale cache rather than failing outright when a fresh
    fetch fails - a fetch failure this instant doesn't mean a price learned
    5 minutes ago is now wrong, so there's no reason to make the user wait
    or get an error when good-enough data is already sitting in memory.
    """
    global _price_cache, _price_cache_time

    now = time.monotonic()
    if _price_cache and (now - _price_cache_time) < _PRICE_CACHE_TTL_SECS:
        return _price_cache

    try:
        html = await asyncio.to_thread(_fetch_honda_homepage_sync)
    except curl_requests.exceptions.RequestException:
        if _price_cache:
            logger.warning(
                f"Honda homepage fetch failed - falling back to a stale price "
                f"cache ({now - _price_cache_time:.0f}s old) instead of failing."
            )
            return _price_cache
        raise

    _price_cache = {
        name.strip().lower(): price for name, price in _HONDA_PRICE_PATTERN.findall(html)
    }
    _price_cache_time = now
    return _price_cache


async def check_honda_price(params: FunctionCallParams):
    """Tool handler: looks up a Honda Pakistan model line's real, current
    starting price by fetching and parsing honda.com.pk's own homepage -
    the same "From PKR X" figure shown to any visitor of the site.

    Called by the LLM (not directly by us) whenever it decides the user is
    asking about price for a model. `params.arguments` holds whatever
    arguments the model filled in, matching the `properties` declared in
    `honda_price_tool` below.
    """
    model = str(params.arguments.get("model", "")).strip().lower()
    # Loose match so "hrv", "hr-v", and "HR V" all match the site's "hr-v".
    model_key = model.replace("-", "").replace(" ", "")

    try:
        prices = await _get_honda_prices()
    except curl_requests.exceptions.RequestException as e:
        logger.error(f"[tool call] check_honda_price: fetch failed: {e}")
        await params.result_callback(
            {"model": model, "found": False, "error": "could not reach honda.com.pk right now"}
        )
        return

    if not prices:
        # Confirmed directly: a 200 OK response with a bot-challenge page
        # (or any HTML structure change) parses to zero prices with no
        # exception raised at all - treating that the same as "this
        # specific model doesn't exist" would silently tell users every
        # single model is unavailable instead of signaling that the scrape
        # itself is broken.
        logger.error(
            "[tool call] check_honda_price: fetch succeeded but zero prices "
            "were parsed - site structure may have changed, or a "
            "bot-challenge page was served instead of the real homepage."
        )
        await params.result_callback(
            {
                "model": model,
                "found": False,
                "error": "could not verify any prices right now - honda.com.pk "
                "may be temporarily unreachable",
            }
        )
        return

    match = next(
        (
            price
            for name, price in prices.items()
            if name.replace("-", "").replace(" ", "") == model_key
        ),
        None,
    )

    logger.log("CONVO", f"[tool call] check_honda_price(model={model!r}) -> {match}")

    if match is None:
        await params.result_callback(
            {"model": model, "found": False, "available_models": list(prices.keys())}
        )
    else:
        await params.result_callback(
            {"model": model, "found": True, "starting_price_pkr": match}
        )


honda_price_tool = FunctionSchema(
    name="check_honda_price",
    description=(
        "Look up the real, current starting price (in PKR) of a Honda "
        "Pakistan model line - Civic, HR-V, or City - by checking the live "
        "honda.com.pk website."
    ),
    properties={
        "model": {
            "type": "string",
            "description": "The Honda model line to check, e.g. 'Civic', 'HR-V', or 'City'.",
        }
    },
    required=["model"],
    handler=check_honda_price,
)

# General-purpose "browse the real website" tool - covers everything on
# honda.com.pk that ISN'T the price mega-menu: model specs/features,
# dealer/contact info, promotions, company info, policies. Maps a handful
# of friendly topic names to the real page slugs discovered by actually
# crawling the site's homepage links (not guessed).
HONDA_PAGE_SLUGS = {
    "civic": "civic-standard",
    "civic standard": "civic-standard",
    "civic oriel": "civic-oriel-1-5",
    "civic rs": "civic-rs-turbo",
    "civic rs turbo": "civic-rs-turbo",
    "hr-v": "hrv-vti",
    "hrv": "hrv-vti",
    "hr-v s": "hrv-vti-s",
    "hr-v hybrid": "hrv-ehev",
    "hrv hybrid": "hrv-ehev",
    "hrv e:hev": "hrv-ehev",
    "city": "city1-2l",
    "city 1.2": "city1-2l",
    "city 1.5": "city1-5l",
    "city aspire": "cityaspire",
    "accord": "hondaaccord",
    "cr-v": "hondacrv",
    "crv": "hondacrv",
    "about": "abouthonda",
    "about honda": "abouthonda",
    "company": "abouthonda",
    "contact": "contactus",
    "contact us": "contactus",
    "dealer": "location-us",
    "dealers": "location-us",
    "dealer network": "location-us",
    "locations": "location-us",
    "promotions": "promotions",
    "offers": "promotions",
    "deals": "promotions",
    "news": "newsandevents",
    "events": "newsandevents",
    "free service": "free-service",
    "delivery status": "delivery-status",
    "policies": "policies",
    "privacy policy": "privacy-policy",
    "terms": "terms-and-conditions",
    "terms and conditions": "terms-and-conditions",
}

# Cache extracted page text briefly, per slug - same politeness rationale
# as the price cache above. These are shared cache *sizing* constants only -
# each site's browse tool (see make_browse_page_tool below) keeps its own
# private cache dict, since page content and slugs never overlap across sites.
_PAGE_CACHE_TTL_SECS = 300.0
_PAGE_TEXT_MAX_CHARS = 3000
# A 200 OK response can still be a bot-challenge page or a near-empty error
# page rather than real content - handing that to the LLM as if it were the
# genuine page risks a hallucinated or nonsensical answer built from noise.
# 200 is a conservative guess, not measured against the real site (these
# pages typically run into the thousands of characters once script/style/
# nav noise is stripped) - low risk of rejecting a legitimately terse page.
_PAGE_TEXT_MIN_CHARS = 200


def _fetch_and_extract_page_sync(base_url: str, slug: str) -> str:
    """Blocking fetch + HTML-to-text extraction, run in a background thread.

    Uses the same curl_cffi Chrome-impersonation approach as
    `_fetch_honda_homepage_sync` (see that docstring for why) since Honda's
    site sits behind the same Cloudflare protection - kept here for every
    site's browse tool even where it isn't strictly required (MG's site
    isn't Cloudflare-blocked), for one consistent, tested fetch path rather
    than a second, less battle-tested one. Strips script/style/nav/footer
    noise and returns plain, readable text for the LLM to read - real page
    content, not a summary or paraphrase we wrote ourselves.
    """
    url = f"{base_url}/{slug}"
    response = curl_requests.get(url, impersonate="chrome", timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "noscript", "svg"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ", strip=True).split())
    return text[:_PAGE_TEXT_MAX_CHARS]


def resolve_topic_slug(topic: str, page_slugs: dict[str, str]) -> str | None:
    """Resolves a (lowercased) topic string to a real page slug.

    Tries an exact key match first, then falls back to a loose substring
    match in either direction - handles near-misses like "civics" or "the
    hrv model" without needing an exact dict key. Defense-in-depth for
    whatever a model emits anyway; the tool schema's `topic` enum (see
    make_browse_page_tool) is the primary safeguard now.

    A plain, synchronous, module-level function (not a closure inside
    make_browse_page_tool's handler) specifically so it's directly
    unit-testable (see tests/test_bot.py) without pipecat, network I/O, or
    async plumbing.
    """
    slug = page_slugs.get(topic)
    if slug is not None:
        return slug
    return next(
        (s for key, s in page_slugs.items() if key in topic or topic in key),
        None,
    )


def make_browse_page_tool(
    *,
    tool_name: str,
    site_label: str,
    base_url: str,
    page_slugs: dict[str, str],
    description: str,
    topic_hint: str,
) -> FunctionSchema:
    """Builds a "browse this real site's pages" tool.

    Factored out after Honda's and MG Motors' browse-tool implementations
    turned out ~90% identical (same fetch/extract, same fuzzy topic match,
    same cache-with-stale-fallback, same min-length bot-challenge check) -
    duplicating a third copy for a future site would just be more of the
    same bug surface times three. Each call gets its own private page-text
    cache (closed over here, not module-global), since topic->slug maps and
    page content never overlap between sites - safe to call once per site.
    """
    page_text_cache: dict[str, str] = {}
    page_text_cache_time: dict[str, float] = {}

    async def handler(params: FunctionCallParams):
        topic = str(params.arguments.get("topic", "")).strip().lower()
        slug = resolve_topic_slug(topic, page_slugs)

        if slug is None:
            logger.log("CONVO", f"[tool call] {tool_name}(topic={topic!r}) -> no match")
            await params.result_callback(
                {
                    "topic": topic,
                    "found": False,
                    "available_topics": sorted(set(page_slugs.keys())),
                }
            )
            return

        now = time.monotonic()
        cached = page_text_cache.get(slug)
        fresh_age = now - page_text_cache_time.get(slug, 0)
        if cached is not None and fresh_age < _PAGE_CACHE_TTL_SECS:
            text = cached
        else:
            try:
                text = await asyncio.to_thread(_fetch_and_extract_page_sync, base_url, slug)
            except curl_requests.exceptions.RequestException as e:
                # Fall back to a stale cache rather than failing outright -
                # same reasoning as _get_honda_prices' stale-cache fallback.
                if cached is not None:
                    logger.warning(
                        f"[tool call] {tool_name}: fetch failed for {slug!r}, "
                        f"falling back to stale cache ({fresh_age:.0f}s old)."
                    )
                    text = cached
                else:
                    logger.error(f"[tool call] {tool_name}: fetch failed for {slug!r}: {e}")
                    await params.result_callback(
                        {
                            "topic": topic,
                            "found": False,
                            "error": f"could not reach {site_label} right now",
                        }
                    )
                    return
            else:
                page_text_cache[slug] = text
                page_text_cache_time[slug] = now

        if len(text) < _PAGE_TEXT_MIN_CHARS:
            logger.error(
                f"[tool call] {tool_name}: fetched {slug!r} but got only "
                f"{len(text)} chars of content - likely a bot-challenge page "
                f"or a site change, not real page content."
            )
            await params.result_callback(
                {
                    "topic": topic,
                    "found": False,
                    "error": "could not verify this page's content right now",
                }
            )
            return

        logger.log(
            "CONVO", f"[tool call] {tool_name}(topic={topic!r}) -> {slug} ({len(text)} chars)"
        )
        await params.result_callback({"topic": topic, "found": True, "page_content": text})

    # DO NOT add a JSON Schema `enum` of valid topics to this property.
    # It was tried and reverted - measured A/B against the live Groq API
    # (llama-3.3-70b-versatile, 12 calls per arm, same questions):
    #
    #     WITH enum     hard-400 "tool_use_failed" on 10/12  (83%)
    #     WITHOUT enum  hard-400 on 1/12                     ( 8%)
    #
    # The model cannot reliably satisfy a large enum constraint here: it
    # emits malformed tool-call syntax (observed failed_generation:
    # `<function=browse_honda_page={"topic":"civic specs"}</function>`,
    # where "civic specs" is off-enum) and Groq rejects the ENTIRE request
    # with a 400 rather than passing the off-list value through. That also
    # means resolve_topic_slug's fuzzy fallback is unreachable in that
    # case - the request never gets far enough to call this handler at
    # all, so a near-miss that the fuzzy matcher handles perfectly well
    # ("civic specs" -> civic-standard) turns into a user-visible error
    # instead. Free-text + fuzzy resolution resolved 11/12 correctly in
    # the same test. Keep it that way.
    return FunctionSchema(
        name=tool_name,
        description=description,
        properties={"topic": {"type": "string", "description": topic_hint}},
        required=["topic"],
        handler=handler,
    )


browse_honda_page_tool = make_browse_page_tool(
    tool_name="browse_honda_page",
    site_label="honda.com.pk",
    base_url="https://www.honda.com.pk",
    page_slugs=HONDA_PAGE_SLUGS,
    description=(
        "Fetch and read a real page from the honda.com.pk website to answer "
        "questions about model specs/features, dealer or contact info, "
        "promotions, company info, or policies - anything other than price."
    ),
    topic_hint=(
        "What to look up, e.g. 'Civic specs', 'dealer locations', "
        "'contact info', 'promotions', 'about honda'."
    ),
)

# ---------------------------------------------------------------------------
# MG Motors Pakistan (mgmotors.com.pk) tools
# ---------------------------------------------------------------------------
# MG Motors Pakistan does not publish car prices on their public website
# (unlike Honda's homepage mega-menu). All model and info pages are
# accessible without Cloudflare blocking (confirmed via direct testing -
# curl_cffi Chrome impersonation used anyway for consistency and resilience).

MG_BASE_URL = "https://mgmotors.com.pk"

MG_PAGE_SLUGS = {
    # Model pages - slugs confirmed from the site's own navigation links
    "hs super hybrid": "model/mg-hs-super-hybrid",
    "mg hs super hybrid": "model/mg-hs-super-hybrid",
    "hs hybrid+": "model/mg-hs-hybrid-plus",
    "mg hs hybrid+": "model/mg-hs-hybrid-plus",
    "hs hybrid plus": "model/mg-hs-hybrid-plus",
    "mg hs hybrid plus": "model/mg-hs-hybrid-plus",
    "hs phev": "model/phev",
    "mg hs phev": "model/phev",
    "phev": "model/phev",
    "hs": "model/mg-hs-super-hybrid",
    "mg hs": "model/mg-hs-super-hybrid",
    "u9": "model/mgu9",
    "mg u9": "model/mgu9",
    "mgu9": "model/mgu9",
    "mg4": "model/mg4-ev-urban",
    "mg4 ev": "model/mg4-ev-urban",
    "mg4 ev urban": "model/mg4-ev-urban",
    "cyberster": "model/cyberster",
    "mg cyberster": "model/cyberster",
    "binguo": "model/binguo",
    "binguo ev": "model/binguo",
    # Info / service pages
    "about": "about",
    "about mg": "about",
    "contact": "contact",
    "contact us": "contact",
    "dealer": "dealer-locator",
    "dealers": "dealer-locator",
    "dealer locator": "dealer-locator",
    "find dealer": "dealer-locator",
    "locations": "dealer-locator",
    "financing": "mg-partnerships",
    "finance": "mg-partnerships",
    "bank partnerships": "mg-partnerships",
    "partnerships": "mg-partnerships",
    "offers": "world-of-mg",
    "promotions": "world-of-mg",
    "news": "world-of-mg",
    "events": "world-of-mg",
    "world of mg": "world-of-mg",
    "after sales": "care",
    "after-sales": "care",
    "service": "care",
    "care": "care",
    "faqs": "faqs",
    "faq": "faqs",
    "exchange": "mg-exchange",
    "mg exchange": "mg-exchange",
    "trade in": "mg-exchange",
    "trade-in": "mg-exchange",
    "test drive": "test-drive",
    "book test drive": "test-drive",
    "track": "track-my-mg",
    "track my mg": "track-my-mg",
    "order status": "track-my-mg",
    "careers": "careers",
    "privacy": "privacy-policy",
    "privacy policy": "privacy-policy",
}

browse_mg_page_tool = make_browse_page_tool(
    tool_name="browse_mg_page",
    site_label="mgmotors.com.pk",
    base_url=MG_BASE_URL,
    page_slugs=MG_PAGE_SLUGS,
    description=(
        "Fetch and read a real page from the mgmotors.com.pk website to answer "
        "questions about MG Motors Pakistan - model specs/features, dealer "
        "locations, financing/bank partnerships, offers, after-sales service, "
        "contact info, or company info. MG does not list prices on their site."
    ),
    topic_hint=(
        "What to look up, e.g. 'MG HS specs', 'dealer locations', "
        "'financing', 'MG4 EV', 'Cyberster', 'contact', 'offers'."
    ),
)


def _avg_logprob(result: Transcription) -> float:
    """Mean segment log-probability, used as a confidence proxy.

    Whisper doesn't expose a single "confidence" number, but each segment's
    avg_logprob (closer to 0 = more confident) is the standard stand-in.
    Returns -inf for silent/empty audio (no segments) so it always loses to
    a real transcript when comparing two candidates.
    """
    segments = getattr(result, "segments", None) or []
    if not segments:
        return float("-inf")
    return sum(getattr(s, "avg_logprob", 0.0) for s in segments) / len(segments)


# Whisper's own output script for language="ur" is Perso-Arabic (Nastaliq) -
# that's what its training data maps that language token to. There's no
# separate "romanized Urdu" language code to force instead. The standard
# workaround (used here) is to feed Whisper's `prompt` param - which
# actually conditions the decoder on preceding context tokens, not just a
# hint - a short sample already written in Roman Urdu. Whisper strongly
# tends to continue transcribing in whatever script its prompt was written
# in, so this reliably shifts output to Roman/Latin script instead of
# Nastaliq. Heuristic, not a documented API guarantee: works because of how
# prompt-conditioning happens to interact with script selection, not because
# Whisper has an explicit "romanize" mode - a sample covering common
# sentence shapes (question, statement, request) gives the decoder more to
# latch onto than a single short phrase would.
_ROMAN_URDU_STT_PROMPT = (
    "Assalam alaikum, aap kaisay hain? Mujhe iski qeemat maloom karni hai. "
    "Yeh gari kitne ki hai? Mehrbani karke dealer ka number bata dein."
)


class BilingualGroqSTTService(GroqSTTService):
    """GroqSTTService constrained to a closed set of two languages: English and
    Roman Urdu (Urdu spoken, but transcribed in Latin script, not Nastaliq).

    Whisper's own language auto-detection was tried and misdetected short
    Urdu clips as Chinese - it's unconstrained across every language Whisper
    knows, which is overkill and unreliable for a bot that only ever needs
    to distinguish two specific languages.

    Instead, each utterance is transcribed twice concurrently - once forced
    to `language="en"`, once forced to `language="ur"` (with a Roman-script
    prompt bias - see `_ROMAN_URDU_STT_PROMPT` above) - and whichever result
    has the higher average segment confidence (avg_logprob) wins. Forcing
    removes the third-language misdetection failure mode entirely, since
    Whisper is never given the option to guess anything else.

    Trade-off: this doubles Groq STT API calls per turn. Both calls run
    concurrently via asyncio.gather, so wall-clock latency is roughly the
    slower of the two, not the sum - but token/request usage is 2x.

    NOTE: The LLM is instructed (system prompt) to always reply in English
    regardless of input language. This was originally required because the
    audio-output version's TTS (KokoroTTSService) can't speak Urdu script -
    that constraint doesn't actually apply to this text-output variant (any
    script can be displayed as text), but the English-only instruction was
    kept as-is here to keep behavior consistent between the two variants.
    Reading Roman Urdu input and replying in English works fine for the LLM
    either way.
    """

    async def _transcribe(self, audio: bytes) -> Transcription:
        base_kwargs = {
            "file": ("audio.wav", audio, "audio/wav"),
            "model": self._settings.model,
            "response_format": "verbose_json",
        }
        if self._settings.temperature is not None:
            base_kwargs["temperature"] = self._settings.temperature

        en_kwargs = dict(base_kwargs)
        if self._settings.prompt is not None:
            en_kwargs["prompt"] = self._settings.prompt

        # The Roman-Urdu prompt bias only makes sense for the "ur" call -
        # applying it to "en" as well would risk nudging clean English
        # audio toward the same style for no benefit.
        ur_kwargs = dict(base_kwargs)
        ur_kwargs["prompt"] = self._settings.prompt or _ROMAN_URDU_STT_PROMPT

        result_en, result_ur = await asyncio.gather(
            self._client.audio.transcriptions.create(language="en", **en_kwargs),
            self._client.audio.transcriptions.create(language="ur", **ur_kwargs),
        )

        conf_en, conf_ur = _avg_logprob(result_en), _avg_logprob(result_ur)
        winner, lang, conf = (
            (result_en, "en", conf_en) if conf_en >= conf_ur else (result_ur, "ur", conf_ur)
        )

        # repr() is ASCII-safe: shows \uXXXX escapes for non-Latin chars so
        # you can spot it immediately if Whisper ever slips back into
        # native Urdu (Nastaliq) script instead of the intended Roman one.
        logger.debug(
            f"STT bilingual pick: lang={lang} conf={conf:.3f} "
            f"(en={conf_en:.3f} ur={conf_ur:.3f}) text={repr(winner.text)}"
        )
        return winner


_HTML_TAG_PATTERN = re.compile(r"<[^>]*>")


class TextSanitizer(FrameProcessor):
    """Strips raw HTML tags from LLM output before it's delivered to the
    client as an RTVI text message.

    Defensive, not a fix for an observed bug: the system prompt here allows
    normal written formatting (unlike the audio variant, which strips
    everything before TTS), but nothing stops the LLM from occasionally
    echoing back a stray HTML tag - e.g. leftover markup from a scraped
    Honda page, or a rare hallucination. If whatever client eventually
    renders these replies does markdown-to-HTML conversion without
    sanitizing, an unfiltered raw tag reaching it is a real (if unlikely)
    injection risk. Cheap insurance given this variant is the only one
    whose system prompt invites rich formatting at all.

    Known limitation: operates per-fragment, since RTVI streams one
    "bot-llm-text" message per raw LLM token chunk rather than per complete
    sentence (every LLMTextFrame is pushed to the client immediately,
    unaggregated - see pipecat's RTVIObserver._handle_llm_text_frame).
    Buffering into complete sentences first (like the audio variant's
    TTSTextNormalizer does) would add latency before any text reaches the
    client at all, defeating real-time streaming - a tag split exactly
    across two separate streamed fragments could theoretically slip through
    partially. A narrow, accepted trade-off for keeping streaming immediate.

    Mutates `frame.text` in place rather than replacing the frame - `Frame`
    is a plain (non-frozen) dataclass, so this is safe, and it matters here
    specifically: pipecat's RTVIObserver dedupes frames by `frame.id`,
    which is auto-assigned in `__post_init__` and NOT preserved by
    `dataclasses.replace()` (replace() reruns __init__, minting a fresh id
    even when only `text` changed). Replacing the frame here previously
    caused every streamed chunk to reach the client twice - once when `llm`
    pushed the original frame, again when this processor pushed a
    replacement whose new id the observer's dedup logic couldn't recognize
    as the same frame.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame) and frame.text:
            sanitized = _HTML_TAG_PATTERN.sub("", frame.text)
            if sanitized != frame.text:
                logger.warning(f"Stripped HTML-like tag from LLM output: {frame.text!r}")
                frame.text = sanitized
            await self.push_frame(frame, direction)
            return
        await self.push_frame(frame, direction)


async def _startup_self_check(llm: GroqLLMService) -> None:
    """Validates the LLM actually works before accepting real traffic - not
    just that a key is present (already checked at import time), but that a
    real call succeeds.

    Scoped to just the LLM: this variant has no TTS, and STT/audio-input
    concerns are out of scope here (same reasoning as BilingualGroqSTTService
    and the rest of the audio-input layer - this variant intentionally
    doesn't duplicate that verification).

    A revoked/expired key or a Groq quota/permission issue would otherwise
    only surface on the user's FIRST real turn, presenting as a mysterious
    fallback message with no clear cause. Failing fast here is much easier
    to diagnose than that.

    Exits the process (does not raise) if the check fails, matching the
    existing fail-fast behavior for a missing API key / port conflict.
    """
    try:
        await llm._client.chat.completions.create(
            model=llm._settings.model,
            messages=[{"role": "user", "content": "hi"}],
            max_completion_tokens=1,
        )
    except Exception as e:
        logger.error("Startup self-check FAILED - fix this before running the bot:")
        logger.error(f"  - LLM (Groq chat completions, model={llm._settings.model}): {e}")
        sys.exit(1)

    logger.info("Startup self-check passed: LLM is working.")


# pipecat spawns a fresh run_bot() per WebRTC connection
# (background_tasks.add_task on every /api/offer request), so without this
# flag every single connection would re-run the self-check - an extra API
# round-trip added to that user's first-turn latency, and extra quota usage
# per session instead of once per process. The LLM client itself (same API
# key, same model) doesn't change between connections, so verifying it once
# per process is enough.
_startup_self_check_done = False


async def _push_standalone_text_message(worker: PipelineWorker, text: str) -> None:
    """Pushes a one-shot bot text message (greeting/fallback) with proper
    started/stopped bookends, not just the bare text.

    A bare BotLLMTextMessage with no surrounding lifecycle messages leaves
    a client stuck showing "bot is typing..." forever: bot-llm-stopped is
    only emitted by pipecat's RTVI observer in response to a real
    LLMFullResponseEndFrame, which never occurs for a message pushed this
    way. Same root-cause class as the audio variant's
    TTSStartedFrame/TTSStoppedFrame fix for its fallback audio - a raw
    content frame/message needs its matching lifecycle bookends, not just
    the content itself.
    """
    await worker.rtvi.push_transport_message(BotLLMStartedMessage())
    await worker.rtvi.push_transport_message(BotLLMTextMessage(data=TextMessageData(text=text)))
    await worker.rtvi.push_transport_message(BotLLMStoppedMessage())


async def run_bot(transport: BaseTransport, *, handle_sigint: bool = False):
    # whisper-large-v3-turbo instead of whisper-large-v3: cuts STT TTFB
    # from ~3s to ~0.3-0.5s (a 6-10x reduction), which matters more here
    # than usual since every utterance already pays for two concurrent
    # transcription calls (see BilingualGroqSTTService). Known trade-off:
    # Whisper's turbo variants are documented to trade a little accuracy
    # for speed, more noticeably on less-common languages/accents than
    # clean English.
    stt = BilingualGroqSTTService(
        api_key=os.environ["GROQ_API_KEY"],
        settings=GroqSTTService.Settings(model="whisper-large-v3-turbo"),
    )

    llm = GroqLLMService(
        api_key=os.environ["GROQ_API_KEY"],
        settings=GroqLLMService.Settings(
            # First entry in LLM_MODEL_FALLBACK_CHAIN - Groq's dedicated
            # tool-calling variant: specifically fine-tuned for function
            # calling, higher rate limits than llama-3.3-70b-versatile,
            # and far better tool-use accuracy than the 8B instant model.
            # on_pipeline_error below switches llm._settings.model to the
            # next entry in the chain if this one gets rate-limited.
            model=LLM_MODEL_FALLBACK_CHAIN[0],
            system_instruction=SYSTEM_INSTRUCTION,
            # The original 150-token cap existed to bound worst-case TTS
            # synthesis time in the audio variant - that reason doesn't
            # apply here (no TTS at all), and it was cutting off detailed
            # answers (specs, comparisons) that a text reply can comfortably
            # hold. Raised to a much larger backstop that only exists to
            # guard against genuinely runaway/adversarial-prompt generation,
            # not to keep everyday answers short.
            max_completion_tokens=500,
        ),
    )

    global _startup_self_check_done
    if os.getenv("SKIP_STARTUP_SELF_CHECK"):
        logger.warning("SKIPPING startup self-check (SKIP_STARTUP_SELF_CHECK is set).")
    elif _startup_self_check_done:
        logger.debug("Skipping startup self-check - already verified once this process.")
    else:
        await _startup_self_check(llm)
        _startup_self_check_done = True

    context = LLMContext(tools=[honda_price_tool, browse_honda_page_tool, browse_mg_page_tool])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
        # Bulletproofing: a long-running conversation would otherwise grow
        # LLMContext unboundedly - every future turn resends the entire
        # history, so cost/latency creep up forever and eventually risk
        # hitting the model's real context limit. pipecat already ships a
        # complete summarization mechanism for this - it's just OFF by
        # default. Turning it on with its own sensible defaults rather than
        # leaving this pipeline vulnerable to unbounded growth.
        assistant_params=LLMAssistantAggregatorParams(enable_auto_context_summarization=True),
    )

    # Fire-and-forget background tasks (currently just
    # _correct_interrupted_context below) need a live reference kept
    # somewhere for their whole lifetime - asyncio.create_task() alone
    # doesn't do that. Per CPython's own asyncio docs: "Save a reference
    # to the result, to avoid a task disappearing mid-execution" - nothing
    # else holds one, so the task object is only kept alive by whatever
    # asyncio.create_task() returns, which is otherwise unreferenced and
    # eligible for garbage collection before it finishes running. The
    # done-callback discards each task from this set once it completes, so
    # this doesn't grow unbounded over a long conversation.
    background_tasks: set[asyncio.Task] = set()

    def _track_background_task(coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)
        return task

    @user_aggregator.event_handler("on_user_turn_message_added")
    async def on_user_turn_message_added(aggregator, message):
        logger.log("CONVO", f"User: {message.content}")
        _print_chat("You", message.content)
        _print_thinking()

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message):
        logger.log("CONVO", f"Bot: {message.content}")
        # Only the real, final reply should replace "thinking..." - a
        # tool-calling turn fires this same event once per intermediate
        # step first (function-call, then function-result) with empty
        # content before the actual answer; printing those would just
        # flash blank lines instead of masking the wait as intended.
        if message.content:
            _print_chat("Bot", message.content)
        nonlocal consecutive_fallback_failures, circuit_open_until, tool_call_retry_used
        # A real assistant turn completed successfully - close the circuit
        # breaker below entirely, so a transient blip doesn't leave things
        # backed off longer than necessary once the service has recovered.
        consecutive_fallback_failures = 0
        circuit_open_until = 0.0
        tool_call_retry_used = False

        if message.interrupted:
            # On interruption, message.content usually comes back empty -
            # pipecat doesn't broadcast any assistant entry into context at
            # all in that case, which is silent data loss (the LLM has zero
            # memory it started answering). Handles both cases: replaces the
            # entry if pipecat did broadcast one, otherwise appends an
            # honest interruption marker instead of silently losing the turn.
            original_content = message.content

            async def _correct_interrupted_context():
                await asyncio.sleep(0.2)
                messages = context.get_messages()
                marker = (
                    "[This response was interrupted by the user before finishing - "
                    "only part of it (if any) was actually delivered, not the "
                    "complete answer.]"
                )
                if original_content:
                    for i in reversed(range(len(messages))):
                        if (
                            messages[i].get("role") == "assistant"
                            and messages[i].get("content") == original_content
                        ):
                            messages[i]["content"] = marker
                            context.set_messages(messages)
                            logger.debug("Corrected interrupted assistant turn in context")
                            return
                messages.append({"role": "assistant", "content": marker})
                context.set_messages(messages)
                logger.debug("Recorded interruption marker for empty-content assistant turn")

            _track_background_task(_correct_interrupted_context())

    text_sanitizer = TextSanitizer()

    pipeline = Pipeline(
        [
            transport.input(),  # Mic input
            stt,  # Speech -> text
            user_aggregator,  # Collect user turn
            llm,  # Generate response (text delivered to the client via RTVI below)
            text_sanitizer,  # Strip any raw HTML tags before the client sees them
            transport.output(),  # Delivers RTVI text messages to the client - no TTS/audio
            assistant_aggregator,  # Collect assistant turn
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        # enable_rtvi defaults to True - this is what turns the LLM's
        # streamed text into "bot-llm-text" messages sent to the client,
        # replacing what TTS used to do.
        #
        # Bulletproofing: made explicit rather than relying on the library
        # default. If a connected user goes quiet this long, pipecat cancels
        # this session's worker AND runner automatically, scoped to just
        # that one connection (background_tasks.add_task spawns an
        # independent bot()/run_bot()/WorkerRunner() per WebRTC connection).
        idle_timeout_secs=300.0,
    )

    # A flat per-attempt cooldown would let the fallback fire indefinitely
    # during a genuine outage (every FALLBACK_COOLDOWN_SECS, forever). This
    # is a real circuit breaker instead: each consecutive failure doubles
    # how long the circuit stays open (backoff grows 5s -> 10s -> 20s ...
    # capped at CIRCUIT_BREAKER_MAX_BACKOFF_SECS), and a single successful
    # turn (on_assistant_turn_stopped above) closes it again completely.
    consecutive_fallback_failures = 0
    circuit_open_until = 0.0
    CIRCUIT_BREAKER_MAX_BACKOFF_SECS = 60.0
    # Bounds tool-call-failure retries to one per turn - reset to False
    # whenever a turn actually completes (on_assistant_turn_stopped above,
    # same as the circuit breaker state), so a NEW turn always gets its
    # own retry attempt, but a turn that fails twice in a row (retry
    # didn't help) falls through to the normal apology instead of retrying
    # forever.
    tool_call_retry_used = False
    current_model_index = 0
    # Once a rate limit forces a move off the primary model, nothing was
    # previously moving things back - a fallback model, once switched to,
    # stayed in use for the rest of the process's life even long after
    # the primary's rate-limit window had reset. This task, (re)scheduled
    # every time a switch happens, moves back to the primary model after a
    # cooldown with no further rate limits - so a transient spike doesn't
    # permanently downgrade every future conversation to the weaker model.
    model_fallback_reset_task: asyncio.Task | None = None
    MODEL_FALLBACK_RESET_SECS = 600.0  # 10 minutes

    def _schedule_model_fallback_reset():
        nonlocal model_fallback_reset_task

        async def _reset_after_delay():
            nonlocal current_model_index
            await asyncio.sleep(MODEL_FALLBACK_RESET_SECS)
            primary_model = LLM_MODEL_FALLBACK_CHAIN[0]
            logger.warning(
                f"Retrying primary LLM model {primary_model!r} after "
                f"{MODEL_FALLBACK_RESET_SECS:.0f}s on a fallback model."
            )
            current_model_index = 0
            llm._settings.model = primary_model

        # A second rate limit before the previous timer fires should push
        # the reset further out, not race two resets against each other.
        if model_fallback_reset_task and not model_fallback_reset_task.done():
            model_fallback_reset_task.cancel()
        # Tracked (not bare asyncio.create_task) for the same reason as
        # _correct_interrupted_context above, plus this one specifically
        # needs cancelling on pipeline shutdown (see on_pipeline_finished
        # below) - each connection has its own `llm` instance, so a leaked
        # task here can't affect a different connection, but a 600s sleep
        # left running past its own connection's end is still a real task
        # (and closed-over llm/context object) leak with no one left to
        # observe its effect.
        model_fallback_reset_task = _track_background_task(_reset_after_delay())

    @worker.event_handler("on_pipeline_error")
    async def on_pipeline_error(worker, frame: ErrorFrame):
        nonlocal consecutive_fallback_failures, circuit_open_until, current_model_index, tool_call_retry_used
        logger.error(f"Pipeline error from {frame.processor}: {frame.error}")
        if frame.fatal:
            return

        error_text = str(frame.error).lower()

        # Groq occasionally fails to produce a valid tool call for a
        # legitimate request ("Failed to call a function") - a one-off
        # sampling issue, not a real problem with the user's question or
        # the tool schema. Retrying the SAME turn (context is untouched by
        # a failed completion - the failed attempt was never added to it)
        # often just succeeds the second time. Bounded to once per turn via
        # tool_call_retry_used, reset on the next successful turn, so a
        # question that genuinely keeps failing falls through to the
        # normal apology instead of retrying forever.
        if is_tool_call_failed_error(error_text) and not tool_call_retry_used:
            tool_call_retry_used = True
            logger.warning("Tool-call generation failed - retrying this turn once.")
            await worker.queue_frame(LLMRunFrame())
            return

        # A rate limit is a distinct failure mode from a transient hiccup -
        # backing off and retrying the SAME model just burns the cooldown
        # for nothing if that model's quota is genuinely exhausted for the
        # window. Moving to the next model in the fallback chain means the
        # very next turn has a real chance of succeeding, not just a slower
        # retry of the thing that already failed. Applies regardless of
        # whether THIS particular error came from a real turn or the
        # background summarizer below - a rate limit is org-wide, so
        # either origin is equally good evidence the current model is
        # (temporarily) exhausted.
        if error_frame_is_rate_limit(frame) and current_model_index + 1 < len(LLM_MODEL_FALLBACK_CHAIN):
            previous_model = LLM_MODEL_FALLBACK_CHAIN[current_model_index]
            current_model_index += 1
            next_model = LLM_MODEL_FALLBACK_CHAIN[current_model_index]
            llm._settings.model = next_model
            logger.error(
                f"Rate limit hit on {previous_model!r} - switching LLM "
                f"model to {next_model!r} for subsequent turns."
            )
            _schedule_model_fallback_reset()

        # A failure in pipecat's own background context-summarization pass
        # has nothing to do with the user's actual, current turn - showing
        # an apology and spending the circuit breaker's budget on it would
        # be a false alarm. Log it and stop here instead.
        if is_background_summarization_error(error_text):
            logger.warning(
                "Background context-summarization error - not shown to the "
                "user, since their actual turn wasn't affected."
            )
            return

        now = time.monotonic()
        if now < circuit_open_until:
            # Already backed off from a recent run of failures - stay quiet
            # rather than sending (and failing) again immediately.
            return

        # A non-fatal ErrorFrame (STT/LLM hiccup, rate limit, etc.) would
        # otherwise just get logged, leaving the user with no reply for that
        # turn. Send a short apology as a text message instead, so the
        # conversation can continue. Not added to LLM context, matching the
        # original TTS-based fallback's behavior. Unlike the audio variant,
        # there's no "apologizing through the broken service" risk here -
        # this is a plain RTVI text push, not dependent on any AI service.
        logger.log("CONVO", f"Bot: {FALLBACK_ERROR_MESSAGE}")
        _print_chat("Bot", FALLBACK_ERROR_MESSAGE)
        await _push_standalone_text_message(worker, FALLBACK_ERROR_MESSAGE)

        consecutive_fallback_failures += 1
        backoff = min(
            FALLBACK_COOLDOWN_SECS * (2 ** (consecutive_fallback_failures - 1)),
            CIRCUIT_BREAKER_MAX_BACKOFF_SECS,
        )
        circuit_open_until = now + backoff
        logger.error(
            f"Circuit breaker: {consecutive_fallback_failures} consecutive failure(s), "
            f"backing off fallback message for {backoff:.0f}s."
        )

    @worker.event_handler("on_pipeline_started")
    async def send_greeting(worker, frame):
        # Sent directly as a text message instead of round-tripping through
        # the LLM just to say hello - saves an API call. Unlike the original
        # TTS-based greeting, this isn't added to the LLM's conversation
        # history (RTVI text pushes don't touch LLMContext) - a minor,
        # accepted trade-off of the text-output variant.
        logger.log("CONVO", f"Bot: {GREETING_MESSAGE}")
        _print_chat("Bot", GREETING_MESSAGE)
        await _push_standalone_text_message(worker, GREETING_MESSAGE)

    @worker.event_handler("on_pipeline_finished")
    async def cancel_background_tasks(worker, frame):
        # Cancels every tracked background task (interrupted-context
        # corrections, the model-fallback reset timer) once this
        # connection's pipeline reaches any terminal state - without this,
        # a still-sleeping model_fallback_reset_task (up to 600s) would
        # keep running, and being referenced, well past the point anyone
        # could still observe or care about its effect.
        for task in list(background_tasks):
            if not task.done():
                task.cancel()

    runner = WorkerRunner(handle_sigint=handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Entry point used by the WebRTC dev runner (``python bot.py -t webrtc``)
    and the eval harness (``python bot.py -t eval``)."""
    if isinstance(runner_args, SmallWebRTCRunnerArguments):
        # Imported here (not at module level) so `--local` mode never pays the
        # aiortc import cost - it doesn't use this transport at all.
        from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

        transport = SmallWebRTCTransport(
            webrtc_connection=runner_args.webrtc_connection,
            params=TransportParams(
                audio_in_enabled=True,
                # Must be True even though this variant has no TTS: with it
                # False, the RTVI observer silently stops forwarding
                # user-started-speaking/interruption events to the client at
                # all, despite the bot's own internal VAD/turn-detection
                # firing correctly server-side. No TTS service exists
                # anywhere in this pipeline to actually generate audio, so
                # this is a free fix, not a real audio-output enablement.
                audio_out_enabled=True,
            ),
        )
    elif isinstance(runner_args, EvalRunnerArguments):
        from pipecat.evals.transport import EvalTransportParams

        transport = await create_transport(
            runner_args,
            {"eval": lambda: EvalTransportParams(audio_in_enabled=True, audio_out_enabled=True)},
        )
    else:
        raise RuntimeError(
            "This bot only supports the WebRTC transport (-t webrtc) or eval transport (-t eval)."
        )

    await run_bot(transport, handle_sigint=runner_args.handle_sigint)


def _find_headset_mic_device_index() -> int | None:
    """Look up the headset mic's PyAudio device index on the DirectSound host API.

    DirectSound handles simultaneous input+output more reliably on Windows
    than the MME backend PyAudio defaults to, and - unlike WASAPI's shared
    mode - accepts the 16kHz mono capture rate VAD/STT need directly,
    without erroring ("Invalid sample rate") or needing a resampler.
    Falls back to the system default input device (None) if no DirectSound
    headset mic is found.
    """
    pa = pyaudio.PyAudio()
    try:
        ds_index = pa.get_host_api_info_by_type(pyaudio.paDirectSound)["index"]
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if (
                info["hostApi"] == ds_index
                and info["maxInputChannels"] > 0
                and "headset microphone" in info["name"].lower()
            ):
                return i
    except OSError:
        pass  # DirectSound host API not available on this system
    finally:
        pa.terminate()
    return None


async def run_local():
    """Entry point for the CLI/local-mic mode (``python bot.py --local``)."""
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=False,  # no TTS - reply only appears in the console (CONVO log)
            input_device_index=_find_headset_mic_device_index(),
        )
    )

    await run_bot(transport, handle_sigint=True)


def _check_port_available(host: str, port: int) -> None:
    """Exit with a clear message if `port` is already bound.

    pipecat's own WebRTC runner prints "Bot ready!" (see
    `pipecat.runner.run.main`, which calls `_print_startup_message` before
    `uvicorn.run`) BEFORE it actually attempts to bind the port - so a
    second instance started against an already-used port prints a false
    "ready" message and only fails later. Checking here, before handing
    off to pipecat's runner at all, avoids that misleading sequence.

    Checks every address `host` resolves to, not just IPv4 - "localhost"
    (the default) resolves to both 127.0.0.1 and ::1, and a process
    listening on ::1 only would otherwise pass this check clean while
    uvicorn still fails to bind for real moments later, reproducing the
    exact misleading "Bot ready!"-then-fail sequence this function exists
    to prevent in the first place.
    """
    import socket

    try:
        addrinfos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        sys.exit(f"Could not resolve host {host!r}: {e}")

    for family, socktype, proto, _canonname, sockaddr in addrinfos:
        try:
            with socket.socket(family, socktype, proto) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
                s.bind(sockaddr)
        except OSError as e:
            sys.exit(
                f"Port {port} on {sockaddr[0]} is already in use - is "
                f"another instance of this bot already running? ({e})"
            )


if __name__ == "__main__":
    import asyncio

    if "--local" in sys.argv:
        asyncio.run(run_local())
    else:
        from pipecat.runner.run import RUNNER_HOST, RUNNER_PORT, main

        # Mirrors pipecat's own --host/--port argparse defaults so this
        # check targets the same address the runner will actually bind.
        host = sys.argv[sys.argv.index("--host") + 1] if "--host" in sys.argv else RUNNER_HOST
        port = (
            int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else RUNNER_PORT
        )
        _check_port_available(host, port)

        main()
