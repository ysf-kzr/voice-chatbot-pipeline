"""Unit tests for bot.py's pure, network-free functions.

Deliberately scoped to logic that doesn't need pipecat's runtime, a live
Groq connection, or real network access - the eval suite in evals/ already
covers full pipeline behavior end-to-end (Kokoro-synthesized speech through
the real bot). These run in well under a second and exist for the pieces
that should never need several seconds and a live API key to verify -
resolve_topic_slug's real-data cases below are the exact function/inputs
used to measure the 30% topic-resolution hit rate that motivated adding the
`enum` constraint to the tool schema; a test like this would have caught
that regression immediately instead of needing a live session to surface it.

Run (from the repo root, using the project's pinned venv):
    <path-to-venv>/Scripts/python.exe -m pytest tests/ -v
"""

import socket

import httpx
import openai
import pytest

import bot


class TestAvgLogprob:
    def test_no_segments_returns_negative_infinity(self):
        result = type("Result", (), {"segments": []})()
        assert bot._avg_logprob(result) == float("-inf")

    def test_none_segments_returns_negative_infinity(self):
        result = type("Result", (), {"segments": None})()
        assert bot._avg_logprob(result) == float("-inf")

    def test_computes_mean_of_segment_logprobs(self):
        segments = [
            type("Segment", (), {"avg_logprob": -0.1})(),
            type("Segment", (), {"avg_logprob": -0.3})(),
        ]
        result = type("Result", (), {"segments": segments})()
        assert bot._avg_logprob(result) == pytest.approx(-0.2)

    def test_missing_avg_logprob_attribute_defaults_to_zero(self):
        segments = [type("Segment", (), {})()]  # no avg_logprob attr at all
        result = type("Result", (), {"segments": segments})()
        assert bot._avg_logprob(result) == 0.0


class TestResolveTopicSlug:
    SLUGS = {"civic": "civic-standard", "hr-v": "hrv-vti", "about honda": "abouthonda"}

    def test_exact_match(self):
        assert bot.resolve_topic_slug("civic", self.SLUGS) == "civic-standard"

    def test_fuzzy_match_when_a_real_key_appears_inside_the_topic(self):
        topic = "tell me about the civic specs"
        assert bot.resolve_topic_slug(topic, self.SLUGS) == "civic-standard"

    def test_fuzzy_match_when_the_topic_is_a_substring_of_a_real_key(self):
        assert bot.resolve_topic_slug("honda", self.SLUGS) == "abouthonda"

    def test_no_match_returns_none(self):
        assert bot.resolve_topic_slug("completely unrelated topic", self.SLUGS) is None

    def test_empty_slug_dict_returns_none(self):
        assert bot.resolve_topic_slug("anything", {}) is None


class TestResolveTopicSlugRealData:
    """Regression tests against the actual production slug dicts, using
    real topic strings this pipeline's LLM emitted in a live session
    (captured from server logs) - not synthetic examples.

    The "still miss" cases below are documented gaps in the fuzzy fallback
    alone, not bugs to fix here - they're exactly what motivated adding
    the `enum` constraint to the tool schema (see make_browse_page_tool)
    rather than trying to patch the matching heuristic itself, which can't
    reliably guess arbitrary free-text phrasing. The enum now stops the
    model from emitting most of these in normal operation; this function
    is defense-in-depth for whatever still gets through, not the primary
    safeguard - so these are pinned as known behavior, not xfail.
    """

    @pytest.mark.parametrize(
        "page_slugs,topic,expected_slug",
        [
            (bot.HONDA_PAGE_SLUGS, "honda company info and models available in pakistan", "abouthonda"),
            (bot.HONDA_PAGE_SLUGS, "honda civic specs and features", "civic-standard"),
            (bot.MG_PAGE_SLUGS, "mg hs specs and features", "model/mg-hs-super-hybrid"),
        ],
    )
    def test_real_topics_that_resolve(self, page_slugs, topic, expected_slug):
        assert bot.resolve_topic_slug(topic, page_slugs) == expected_slug

    @pytest.mark.parametrize(
        "page_slugs,topic",
        [
            (bot.MG_PAGE_SLUGS, "mg company info and models available in pakistan"),
            (bot.HONDA_PAGE_SLUGS, "honda pakistan models and features"),
            (bot.MG_PAGE_SLUGS, "mg motors pakistan models and features"),
            (bot.HONDA_PAGE_SLUGS, "honda pakistan best selling models"),
            (bot.HONDA_PAGE_SLUGS, "compare models"),
            (bot.MG_PAGE_SLUGS, "mg vs honda"),
            (bot.HONDA_PAGE_SLUGS, "compare with other brands"),
        ],
    )
    def test_real_topics_that_still_miss_the_fuzzy_fallback(self, page_slugs, topic):
        assert bot.resolve_topic_slug(topic, page_slugs) is None


class TestBrowseToolSchemas:
    """Regression guard: the browse tools' `topic` argument must stay FREE
    TEXT, with no JSON Schema `enum` of valid topics.

    An enum was tried and reverted. Measured A/B against the live Groq API
    (llama-3.3-70b-versatile, 12 calls per arm, identical questions): the
    enum arm hard-400'd with "tool_use_failed" on 10/12 calls (83%) versus
    1/12 (8%) without it - the model emits malformed tool-call syntax
    trying to satisfy the constraint and Groq rejects the whole request,
    which also makes resolve_topic_slug's fuzzy fallback unreachable. See
    make_browse_page_tool's comment for the full rationale.
    """

    @pytest.mark.parametrize(
        "tool",
        [bot.browse_honda_page_tool, bot.browse_mg_page_tool, bot.honda_price_tool],
    )
    def test_no_enum_on_any_tool_argument(self, tool):
        for arg_name, spec in tool.properties.items():
            assert "enum" not in spec, (
                f"{tool.name}.{arg_name} has an enum - this caused an 83% "
                f"hard-400 rate against Groq. See make_browse_page_tool."
            )

    def test_browse_tools_still_declare_a_free_text_topic_argument(self):
        for tool in (bot.browse_honda_page_tool, bot.browse_mg_page_tool):
            assert tool.properties["topic"]["type"] == "string"
            assert tool.required == ["topic"]


class TestHondaPricePattern:
    SAMPLE_HTML = (
        '<div><h4>Honda Civic</h4><div class="model-price">From PKR 8,499,000</div></div>'
        '<div><h4>Honda City</h4><div class="model-price">From PKR 4,737,000</div></div>'
    )

    def test_extracts_every_model_and_price(self):
        matches = bot._HONDA_PRICE_PATTERN.findall(self.SAMPLE_HTML)
        assert matches == [("Civic", "8,499,000"), ("City", "4,737,000")]

    def test_no_matches_on_unrelated_html(self):
        assert bot._HONDA_PRICE_PATTERN.findall("<div>nothing relevant here</div>") == []


class TestHtmlTagPattern:
    def test_strips_simple_tags(self):
        assert bot._HTML_TAG_PATTERN.sub("", "<b>bold</b> text") == "bold text"

    def test_strips_tags_with_attributes(self):
        assert bot._HTML_TAG_PATTERN.sub("", '<a href="x">link</a>') == "link"

    def test_leaves_plain_text_untouched(self):
        text = "This is plain text with no markup."
        assert bot._HTML_TAG_PATTERN.sub("", text) == text


class TestIsRateLimitError:
    @pytest.mark.parametrize(
        "error_text",
        [
            "rate limit reached for model llama-3.3-70b-versatile",
            "error code: 429",
            "rate_limit_exceeded",
        ],
    )
    def test_detects_known_rate_limit_phrasings(self, error_text):
        assert bot.is_rate_limit_error(error_text) is True

    def test_does_not_flag_an_unrelated_error(self):
        assert bot.is_rate_limit_error("connection timed out") is False

    def test_known_false_positive_on_an_unrelated_429_substring(self):
        # Documented limitation (see is_rate_limit_error's docstring): a
        # bare "429" substring check can't distinguish a real HTTP 429
        # from an unrelated number that happens to contain those digits.
        # This pins that known gap as CURRENT behavior, not something this
        # test is meant to catch - a regression guard for whenever a more
        # precise check (e.g. inspecting a real status code) replaces this.
        assert bot.is_rate_limit_error("requested 4291 tokens") is True


def _make_error_frame(exception=None, error_text=""):
    # Duck-typed stand-in for a real pipecat ErrorFrame - error_frame_is_
    # rate_limit only ever reads .exception and .error, so a real
    # dataclass instance (needing the full pipecat frame machinery) isn't
    # necessary here.
    return type("FakeErrorFrame", (), {"exception": exception, "error": error_text})()


def _make_openai_error(error_cls, status_code):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return error_cls("test error", response=response, body=None)


class TestErrorFrameIsRateLimit:
    """error_frame_is_rate_limit is the function on_pipeline_error actually
    calls now - is_rate_limit_error above is only its text-only fallback.
    """

    def test_true_for_a_real_openai_rate_limit_error(self):
        exc = _make_openai_error(openai.RateLimitError, 429)
        frame = _make_error_frame(exception=exc, error_text="doesn't matter")
        assert bot.error_frame_is_rate_limit(frame) is True

    def test_false_for_a_different_typed_exception_even_if_text_mentions_429(self):
        # The whole point of preferring the typed check: a non-rate-limit
        # exception whose text happens to contain "429" must NOT be
        # miscounted as a rate limit just because the text heuristic alone
        # would have said yes.
        exc = _make_openai_error(openai.BadRequestError, 400)
        frame = _make_error_frame(exception=exc, error_text="something mentions 429 in passing")
        assert bot.error_frame_is_rate_limit(frame) is False

    def test_falls_back_to_text_heuristic_when_no_exception_is_present(self):
        frame = _make_error_frame(exception=None, error_text="rate limit reached")
        assert bot.error_frame_is_rate_limit(frame) is True

    def test_falls_back_to_text_heuristic_and_can_still_return_false(self):
        frame = _make_error_frame(exception=None, error_text="connection timed out")
        assert bot.error_frame_is_rate_limit(frame) is False


class TestIsBackgroundSummarizationError:
    def test_detects_the_generation_failure_prefix(self):
        text = "error generating context summary: error code: 429 - rate_limit_exceeded"
        assert bot.is_background_summarization_error(text) is True

    def test_detects_the_timeout_prefix(self):
        assert bot.is_background_summarization_error("context summarization timed out after 10s") is True

    def test_does_not_flag_a_normal_turn_error(self):
        text = "error during completion: error code: 429 - rate_limit_exceeded"
        assert bot.is_background_summarization_error(text) is False

    def test_does_not_flag_an_unrelated_error(self):
        assert bot.is_background_summarization_error("connection timed out") is False


class TestIsToolCallFailedError:
    def test_detects_the_real_groq_message(self):
        # Verbatim (lowercased) text captured live from a real Groq
        # response - not a synthetic example.
        text = "error during completion: failed to call a function. please adjust your prompt. see 'failed_generation' for more details."
        assert bot.is_tool_call_failed_error(text) is True

    def test_does_not_flag_a_rate_limit_error(self):
        assert bot.is_tool_call_failed_error("error code: 429 - rate_limit_exceeded") is False

    def test_does_not_flag_an_unrelated_error(self):
        assert bot.is_tool_call_failed_error("connection timed out") is False


class TestLlmModelFallbackChain:
    def test_chain_is_non_empty(self):
        assert len(bot.LLM_MODEL_FALLBACK_CHAIN) >= 1

    def test_all_entries_are_non_empty_strings(self):
        assert all(isinstance(m, str) and m for m in bot.LLM_MODEL_FALLBACK_CHAIN)


class TestCheckPortAvailable:
    def test_raises_systemexit_when_the_port_is_bound(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            port = s.getsockname()[1]
            with pytest.raises(SystemExit):
                bot._check_port_available("127.0.0.1", port)

    def test_does_not_raise_when_the_port_is_free(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        # socket closed here (released the port) before the actual check
        bot._check_port_available("127.0.0.1", port)  # should not raise

    def test_catches_an_ipv6_only_conflict_when_checking_localhost(self):
        # Regression test for the exact bug hit live: a stale process was
        # listening on ::1 only, and the OLD IPv4-only implementation of
        # this function passed clean while uvicorn still failed to bind
        # moments later. Skips gracefully if this environment has no IPv6
        # loopback at all, rather than failing for an unrelated reason.
        try:
            probe = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            probe.bind(("::1", 0))
        except OSError:
            pytest.skip("IPv6 loopback not available in this environment")
            return
        with probe:
            probe.listen(1)
            port = probe.getsockname()[1]
            with pytest.raises(SystemExit):
                bot._check_port_available("localhost", port)
