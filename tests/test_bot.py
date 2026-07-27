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
    """Confirms the `enum` constraint actually landed on the tool schema
    the LLM sees, and stays in sync with the slug dicts it's derived from -
    not just that resolve_topic_slug works in isolation.
    """

    def test_honda_tool_topic_enum_matches_slug_keys(self):
        enum_values = set(bot.browse_honda_page_tool.properties["topic"]["enum"])
        assert enum_values == set(bot.HONDA_PAGE_SLUGS.keys())

    def test_mg_tool_topic_enum_matches_slug_keys(self):
        enum_values = set(bot.browse_mg_page_tool.properties["topic"]["enum"])
        assert enum_values == set(bot.MG_PAGE_SLUGS.keys())

    def test_honda_price_tool_model_argument_has_no_enum(self):
        # check_honda_price's "model" argument isn't backed by a static
        # dict (prices come from a live scrape - see _get_honda_prices),
        # so it deliberately has no enum. Documents that as intentional,
        # not an inconsistency with the two tools above.
        assert "enum" not in bot.honda_price_tool.properties["model"]


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


class TestLlmModelFallbackChain:
    def test_chain_is_non_empty(self):
        assert len(bot.LLM_MODEL_FALLBACK_CHAIN) >= 1

    def test_all_entries_are_non_empty_strings(self):
        assert all(isinstance(m, str) and m for m in bot.LLM_MODEL_FALLBACK_CHAIN)
