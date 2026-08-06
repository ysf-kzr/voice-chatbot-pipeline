"""Unit tests for bot.py's pure, network-free functions.

Deliberately scoped to logic that doesn't need pipecat's runtime, a live
Groq connection, or real network access - the eval suite in evals/ already
covers full pipeline behavior end-to-end (Kokoro-synthesized speech through
the real bot). These run in well under a second and exist for the pieces
that should never need several seconds and a live API key to verify -
resolve_topic_slug's real-data cases below are real topic strings this
pipeline's LLM emitted in a live session, pinning the fuzzy fallback's
current hit/miss behavior against them (see TestResolveTopicSlugRealData
and TestBrowseToolSchemas for why an `enum` constraint was tried here and
reverted, not adopted).

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

    @staticmethod
    def _segment(avg_logprob, start, end):
        return type("Segment", (), {"avg_logprob": avg_logprob, "start": start, "end": end})()

    def test_weights_by_segment_duration_when_available(self):
        # A 0.1s segment at -0.05 and a 1.9s segment at -0.5: a flat mean
        # would be -0.275, but the long segment should dominate since it
        # covers almost all of the audio.
        segments = [self._segment(-0.05, 0.0, 0.1), self._segment(-0.5, 0.1, 2.0)]
        result = type("Result", (), {"segments": segments})()
        expected = (-0.05 * 0.1 + -0.5 * 1.9) / 2.0
        assert bot._avg_logprob(result) == pytest.approx(expected)

    def test_a_long_segment_dominates_a_short_outlier_within_one_transcript(self):
        # What duration weighting actually changes vs. a flat mean: within
        # a SINGLE transcript, a long real segment now dominates the
        # overall score, instead of a short outlier segment (e.g. a brief
        # burst of noise scored differently than the surrounding real
        # speech) pulling the average further than its share of the audio
        # warrants.
        segments = [self._segment(-0.15, 0.0, 2.9), self._segment(-0.9, 2.9, 3.0)]
        result = type("Result", (), {"segments": segments})()
        weighted = bot._avg_logprob(result)
        flat_mean = (-0.15 + -0.9) / 2
        assert weighted > flat_mean
        assert weighted == pytest.approx((-0.15 * 2.9 + -0.9 * 0.1) / 3.0)

    def test_falls_back_to_flat_mean_when_segments_carry_no_duration(self):
        # Hand-built segments with no start/end (as in
        # test_computes_mean_of_segment_logprobs above) shouldn't divide by
        # zero - they fall back to the original flat mean.
        segments = [self._segment(-0.1, 0.0, 0.0), self._segment(-0.3, 0.0, 0.0)]
        result = type("Result", (), {"segments": segments})()
        assert bot._avg_logprob(result) == pytest.approx(-0.2)


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

    def test_empty_topic_returns_none(self):
        # Regression test: an empty string is a substring of every dict
        # key, so the old "topic in key" fuzzy fallback matched the first
        # key unconditionally - resolve_topic_slug("", HONDA_PAGE_SLUGS)
        # used to resolve to "civic-standard".
        assert bot.resolve_topic_slug("", self.SLUGS) is None

    def test_whitespace_only_topic_returns_none(self):
        assert bot.resolve_topic_slug("   ", self.SLUGS) is None

    def test_single_char_topic_returns_none(self):
        assert bot.resolve_topic_slug("a", self.SLUGS) is None

    def test_topic_just_at_the_minimum_length_can_still_fuzzy_match(self):
        # 3 chars is the minimum length allowed through to the fuzzy
        # fallback - confirms the guard doesn't block a legitimate short
        # topic that's a real substring of a key ("cit" in "city").
        slugs = {"city": "city1-2l"}
        assert bot.resolve_topic_slug("cit", slugs) == "city1-2l"


class TestResolveTopicSlugRealData:
    """Regression tests against the actual production slug dicts, using
    real topic strings this pipeline's LLM emitted in a live session
    (captured from server logs) - not synthetic examples.

    The "still miss" cases below are documented gaps in the fuzzy fallback
    alone, not bugs to fix here - an `enum` constraint on the tool schema
    was tried for exactly these cases and reverted (see
    make_browse_page_tool's comment and TestBrowseToolSchemas below: it
    caused an 83% hard-400 rate against Groq). Free-text + this fuzzy
    fallback is the primary safeguard now, not defense-in-depth for an
    enum - these misses are pinned as known behavior, not xfail.
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


def _make_wav_bytes(duration_secs, sample_rate=16000):
    import wave
    from io import BytesIO

    num_frames = int(duration_secs * sample_rate)
    buf = BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * num_frames)
    return buf.getvalue()


class TestWavDurationSecs:
    def test_computes_duration_from_frame_count_and_rate(self):
        audio = _make_wav_bytes(1.5, sample_rate=16000)
        assert bot._wav_duration_secs(audio) == pytest.approx(1.5, abs=0.01)

    def test_zero_length_audio_is_zero_duration(self):
        audio = _make_wav_bytes(0.0)
        assert bot._wav_duration_secs(audio) == 0.0

    def test_malformed_audio_returns_zero_rather_than_raising(self):
        assert bot._wav_duration_secs(b"not a real wav file") == 0.0


class TestPickBilingualResult:
    def test_en_wins_a_raw_tie(self):
        # Regression case for the original bug: a bare `conf_en >= conf_ur`
        # comparison let "ur" win coin-flip ties on plainly English audio.
        assert bot._pick_bilingual_result(-0.2, -0.2, duration_secs=5.0) == "en"

    def test_en_wins_when_ur_is_ahead_but_below_the_margin(self):
        margin = bot._UR_CONFIDENCE_MARGIN
        assert bot._pick_bilingual_result(-0.383, -0.383 + margin - 0.01, duration_secs=5.0) == "en"

    def test_ur_wins_once_it_clears_the_margin(self):
        margin = bot._UR_CONFIDENCE_MARGIN
        assert bot._pick_bilingual_result(-0.383, -0.383 + margin + 0.01, duration_secs=5.0) == "ur"

    def test_short_utterance_requires_the_larger_margin(self):
        short_margin = bot._SHORT_UTTERANCE_UR_MARGIN
        normal_margin = bot._UR_CONFIDENCE_MARGIN
        conf_en = -0.3
        conf_ur = conf_en + normal_margin + 0.01  # clears the normal margin...
        assert conf_ur - conf_en < short_margin  # ...but not the short-utterance one
        assert (
            bot._pick_bilingual_result(conf_en, conf_ur, duration_secs=0.5) == "en"
        )
        assert (
            bot._pick_bilingual_result(conf_en, conf_ur, duration_secs=5.0) == "ur"
        )


class TestIsPromptEcho:
    def test_full_prompt_text_is_an_echo(self):
        assert bot._is_prompt_echo("Mujhe iski qeemat maloom karni hai.") is True

    def test_short_common_word_is_not_an_echo(self):
        # Regression case: the original substring-containment check flagged
        # any short word that happens to be a substring of the prompt text
        # ("hai" is literally inside "...karni hai.") as an echo, silently
        # dropping a legitimate one-word reply.
        assert bot._is_prompt_echo("hai") is False
        assert bot._is_prompt_echo("aap") is False

    def test_unrelated_text_is_not_an_echo(self):
        assert bot._is_prompt_echo("Compare the Honda Civic to the MG HS") is False


class TestLooksLikeKnownHallucination:
    def test_detects_a_known_stock_phrase(self):
        assert bot._looks_like_known_hallucination("Thanks for watching!") is True

    def test_detects_a_known_phrase_with_surrounding_words(self):
        assert bot._looks_like_known_hallucination("Okay, please subscribe to my channel!") is True

    def test_real_speech_is_not_flagged(self):
        assert bot._looks_like_known_hallucination("What does the Honda Civic cost?") is False


class TestRejectTranscriptReason:
    def test_accepts_a_normal_confident_transcript(self):
        assert bot._reject_transcript_reason("What is the water cycle?", -0.2) is None

    def test_empty_text_is_never_rejected(self):
        assert bot._reject_transcript_reason("", -5.0) is None
        assert bot._reject_transcript_reason("   ", -5.0) is None

    def test_rejects_below_the_confidence_floor(self):
        floor = bot._MIN_ACCEPTABLE_CONFIDENCE
        reason = bot._reject_transcript_reason("some text", floor - 0.01)
        assert reason is not None
        assert "confidence" in reason

    def test_rejects_a_prompt_echo_even_at_high_confidence(self):
        reason = bot._reject_transcript_reason("Mujhe iski qeemat maloom karni hai.", -0.05)
        assert reason is not None
        assert "echo" in reason

    def test_rejects_a_known_hallucination_phrase_even_at_high_confidence(self):
        reason = bot._reject_transcript_reason("Thanks for watching!", -0.05)
        assert reason is not None
        assert "hallucination" in reason

    def test_accepts_clean_transliterated_roman_text(self):
        # What _transcribe actually passes in here after transliteration -
        # by the time this runs, real Nastaliq text has already been
        # converted, so this should just be a normal accepted transcript.
        assert bot._reject_transcript_reason("mjhe iski qimt malom krni hai", -0.1) is None


class TestContainsArabicScript:
    def test_detects_nastaliq_text(self):
        assert bot._contains_arabic_script("مجھے اس کی قیمت معلوم کرنی ہے") is True

    def test_does_not_flag_roman_urdu(self):
        assert bot._contains_arabic_script("Mujhe iski qeemat maloom karni hai") is False

    def test_does_not_flag_plain_english(self):
        assert bot._contains_arabic_script("What is the price of the Civic?") is False

    def test_does_not_flag_empty_string(self):
        assert bot._contains_arabic_script("") is False

    def test_detects_a_single_stray_nastaliq_character_in_otherwise_roman_text(self):
        # Confirms the check isn't an all-or-nothing script classifier -
        # one leftover native character (e.g. from an incomplete
        # transliteration) is enough to flag.
        assert bot._contains_arabic_script("Mujhe ے chahiye") is True


class TestTransliterateNastaliqToRoman:
    def test_transliterates_a_real_urdu_sentence_to_readable_roman_script(self):
        result = bot._transliterate_nastaliq_to_roman("مجھے اس کی قیمت معلوم کرنی ہے")
        # Not asserting exact output (short vowels are lost by design, see
        # the function's docstring) - asserting the output is now plain
        # ASCII/Latin, i.e. actually in Roman script.
        assert result.isascii()
        assert not bot._contains_arabic_script(result)
        assert len(result) > 0

    def test_aspirated_digraph_depends_on_the_preceding_consonant(self):
        # بھ (be + do-chashmi he) -> "bh", not "b" + "h" separately handled
        # wrong, and not confused with, say, کھ -> "kh".
        assert bot._transliterate_nastaliq_to_roman("بھ") == "bh"
        assert bot._transliterate_nastaliq_to_roman("کھ") == "kh"
        assert bot._transliterate_nastaliq_to_roman("تھ") == "th"

    def test_leaves_latin_text_and_spaces_untouched(self):
        assert bot._transliterate_nastaliq_to_roman("hello world 123") == "hello world 123"

    def test_leaves_an_unmapped_character_as_is_rather_than_dropping_it(self):
        # Anything the table has no entry for survives unchanged (not
        # deleted) - this is what lets _contains_arabic_script's backstop
        # in _reject_transcript_reason still catch a transliteration this
        # table couldn't fully clean up, instead of silently losing
        # content and looking like it succeeded.
        assert "€" in bot._transliterate_nastaliq_to_roman("€100")

    def test_diacritics_on_a_consonant_produce_the_vowel_they_represent(self):
        # When present (fully-vocalized text only - rare in practice),
        # fatha/kasra/damma carry real, recoverable vowel information and
        # should become that vowel, not be discarded. ک = "k".
        assert bot._transliterate_nastaliq_to_roman("کَ") == "ka"
        assert bot._transliterate_nastaliq_to_roman("کِ") == "ki"
        assert bot._transliterate_nastaliq_to_roman("کُ") == "ku"

    def test_sukun_marks_absence_of_vowel_not_a_dropped_one(self):
        # سْ = seen + sukun ("no vowel here") - correctly empty, but for a
        # different reason than "this diacritic has no mapping": sukun
        # explicitly says there ISN'T a vowel, so "" is the right answer,
        # not a gap.
        assert bot._transliterate_nastaliq_to_roman("سْ") == "s"

    def test_shadda_doubles_the_preceding_consonant(self):
        # کّ = kaf + shadda (gemination mark) -> "kk", not "k" alone and
        # not a standalone shadda sound - shadda has no independent Roman
        # letter, it modifies what came before it.
        assert bot._transliterate_nastaliq_to_roman("کّ") == "kk"

    def test_tanwin_variants_add_the_n_ending_they_represent(self):
        assert bot._transliterate_nastaliq_to_roman("کً") == "kan"
        assert bot._transliterate_nastaliq_to_roman("کٍ") == "kin"
        assert bot._transliterate_nastaliq_to_roman("کٌ") == "kun"

    def test_no_arabic_script_survives_a_fully_vocalized_word(self):
        result = bot._transliterate_nastaliq_to_roman("کَتّاب")
        assert not bot._contains_arabic_script(result)


class TestMaybeTransliterate:
    def test_transliterates_nastaliq_text(self):
        result = bot._maybe_transliterate("مجھے", "ur")
        assert result.isascii()

    def test_leaves_roman_text_unchanged(self):
        text = "Mujhe iski qeemat maloom karni hai"
        assert bot._maybe_transliterate(text, "ur") == text

    def test_leaves_english_text_unchanged(self):
        text = "What is the price of the Civic?"
        assert bot._maybe_transliterate(text, "en") == text


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


def _make_openai_error(error_cls, status_code, code=None):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    body = {"message": "test error", "type": "tokens", "code": code} if code else None
    return error_cls("test error", response=response, body=body)


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

    def test_true_for_a_413_carrying_groqs_rate_limit_exceeded_code(self):
        # Regression test for a real live case: Groq returns HTTP 413 (not
        # 429) when a single request's token count already exceeds a
        # model's entire per-minute budget - openai-python only maps 429
        # to RateLimitError, so this used to fall into the generic
        # APIStatusError bucket and get trusted as "definitely not a rate
        # limit" (see test_false_for_a_different_typed_exception... above),
        # skipping model/key fallback entirely for exactly the case
        # fallback exists for. Confirmed live: Groq's own error body still
        # carries code="rate_limit_exceeded" on this 413.
        exc = _make_openai_error(openai.APIStatusError, 413, code="rate_limit_exceeded")
        frame = _make_error_frame(exception=exc, error_text="doesn't matter")
        assert bot.error_frame_is_rate_limit(frame) is True

    def test_false_for_a_413_without_the_rate_limit_code(self):
        # A 413 for some other reason (no rate-limit code in the body)
        # must not be swept in just because the status code matches the
        # regression case above.
        exc = _make_openai_error(openai.APIStatusError, 413, code="something_else")
        frame = _make_error_frame(exception=exc, error_text="doesn't matter")
        assert bot.error_frame_is_rate_limit(frame) is False

    def test_false_for_a_400_without_a_rate_limit_code(self):
        # test_false_for_a_different_typed_exception_even_if_text_mentions_429
        # above already covers this with body=None (code=None); this
        # covers the same "not a rate limit" outcome now that BadRequestError
        # is also an APIError and goes through the new .code check.
        exc = _make_openai_error(openai.BadRequestError, 400, code="invalid_request_error")
        frame = _make_error_frame(exception=exc, error_text="doesn't matter")
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


class TestGroqApiChain:
    def test_contains_no_falsy_entries(self):
        # Regression test: GROQ_API_KEY_2 used to be appended unconditionally
        # via os.environ.get(), so this list always had length 2 (with a
        # None second entry) for anyone who hadn't set a real second key -
        # which made on_pipeline_error's key-fallback branch reachable and
        # crashed trying to build a client with api_key=None.
        assert all(bot.GROQ_API_CHAIN)

    def test_length_matches_number_of_real_keys_actually_set(self):
        # Environment-dependent (a real GROQ_API_KEY_2 may or may not be
        # set, locally or in CI) - the regression guard is that the chain's
        # length always tracks the real environment, never a hardcoded 2.
        import os

        expected = len([k for k in (os.environ["GROQ_API_KEY"], os.environ.get("GROQ_API_KEY_2")) if k])
        assert len(bot.GROQ_API_CHAIN) == expected


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
