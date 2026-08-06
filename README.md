# Voice-In, Text-Out AI Chatbot

Pipeline built with [Pipecat](https://github.com/pipecat-ai/pipecat):

mic (WebRTC or local) -> Silero VAD -> Groq (Whisper STT, bilingual English/Urdu) -> Groq (Llama 3.3 70B) -> reply delivered as text (RTVI message or console) — **no TTS, no speaker output**.

This is a sibling variant of the voice-in/voice-out bot on the `fix-edge-cases` branch: same input pipeline (mic, VAD, bilingual STT), but replies are delivered as text instead of being spoken. It also includes a worked example of LLM tool-calling against a real website (`honda.com.pk`), not fake/hardcoded data.

## Setup

1. Create a `.env` file with your Groq API key:

   ```
   GROQ_API_KEY=your_key_here
   ```

   Get a free key at https://console.groq.com. No other API keys are needed.

   Optionally, add a second key as `GROQ_API_KEY_2` - if every model rate-limits on the first key's daily quota, the bot switches to this one automatically and back again after a cooldown (see the Notes section below). Skip this if one key is enough for your usage.

2. Install dependencies (pinned to the exact versions this project is verified against):

   ```
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

   This project was developed against a venv on `D:\venvs\pipecat-voice` (C: was nearly full) — swap in that path below for `.venv` if you're continuing on that same machine, or use your own venv location.

## Run

Two modes:

```
.venv\Scripts\python.exe bot.py             # WebRTC server; open http://localhost:7860
.venv\Scripts\python.exe bot.py --local     # Talk directly via the local mic
```

Speak into your microphone; the bot's reply is delivered as text — over WebRTC it arrives at the connected client as an RTVI `bot-llm-text` message, in `--local` mode it only shows up in the console (`CONVO`-level log line), since there's no client to display it to. Ctrl+C to stop.

Set `LOG_LEVEL=CONVO` to see just the conversation transcript (`User:`/`Bot:` lines) instead of full debug output — much cleaner for `--local` mode. Note this only applies to `--local`: the WebRTC path resets logging back to full debug internally right before starting the server (a `pipecat.runner.run.main()` behavior, not something this file controls).

## Web UI

`client/index.html` is a small standalone chat page (no build step, no npm install) that connects to the bot's WebRTC endpoint using Pipecat's official `@pipecat-ai/client-js` + `@pipecat-ai/small-webrtc-transport` packages, loaded straight from a CDN. With `python bot.py` running, open **`http://localhost:7860/`** in a browser (`bot.py` now serves this file itself, same-origin as `/api/offer`) and click "Start talking" - your speech shows up as a "You:" bubble, the bot's reply streams in as a "Bot:" bubble.

Open it this way, not by double-clicking `client/index.html` from disk - a `file://` origin's microphone-permission grant doesn't reliably persist in Chrome, so mute/unmute kept re-triggering the permission prompt. The file can still technically be opened directly (its `BOT_OFFER_URL` logic falls back to `http://localhost:7860/api/offer` for a `file://` origin), but `http://localhost:7860/` is the one that doesn't have this problem.

## Tool-calling demo

The bot can answer real questions about Honda Pakistan by fetching live data, not guessing:

- `check_honda_price` — real, current starting prices from honda.com.pk's homepage mega-menu (e.g. "How much does the Civic cost?")
- `browse_honda_page` — fetches and reads any of a known set of real pages on the site (model specs, dealer/contact info, promotions, company info, policies) for open-ended questions

Both tools fall back to a recently-cached result if a live fetch fails, and distinguish "the site itself is unreachable/broken" from "this specific model/page doesn't exist" — a naive scrape failure (e.g. a bot-challenge page from Cloudflare) used to silently report every model as "not found" with no signal anything was actually wrong.

## Testing

Two tiers, fast to slow:

**Unit tests** (`tests/`) — pure, network-free functions only (topic→slug resolution, the Honda price regex, rate-limit detection, etc.). No API key, no network, no live bot process needed; runs in a couple of seconds:

```
.venv\Scripts\python.exe -m pytest tests/ -v
```

**Evals** (`evals/`) — full pipeline, end-to-end, against a real running `bot.py` with synthesized speech via Kokoro (no human needed, no TTS output needed to run them since this pipeline doesn't use it). These make real Groq API calls, so there are two tiers:

```
scripts/run_all_evals.sh            # QUICK tier (default) - 6 cheap, single-completion scenarios
scripts/run_all_evals.sh --full     # every scenario, including tool-calling and long-response ones
scripts/run_all_evals.sh foo bar    # only evals/foo.yaml, evals/bar.yaml - runs regardless of tier
```

Groq's free tier caps at 100,000 tokens/day shared across everything on the key (manual testing included) — the quick tier exists so a normal dev-loop run doesn't burn through that. Run `--full` deliberately (e.g. before a release), not on every iteration.

Starts a fresh `bot.py` process per scenario, runs it, tears it down, and prints a pass/fail summary. Bot/eval logs land in `eval_logs/` (gitignored).

Some scenarios use a local Ollama judge (`eval:` semantic checks) — if Ollama isn't running, those fail with an `APIConnectionError` unrelated to the bot itself; the summary flags which scenarios that applies to.

## What's different from the voice variant (`fix-edge-cases` branch)

This branch ported everything from the voice variant's hardening work that **isn't** specific to the STT/audio-input layer: startup self-check, real circuit breaker with exponential backoff, context summarization for long conversations, interrupted-response context handling, missing-key/port-conflict fast-fails, and an ambiguous-input clarifying-question instruction.

Deliberately **not** ported — these are audio-input/STT-layer concerns, out of scope for this variant:
- The backchannel filter (dropping "Mm-hmm"-style filler before it reaches the LLM)

The following WERE ported here after real mistranscription/dropped-speech bugs were confirmed live in this variant's own eval logs (the practical consequence the original version of this section warned about): the Whisper-hallucination phrase filter, and the Urdu-misclassification English confidence bias (`_UR_CONFIDENCE_MARGIN`/`_SHORT_UTTERANCE_UR_MARGIN` in `BilingualGroqSTTService`). VAD (`confidence`/`min_volume`/`stop_secs`) has also been tuned directly against this variant's own measurements rather than reusing the voice variant's numbers verbatim - see the comment above `LLMUserAggregatorParams` in `run_bot`.

## Notes

- `--local` uses `LocalAudioTransport` (your PC's mic directly via PyAudio, no browser).
- STT (`BilingualGroqSTTService`) transcribes each utterance twice concurrently — once forced to English, once forced to Urdu — and picks "ur" only if it clears a real confidence margin over "en", not just a raw comparison (a plain "higher confidence wins" comparison was confirmed live to pick "ur" on English audio by margins as small as 0.002). Unlike the voice variant, there's no technical reason this bot couldn't reply in Urdu too (no TTS to crash) — the system prompt still forces English-only replies, kept as-is for behavioral consistency between the two variants.
- Urdu is transcribed in Roman (Latin) script, not native Nastaliq — Whisper has no API parameter for this (there's no `"ur-Latn"` language code), so it's done via prompt-conditioning (`_ROMAN_URDU_STT_PROMPT`: a Roman-Urdu sample fed as the `prompt` param, which Whisper tends to continue in). That's a heuristic, not a guarantee, so `_contains_arabic_script` checks the actual output and `_transliterate_nastaliq_to_roman` (a plain character/digraph lookup table, no dependency, no network call) converts any Nastaliq that slips through back to Roman script rather than discarding it — falling back to the English decode instead would silently swap real Urdu content for a likely-wrong English guess at the same audio. Short vowels are inherently unrecoverable this way (Nastaliq doesn't write them down at all), so the result is readable but not phonetically exact.
- STT model is `whisper-large-v3` (the full model, not the turbo variant) — turbo was tried first and is ~6-10x faster (3s → 0.3-0.5s), but a real `--local` session surfaced genuine speech coming back misrecognized as different words (not hallucinated on silence - actually spoken audio, transcribed wrong), matching Whisper's own documented turbo trade-off. Switch back to `"whisper-large-v3-turbo"` in `run_bot` if the extra latency (STT already pays for two concurrent calls per utterance - see `BilingualGroqSTTService`) matters more than the accuracy gain for your use case.
- `SanitizingGroqLLMService` strips any raw HTML-like tags from LLM output before it's pushed at all — defensive (no observed injection), since this variant's system prompt uniquely allows rich formatting and a client that renders markdown-to-HTML without sanitizing would be at risk from any stray tag the LLM echoes back. This has to happen at the LLM service's own `push_frame` (not a separate downstream processor) because pipecat's RTVI observer sends a frame's content to the client the first time it sees that `frame.id` - which happens when `llm` pushes it, before any later pipeline stage runs.
- An optional `GROQ_API_KEY_2` in `.env` extends the daily token quota: if every model in `LLM_MODEL_FALLBACK_CHAIN` rate-limits on the primary key, the bot switches to this second key automatically (see `GROQ_API_CHAIN` / `on_pipeline_error` in `bot.py`) and switches back after a cooldown with no further rate limits. Optional - the bot runs fine with just `GROQ_API_KEY`.
- End-of-turn detection uses `SpeechTimeoutUserTurnStopStrategy` (a plain post-VAD-stop timer), set explicitly in `LLMUserAggregatorParams`. Left unset, pipecat silently defaults to `TurnAnalyzerUserTurnStopStrategy` wrapping `LocalSmartTurnAnalyzerV3` - a local ONNX model that predicts end-of-turn from the audio's acoustic/prosodic content rather than just silence duration, running invisibly on every turn (measured directly: ~209ms of real inference latency per turn). It's also a second, independent way a turn can end early - it can judge a mid-sentence pause "sounds finished" regardless of how the VAD `stop_secs` above is tuned, a second contributor to the same fragmentation bug that motivated retuning `stop_secs` in the first place. Turned off here in favor of the plain timer already reasoned about above; re-enable by setting `user_turn_strategies` back to the pipecat default if prosody-aware turn detection is worth the latency for your use case.
- See `EDGE_CASES.md` for the original edge-case audit this whole project started from — note its header explains most of it is about the *voice* variant specifically.
