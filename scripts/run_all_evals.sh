#!/usr/bin/env bash
# Runs eval scenarios in evals/ against a FRESH bot.py process each time
# (running multiple scenarios against one long-lived process has caused
# real problems before - e.g. Kokoro TTS silently stopped firing after
# several turns, and the eval harness's own event-matching gets confused
# by leftover state from a previous scenario). Restarting fresh per
# scenario trades a bit of wall-clock time for actually trustworthy
# results.
#
# Every scenario makes REAL Groq API calls (no mocking) - each completion
# call costs ~1000 prompt tokens just from the system prompt + the 3
# advertised tool schemas, before any conversation content. Groq's free
# tier has a 100,000-tokens-PER-DAY cap shared across everything on the
# same key (this whole project's manual testing, not just evals) - running
# every scenario on every dev-loop iteration burns through that fast.
#
# So by default, this only runs the QUICK tier: scenarios that make one
# cheap completion call each, no tool-calling round-trips (which double
# the cost per turn: one call to decide to call the tool, a second to
# turn the result into a reply) and no deliberately long "explain in
# detail" completions. The FULL tier - browse_honda_test,
# interrupted_context_test, thrashing_test, tool_calling_test - covers
# real tool-calling and long-response interruption specifically, and
# costs meaningfully more per run; run it deliberately, not on every loop.
#
# Usage:
#   scripts/run_all_evals.sh              # QUICK tier only (cheap, default)
#   scripts/run_all_evals.sh --full       # every scenario in evals/
#   scripts/run_all_evals.sh foo bar      # only evals/foo.yaml, evals/bar.yaml
#                                         # (explicit names always run
#                                         #  regardless of tier)
#
# Exit code is 0 only if every scenario that ran passed.
set -uo pipefail

cd "$(dirname "$0")/.."

PYTHON="/d/venvs/pipecat-voice/Scripts/python.exe"
PIPECAT="/d/venvs/pipecat-voice/Scripts/pipecat.exe"
PORT=7860
LOG_DIR="eval_logs"
mkdir -p "$LOG_DIR"

# Tool-calling scenarios pay for a second completion per turn (initial
# call -> tool call -> second call to turn the result into a reply), and
# the "in detail"/"in great detail" prompts are deliberately long so
# there's a real window to interrupt into - both cost noticeably more
# than the rest of the suite's short, single-completion turns.
FULL_TIER_ONLY="browse_honda_test interrupted_context_test thrashing_test tool_calling_test"

if [ "$#" -gt 0 ] && [ "$1" != "--full" ]; then
    SCENARIOS=()
    for name in "$@"; do
        SCENARIOS+=("evals/${name}.yaml")
    done
elif [ "${1:-}" = "--full" ]; then
    SCENARIOS=(evals/*.yaml)
else
    SCENARIOS=()
    for f in evals/*.yaml; do
        name=$(basename "$f" .yaml)
        if [[ " $FULL_TIER_ONLY " != *" $name "* ]]; then
            SCENARIOS+=("$f")
        fi
    done
    RAN_QUICK_TIER_DEFAULT=1
    echo "Running the QUICK tier (cheap, default) - use --full for every scenario."
    echo "Skipped (full tier only): $FULL_TIER_ONLY"
    echo ""
fi
RAN_QUICK_TIER_DEFAULT="${RAN_QUICK_TIER_DEFAULT:-0}"

declare -A RESULTS
declare -A DURATIONS

# Scenarios that need the bot process started with a non-default
# environment (name -> "VAR=value VAR2=value2 ..."). stt_failure_test
# specifically needs a broken GROQ_API_KEY to exercise the STT-down
# fallback path - starting it normally (real key) means STT succeeds and
# the test can never see the behavior it's checking for.
# SKIP_STARTUP_SELF_CHECK is also needed here: bot.py's own startup
# self-check (added for bulletproofing) would otherwise catch this same
# bad key and exit before the scenario ever gets a chance to run - this
# is the one deliberate, intentional case where that's the wrong thing.
declare -A SCENARIO_ENV=(
    [stt_failure_test]="GROQ_API_KEY=bad_key_for_testing SKIP_STARTUP_SELF_CHECK=1"
)

# Scenarios whose `eval:` semantic-judge checks depend on a local Ollama
# instance being up. When Ollama is down, these fail with an
# APIConnectionError that has nothing to do with the bot itself - flagged
# here so the summary doesn't misreport an infra gap as a bot regression.
OLLAMA_DEPENDENT="ambiguous_input_test"

# Brief pause between scenarios: running bot.py processes back-to-back at
# full speed measurably slows Kokoro's first TTS call in the next
# scenario (confirmed: a scenario that fails under rapid succession passes
# cleanly seconds later in isolation) - almost certainly transient
# resource contention (CPU/model-load) from the just-killed process not
# having fully released yet, not a real bug.
INTER_SCENARIO_COOLDOWN=3

kill_port() {
    # Find and kill whatever's actually listening on $PORT (there's no
    # reliable `kill $pid` for a Windows process launched via `&` in Git
    # Bash - taskkill on the real PID from netstat is what actually works).
    local pids
    pids=$(netstat -ano 2>/dev/null | grep ":$PORT" | grep LISTENING | awk '{print $NF}' | sort -u)
    for pid in $pids; do
        taskkill //F //PID "$pid" >/dev/null 2>&1 || true
    done
}

wait_for_port() {
    local tries=0
    while ! (netstat -ano 2>/dev/null | grep ":$PORT" | grep -q LISTENING); do
        sleep 1
        tries=$((tries + 1))
        if [ "$tries" -ge 30 ]; then
            return 1
        fi
    done
    return 0
}

wait_for_port_clear() {
    # A flat `sleep 1` after kill_port was not enough: taskkill can return
    # before Windows actually releases the socket, so the OLD process's
    # listener can still show up in netstat when the NEXT scenario's
    # wait_for_port checks "is something listening" - which only asks
    # whether ANY listener exists, not whether it's the process THIS
    # scenario just started. Confirmed directly: bot.py's own
    # _check_port_available correctly refused to start for a fresh
    # process (logged "already in use"), while wait_for_port below still
    # found the stale listener and let the eval proceed against it - so a
    # scenario could silently run against the PREVIOUS scenario's bot
    # process instead of a fresh one, defeating the "fresh process per
    # scenario" guarantee this whole script exists for.
    local tries=0
    while netstat -ano 2>/dev/null | grep ":$PORT" | grep -q LISTENING; do
        sleep 1
        tries=$((tries + 1))
        if [ "$tries" -ge 15 ]; then
            return 1
        fi
    done
    return 0
}

echo "Running ${#SCENARIOS[@]} scenario(s)..."
echo ""

for scenario in "${SCENARIOS[@]}"; do
    name=$(basename "$scenario" .yaml)

    if [ ! -f "$scenario" ]; then
        echo "=== $name === SKIPPED (file not found: $scenario)"
        RESULTS["$name"]="SKIP"
        continue
    fi

    echo "=== $name ==="
    needs_special_env="${SCENARIO_ENV[$name]:-}"
    if [ -n "$needs_special_env" ]; then
        echo "  (starting bot with special env: $needs_special_env)"
    fi
    kill_port
    if ! wait_for_port_clear; then
        echo "  FAIL - a previous bot process wouldn't release :$PORT"
        RESULTS["$name"]="FAIL"
        DURATIONS["$name"]="-"
        continue
    fi

    bot_log="$LOG_DIR/${name}.bot.log"
    (
        if [ -n "$needs_special_env" ]; then
            eval "export $needs_special_env"
        fi
        PYTHONUTF8=1 "$PYTHON" bot.py -t eval > "$bot_log" 2>&1
    ) &

    if ! wait_for_port; then
        echo "  FAIL - bot never started listening on :$PORT (see $bot_log)"
        RESULTS["$name"]="FAIL"
        DURATIONS["$name"]="-"
        kill_port
        continue
    fi

    start_ts=$(date +%s)
    if PYTHONUTF8=1 "$PIPECAT" eval run "$scenario" -v 2>&1 | tee "$LOG_DIR/${name}.eval.log"; then
        RESULTS["$name"]="PASS"
    else
        RESULTS["$name"]="FAIL"
    fi
    end_ts=$(date +%s)
    DURATIONS["$name"]="$((end_ts - start_ts))s"

    kill_port
    rm -f ./*.eval.log
    echo ""
    sleep "$INTER_SCENARIO_COOLDOWN"
done

echo "=================================================="
echo "SUMMARY"
echo "=================================================="
pass=0
fail=0
skip=0
for scenario in "${SCENARIOS[@]}"; do
    name=$(basename "$scenario" .yaml)
    result="${RESULTS[$name]:-FAIL}"
    duration="${DURATIONS[$name]:-}"
    note=""
    if [ "$result" != "PASS" ] && [[ " $OLLAMA_DEPENDENT " == *" $name "* ]]; then
        note="  (needs local Ollama running for its judge - check that before treating this as a bot bug)"
    fi
    printf "  %-32s %-6s %-6s%s\n" "$name" "$result" "$duration" "$note"
    case "$result" in
        PASS) pass=$((pass + 1)) ;;
        SKIP) skip=$((skip + 1)) ;;
        *) fail=$((fail + 1)) ;;
    esac
done
echo "--------------------------------------------------"
echo "$pass passed, $fail failed, $skip skipped"
if [ "$RAN_QUICK_TIER_DEFAULT" -eq 1 ]; then
    echo "(quick tier only - run with --full to also cover: $FULL_TIER_ONLY)"
fi
echo "Logs: $LOG_DIR/"

if [ "$fail" -gt 0 ]; then
    exit 1
fi
exit 0
