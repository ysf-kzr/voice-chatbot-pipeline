import os
import sys
from pathlib import Path

# The repo root isn't on sys.path by default when pytest is invoked from an
# arbitrary working directory - added explicitly so `import bot` in the
# test modules resolves regardless of where `pytest` is actually run from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# bot.py's own module-level code calls sys.exit() if GROQ_API_KEY isn't set
# (a deliberate fail-fast for real runs - see bot.py's top-level check).
# Harmless to skip for these tests (nothing here makes a real API call),
# but importing bot.py at all would otherwise kill the whole test session
# before a single test runs. setdefault() only kicks in if nothing's set
# yet - a real .env's key (loaded by bot.py's own load_dotenv(override=True))
# still wins if one happens to be present.
os.environ.setdefault("GROQ_API_KEY", "test-key-not-used-for-real-requests")
