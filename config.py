"""
Configuration constants for the AI Tools Extraction pipeline.
"""

# ---------------------------------------------------------------------------
# Archive URLs to scrape (last LOOKBACK_DAYS of content)
# ---------------------------------------------------------------------------
ARCHIVE_URLS = [
    "https://www.superhuman.ai/archive",
    "https://www.therundown.ai/archive",
    "https://www.theneurondaily.com/archive",
    "https://www.mindstream.news/archive",
    "https://www.bensbites.com/archive",
    "https://importai.substack.com/archive",
]

# ---------------------------------------------------------------------------
# Gmail label to fetch newsletters from
# ---------------------------------------------------------------------------
GMAIL_LABEL = "AI-Newsletters"

# ---------------------------------------------------------------------------
# Google Sheets configuration
# ---------------------------------------------------------------------------
# Set this to your target Google Sheet ID (from the sheet URL).
# Example URL: https://docs.google.com/spreadsheets/d/<SHEET_ID>/edit
SPREADSHEET_ID = ""  # <-- Paste your Sheet ID here

TAB_AI_TOOLS_LOG = "AI Tools Log"
TAB_FIELD_TOOLS = "field tools"

# ---------------------------------------------------------------------------
# Tool categories
# ---------------------------------------------------------------------------
CATEGORIES = [
    "Researching and synthesizing information",
    "Drafting and refining written content",
    "Writing, debugging, and explaining code",
    "Summarizing documents and meeting transcripts",
    "Analyzing and visualizing complex data",
    "Generating and editing images, videos, and audio",
    "Translating languages and practicing conversation",
    "Brainstorming and creative ideation",
    "Automating multi-step tasks (Agentic workflows)",
    "Managing schedules and professional correspondence",
    "Slides preparation",
    "Learning and studying",
]

# ---------------------------------------------------------------------------
# LLM settings
# ---------------------------------------------------------------------------
LLM_MODEL = "gemini-2.5-flash"
LLM_MAX_TOKENS = 32768

# ---------------------------------------------------------------------------
# Gemini free-tier rate limits
# ---------------------------------------------------------------------------
# RPM=10, RPD=250, TPM=250 000, context=1 000 000 tokens
# We use conservative delays to stay well within these limits.
#
# Token budget per call:
#   600k chars / 4 chars-per-token = 150k input tokens
#   + 32k output tokens = 182k total < 250k TPM at 1 call/min  ✓
CHUNK_MAX_CHARS = 600_000        # ~150k tokens per chunk
                                 # (150k input + 32k output = 182k < 250k TPM at 1/min ✓)
CONTENT_CAP_CHARS = 50_000      # per-article/email hard cap
                                 # AI newsletters are 1k-5k words = 4k-25k chars;
                                 # 50k only triggers on nav/boilerplate-heavy fallbacks
LLM_DELAY_HEAVY = 60            # seconds between all LLM calls (ensures TPM budget resets)
MAX_MENTIONS_CHARS = 800_000    # cap combined mentions text for ranking calls

# ---------------------------------------------------------------------------
# Scraping / time-window settings
# ---------------------------------------------------------------------------
LOOKBACK_DAYS = 21
REQUEST_TIMEOUT = 30  # seconds per HTTP request
MAX_ARTICLES_PER_SOURCE = 200  # safety cap per archive
