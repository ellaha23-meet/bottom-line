"""
Configuration constants for the AI Tools Extraction pipeline.
"""

# ---------------------------------------------------------------------------
# Archive URLs to scrape (newsletter website sources)
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
# Known email newsletter providers (used as source_id for email sources)
# Map from sender address → human-readable source ID
# ---------------------------------------------------------------------------
EMAIL_PROVIDERS = {
    "bensbites@substack.com": "bensbites_email",
    "importai@substack.com": "importai_email",
    "news+canned.response@daily.therundown.ai": "therundown_email",
    "news@alphasignal.ai": "alphasignal_email",
    "dan@tldrnewsletter.com": "tldr_email",
    "hi@mail.theresanaiforthat.com": "theresanaiforthat_email",
    "theneuron@newsletter.theneurondaily.com": "theneuron_email",
    "superhuman@mail.joinsuperhuman.ai": "superhuman_email",
    "hello@mindstream.news": "mindstream_email",
}

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
# Scraping / time-window settings
# ---------------------------------------------------------------------------
LOOKBACK_DAYS = 21
REQUEST_TIMEOUT = 30  # seconds per HTTP request
MAX_ARTICLES_PER_SOURCE = 200  # safety cap per archive
