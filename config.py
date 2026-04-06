"""
Configuration constants for the AI Tools Extraction pipeline.
"""

# ---------------------------------------------------------------------------
# Archive URLs to scrape (last 17 days of content)
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
# Canonical source mapping
# ---------------------------------------------------------------------------
# Maps archive-URL domains and email sender addresses to a single canonical
# provider name so that the same newsletter scraped from the web AND received
# via Gmail counts as ONE distinct source, not two.
SOURCE_MAPPING: dict[str, str] = {
    # Archive URL domains (as returned by url.split("/")[2])
    "www.superhuman.ai":        "superhuman.ai",
    "www.therundown.ai":        "therundown.ai",
    "www.theneurondaily.com":   "theneurondaily.com",
    "www.mindstream.news":      "mindstream.news",
    "www.bensbites.com":        "bensbites.com",
    "importai.substack.com":    "importai.substack.com",
    # Email sender addresses
    "bensbites@substack.com":                          "bensbites.com",
    "importai@substack.com":                           "importai.substack.com",
    "news+canned.response@daily.therundown.ai":        "therundown.ai",
    "news@alphasignal.ai":                             "alphasignal.ai",
    "dan@tldrnewsletter.com":                          "tldrnewsletter.com",
    "hi@mail.theresanaiforthat.com":                   "theresanaiforthat.com",
    "theneuron@newsletter.theneurondaily.com":         "theneurondaily.com",
}

# ---------------------------------------------------------------------------
# LLM settings
# ---------------------------------------------------------------------------
LLM_MODEL = "gemini-2.5-flash"
LLM_MAX_TOKENS = 65536  # Gemini 2.5 Flash maximum output tokens

# ---------------------------------------------------------------------------
# Scraping / time-window settings
# ---------------------------------------------------------------------------
LOOKBACK_DAYS = 17
REQUEST_TIMEOUT = 30  # seconds per HTTP request
MAX_ARTICLES_PER_SOURCE = 200  # safety cap per archive
