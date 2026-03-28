"""
Configuration constants for the AI Tools Extraction pipeline.
"""

# ---------------------------------------------------------------------------
# Archive URLs to scrape (last 14 days of content)
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
TAB_FIELD_TOOLS = "Field Tools"

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
LLM_MODEL = "gemini-1.5-flash"
LLM_MAX_TOKENS = 4096

# ---------------------------------------------------------------------------
# Scraping / time-window settings
# ---------------------------------------------------------------------------
LOOKBACK_DAYS = 14
REQUEST_TIMEOUT = 30  # seconds per HTTP request
MAX_ARTICLES_PER_SOURCE = 200  # safety cap per archive
