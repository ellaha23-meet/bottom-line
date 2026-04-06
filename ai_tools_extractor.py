#!/usr/bin/env python3
"""
AI Tools Extraction Pipeline
=============================
Scrapes AI newsletter archives and Gmail, uses an LLM to extract / rank /
categorise AI tools, and writes the results to Google Sheets.

Usage:
    1. Place your Google OAuth credentials.json in the project root.
    2. Set SPREADSHEET_ID in config.py (or via env var SPREADSHEET_ID).
    3. Export your Gemini key(s):
         export GEMINI_API_KEY="primary-key"                  # required
         export GEMINI_API_KEY_EXTRACT="phase-1-key"           # optional
         export GEMINI_API_KEY_FIELDS="phase-3-key"            # optional
       Phase-specific keys fall back to GEMINI_API_KEY if unset.
    4. Run:  python ai_tools_extractor.py
"""

import base64
import json
import logging
import os
import re
from urllib.parse import urlparse, quote_plus
import textwrap
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
load_dotenv()

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

import config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Google OAuth scopes
# ---------------------------------------------------------------------------
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/spreadsheets",
]

# ---------------------------------------------------------------------------
# Gemini API keys
# ---------------------------------------------------------------------------
# Separate keys per LLM phase help stay under per-key rate limits.
# Each phase-specific key falls back to GEMINI_API_KEY (the primary) if unset.
_gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
_gemini_api_key_extract = os.environ.get("GEMINI_API_KEY_EXTRACT", "") or _gemini_api_key
_gemini_api_key_fields = os.environ.get("GEMINI_API_KEY_FIELDS", "") or _gemini_api_key


def _configure_gemini(api_key: str) -> None:
    """(Re)configure the Gemini SDK to use the given API key."""
    if not api_key:
        raise RuntimeError("No Gemini API key available for this phase.")
    genai.configure(api_key=api_key)


# ===================================================================
# 1. Google Authentication
# ===================================================================
def get_google_credentials() -> Credentials:
    """Authenticate with Google using OAuth2 and return credentials.

    Looks for a cached token in token.json; if absent or expired,
    opens the browser-based OAuth consent flow using credentials.json.
    """
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists("credentials.json"):
                raise FileNotFoundError(
                    "credentials.json not found. Download it from Google Cloud Console."
                )
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as tok:
            tok.write(creds.to_json())

    return creds


# ===================================================================
# 2. Gmail Fetching
# ===================================================================
def fetch_emails(creds: Credentials) -> list[dict]:
    """Fetch emails from Gmail with the configured newsletter label.

    Returns a list of dicts: {"subject": ..., "date": ..., "from": ..., "body": ...}.
    Only emails from the last config.LOOKBACK_DAYS days are returned.
    """
    service = build("gmail", "v1", credentials=creds)
    cutoff = datetime.now(timezone.utc) - timedelta(days=config.LOOKBACK_DAYS)

    # Find the label ID for the configured label name
    labels_resp = service.users().labels().list(userId="me").execute()
    label_id = None
    for lbl in labels_resp.get("labels", []):
        if lbl["name"] == config.GMAIL_LABEL:
            label_id = lbl["id"]
            break

    if label_id is None:
        log.warning("Gmail label '%s' not found - skipping email source.", config.GMAIL_LABEL)
        return []

    # Fetch message IDs (paginate to get all)
    messages: list[dict] = []
    page_token = None
    query = f"after:{cutoff.strftime('%Y/%m/%d')}"

    while True:
        resp = (
            service.users()
            .messages()
            .list(userId="me", labelIds=[label_id], q=query, pageToken=page_token)
            .execute()
        )
        messages.extend(resp.get("messages", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    log.info("Found %d emails with label '%s' in the last %d days.",
             len(messages), config.GMAIL_LABEL, config.LOOKBACK_DAYS)

    results = []
    for msg_meta in messages:
        msg = (
            service.users()
            .messages()
            .get(userId="me", id=msg_meta["id"], format="full")
            .execute()
        )
        headers = {h["name"]: h["value"] for h in msg["payload"].get("headers", [])}
        body_text = _extract_email_body(msg["payload"])
        results.append({
            "subject": headers.get("Subject", ""),
            "date": headers.get("Date", ""),
            "from": headers.get("From", ""),
            "body": body_text,
        })

    return results


def _html_to_text_with_links(html: str) -> str:
    """Convert HTML to plain text while preserving hyperlinks inline.

    Converts ``<a href="https://example.com">Example</a>`` into
    ``Example (https://example.com)`` so that the LLM can see and extract
    tool homepage URLs that are embedded as hyperlinks in newsletter HTML.
    """
    soup = BeautifulSoup(html, "html.parser")
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        link_text = a_tag.get_text(strip=True)
        if href and href.startswith("http"):
            # Only inline the URL if it differs from the visible text
            if link_text and href not in link_text:
                a_tag.replace_with(f"{link_text} ({href})")
            elif not link_text:
                a_tag.replace_with(href)
            # else: link text already contains the URL — leave as-is
    return soup.get_text(separator="\n")


def _extract_email_body(payload: dict) -> str:
    """Recursively extract plain-text (or decoded HTML) from a Gmail payload."""
    parts = payload.get("parts", [])
    if not parts:
        data = payload.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
        return ""

    # Prefer text/html (to preserve hyperlinks), fall back to text/plain
    for mime in ("text/html", "text/plain"):
        for part in parts:
            if part.get("mimeType") == mime:
                data = part.get("body", {}).get("data", "")
                if data:
                    decoded = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
                    if mime == "text/html":
                        return _html_to_text_with_links(decoded)
                    return decoded
            # Only recurse into multipart containers, not leaf parts
            if part.get("mimeType", "").startswith("multipart/"):
                nested = _extract_email_body(part)
                if nested:
                    return nested
    return ""


# ===================================================================
# 3. Web Archive Scraping
# ===================================================================
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _fetch_page(url: str, page) -> str:
    """Navigate to a URL with Playwright and return the fully-rendered HTML."""
    page.goto(url, wait_until="domcontentloaded", timeout=15_000)
    return page.content()


def scrape_archives() -> list[dict]:
    """Scrape all configured archive URLs for posts from the last 17 days.

    Returns a list of dicts: {"source": ..., "title": ..., "date": ...,
    "url": ..., "content": ...}.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=config.LOOKBACK_DAYS)
    all_articles: list[dict] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800},
        )
        page = context.new_page()
        # Block images/fonts/media to speed up scraping
        page.route(
            "**/*",
            lambda route: route.abort()
            if route.request.resource_type in ("image", "media", "font")
            else route.continue_(),
        )

        for archive_url in config.ARCHIVE_URLS:
            log.info("Scraping archive: %s", archive_url)
            try:
                articles = _scrape_single_archive(archive_url, cutoff, page)
                log.info("  -> collected %d articles", len(articles))
                all_articles.extend(articles)
            except Exception:
                log.exception("Failed to scrape %s - skipping.", archive_url)

        context.close()
        browser.close()

    return all_articles


def _scrape_single_archive(archive_url: str, cutoff: datetime, page) -> list[dict]:
    """Parse an archive page and fetch individual article content."""
    # Navigate to the archive page
    try:
        page.goto(archive_url, wait_until="networkidle", timeout=config.REQUEST_TIMEOUT * 1000)
    except PlaywrightTimeoutError:
        page.goto(archive_url, wait_until="domcontentloaded", timeout=config.REQUEST_TIMEOUT * 1000)

    # Scroll down repeatedly to trigger infinite scroll and load all articles
    prev_height = 0
    for _ in range(30):  # up to 30 scrolls
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(2000)  # wait 2s for new content to load
        new_height = page.evaluate("document.body.scrollHeight")
        if new_height == prev_height:
            break  # no more content loading
        prev_height = new_height

    html = page.content()
    soup = BeautifulSoup(html, "html.parser")
    articles: list[dict] = []

    link_candidates = _find_article_links(soup, archive_url)
    log.info("  -> found %d link candidates on archive page", len(link_candidates))
    for title, url, date_str in link_candidates[:5]:
        log.info("     sample: [%s] %s (%s)", date_str, title[:60], url[:80])

    for title, url, date_str in link_candidates[: config.MAX_ARTICLES_PER_SOURCE]:
        pub_date = _parse_date_safe(date_str)
        if pub_date and pub_date < cutoff:
            continue  # older than lookback window - skip

        try:
            page_html = _fetch_page(url, page)
            page_soup = BeautifulSoup(page_html, "html.parser")
            # Try <article>, then main, then body
            article_tag = (
                page_soup.find("article")
                or page_soup.find("main")
                or page_soup.find("body")
            )
            content = _html_to_text_with_links(str(article_tag)) if article_tag else ""
        except Exception:
            log.warning("Could not fetch article: %s", url)
            content = title  # fall back to just the title

        articles.append({
            "source": archive_url,
            "title": title,
            "date": date_str,
            "url": url,
            "content": content[:50000],  # generous cap; chunking logic handles total size
        })

    return articles


def _find_article_links(soup: BeautifulSoup, archive_url: str) -> list[tuple[str, str, str]]:
    """Heuristically extract (title, url, date_string) tuples from an archive page."""
    base = "/".join(archive_url.split("/")[:3])  # scheme + host
    results: list[tuple[str, str, str]] = []

    # Strategy 1: Substack-style archives (<a class="post-preview-title"> or similar)
    for a_tag in soup.select("a[data-post-id], a.post-preview-title, a.post-preview"):
        href = a_tag.get("href", "")
        if not href.startswith("http"):
            href = base + href
        title = a_tag.get_text(strip=True)
        date_str = _find_nearby_date(a_tag)
        if title:
            results.append((title, href, date_str))

    # Strategy 2: Generic - any <a> whose href contains /p/ or /post/ or /newsletter/
    if not results:
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if any(seg in href for seg in ["/p/", "/post/", "/newsletter/", "/i/"]):
                if not href.startswith("http"):
                    href = base + href
                title = a_tag.get_text(strip=True) or href.split("/")[-1]
                date_str = _find_nearby_date(a_tag)
                if title and href not in [r[1] for r in results]:
                    results.append((title, href, date_str))

    # Strategy 3: Broad fallback - grab all links that look like articles
    if not results:
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if not href.startswith("http"):
                href = base + href
            if any(skip in href for skip in ["#", "javascript:", "/archive", "/login", "/subscribe"]):
                continue
            title = a_tag.get_text(strip=True)
            if title and len(title) > 15:
                date_str = _find_nearby_date(a_tag)
                results.append((title, href, date_str))

    return results


def _find_nearby_date(tag) -> str:
    """Search parent/sibling elements for a <time> tag or date-like text."""
    parent = tag
    for _ in range(4):
        if parent is None:
            break
        time_tag = parent.find("time")
        if time_tag:
            return time_tag.get("datetime", time_tag.get_text(strip=True))
        parent = parent.parent
    return ""


def _parse_date_safe(date_str: str) -> datetime | None:
    """Parse a date string; return None on failure."""
    if not date_str:
        return None
    try:
        dt = dateparser.parse(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


# ===================================================================
# 4. LLM Analysis
# ===================================================================

def _parse_llm_json(text: str):
    """Extract and parse JSON from LLM response, handling markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```\w*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return json.loads(cleaned.strip())


def _recover_partial_json_array(text: str) -> list:
    """Recover as many complete JSON objects as possible from a truncated array.

    When the LLM output is cut off mid-string, we find the last complete
    object (ending with '}') and close the array so we salvage valid data
    rather than discarding the entire chunk.
    """
    start = text.find("[")
    if start == -1:
        return []
    # Walk backward from the end to find the last complete object boundary
    for end_marker in ("}\n]", "},\n", "}, \n", "},", "}"):
        pos = text.rfind(end_marker, start)
        if pos != -1:
            candidate = text[start : pos + 1] + "]"
            try:
                return json.loads(candidate, strict=False)
            except json.JSONDecodeError:
                continue
    return []


# Pre-build a lookup for fast category normalisation
_CATEGORY_LOOKUP: dict[str, str] = {c.strip().lower(): c for c in config.CATEGORIES}

# Keyword sets per category for fuzzy matching when exact/substring fails
_CATEGORY_KEYWORDS: dict[str, set[str]] = {
    c: set(c.lower().replace("(", "").replace(")", "").replace(",", "").split())
    - {"and", "or", "the", "a", "an", "for", "of", "in", "to"}
    for c in config.CATEGORIES
}


def _normalize_category(raw: str) -> str | None:
    """Map an LLM-returned category to the closest config.CATEGORIES entry.

    Returns the canonical category string, or None if no match is found.
    Tries exact match (case-insensitive) first, then substring containment,
    then keyword overlap scoring.
    """
    raw_lower = raw.strip().lower()
    if not raw_lower:
        return None
    # Exact match (case-insensitive)
    if raw_lower in _CATEGORY_LOOKUP:
        return _CATEGORY_LOOKUP[raw_lower]
    # Substring match: check if one contains the other
    for valid_lower, valid in _CATEGORY_LOOKUP.items():
        if raw_lower in valid_lower or valid_lower in raw_lower:
            return valid
    # Keyword overlap: pick the category with the most keyword matches
    raw_words = set(
        raw_lower.replace("(", "").replace(")", "").replace(",", "").split()
    ) - {"and", "or", "the", "a", "an", "for", "of", "in", "to"}
    best_cat = None
    best_score = 0
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        overlap = len(raw_words & keywords)
        if overlap > best_score:
            best_score = overlap
            best_cat = cat
    # Require at least 2 keyword matches to avoid false positives
    if best_score >= 2 and best_cat:
        return best_cat
    return None


def _canonical_source(raw_identifier: str) -> str:
    """Map a raw domain or email address to its canonical provider name.

    Looks up config.SOURCE_MAPPING first.  Falls back to the raw identifier
    so new / unknown providers still get tracked (just without dedup).
    Always returns a lowercased canonical name for consistent deduplication.
    """
    key = raw_identifier.strip().lower()
    # Direct lookup (covers archive domains and known email addresses)
    if key in config.SOURCE_MAPPING:
        return config.SOURCE_MAPPING[key].lower()
    # Try the original casing too (mapping keys may have mixed case)
    if raw_identifier in config.SOURCE_MAPPING:
        return config.SOURCE_MAPPING[raw_identifier].lower()
    # Fallback: strip common prefixes like "www."
    stripped = key.removeprefix("www.")
    if stripped in config.SOURCE_MAPPING:
        return config.SOURCE_MAPPING[stripped].lower()
    return key


def _extract_sender_address(from_header: str) -> str:
    """Extract the bare email address from a From header.

    Handles formats like:
      "The Rundown AI <news+canned.response@daily.therundown.ai>"
      "news@alphasignal.ai"
    """
    match = re.search(r"<([^>]+)>", from_header)
    addr = match.group(1).strip().lower() if match else from_header.strip().lower()
    return addr or "unknown-sender"


def analyze_content(articles: list[dict], emails: list[dict]) -> dict:
    """Send collected content to the LLM and return structured tool data.

    Returns a dict with keys "tools_log" and "field_tools".

    Pipeline:
      1. LLM extraction: structured JSON with source attribution per chunk
      2. Python aggregation: deterministic ranking by distinct-source count
      3. LLM categorisation: assign ranked tools to field categories
    """
    # Build one entry per article / email — each entry is a self-contained
    # unit with its [Source: ...] header so it is never split across chunks.
    # The [Source: ...] tag uses the CANONICAL provider name so that the
    # same newsletter scraped from the web and received via Gmail maps to
    # one provider, not two.
    entries: list[str] = []

    for art in articles:
        domain = art["source"].split("/")[2]  # e.g. "www.superhuman.ai"
        source_name = _canonical_source(domain)
        entries.append(
            f"[Source: {source_name}]\nTitle: {art['title']}\n"
            f"Date: {art['date']}\nURL: {art['url']}\n"
            f"Content:\n{art['content']}\n---"
        )

    for email in emails:
        sender_addr = _extract_sender_address(email.get("from", ""))
        source_name = _canonical_source(sender_addr)
        entries.append(
            f"[Source: {source_name}]\n"
            f"Subject: {email['subject']}\nDate: {email['date']}\n"
            f"Content:\n{email['body'][:50000]}\n---"
        )

    log.info("Built %d entries (%d articles + %d emails).",
             len(entries), len(articles), len(emails))

    # Pack entries into chunks of ~150K chars (~37K tokens).
    # Each entry stays whole — no article is ever split across chunks.
    chunks = _chunk_by_entries(entries, max_chars=150_000)

    # Phase 1: extract structured tool mentions from each chunk
    all_mentions: list[dict] = []
    queue = list(enumerate(chunks))  # [(index, chunk_text), ...]
    while queue:
        i, chunk = queue.pop(0)
        log.info("LLM extraction pass (chunk %s, %d chars) ...", i, len(chunk))
        try:
            extracted, truncated = _llm_extract_tools(chunk)
        except Exception:
            log.error("Extraction chunk %s failed after retries; skipping.", i)
            extracted, truncated = [], False
        if isinstance(extracted, list):
            all_mentions.extend(extracted)
        else:
            log.warning("Extraction chunk %s returned non-list; skipping.", i)
        # If the response was truncated, split the chunk in half at an
        # entry boundary ("\n---\n") so that [Source: ...] headers stay
        # attached to their content.  Falls back to a plain newline split
        # only when no entry boundary exists near the midpoint.
        if truncated:
            separator = "\n---\n"
            mid_target = len(chunk) // 2
            # Search for the nearest entry boundary around the midpoint
            pos_before = chunk.rfind(separator, 0, mid_target)
            pos_after = chunk.find(separator, mid_target)
            if pos_before != -1 and pos_after != -1:
                # Pick whichever boundary is closer to the true midpoint
                mid = pos_before if (mid_target - pos_before) <= (pos_after - mid_target) else pos_after
            elif pos_before != -1:
                mid = pos_before
            elif pos_after != -1:
                mid = pos_after
            else:
                # No entry boundary found — fall back to newline split
                mid = chunk.rfind("\n", 0, mid_target)
                if mid == -1:
                    mid = mid_target
            split_at = mid + len(separator) if chunk[mid:mid + len(separator)] == separator else mid
            first_half = chunk[:split_at].rstrip("\n")
            second_half = chunk[split_at:].lstrip("\n")
            if first_half.strip() and second_half.strip():
                log.info("Splitting truncated chunk %s into two halves "
                         "(%d + %d chars) for re-extraction.",
                         i, len(first_half), len(second_half))
                queue.insert(0, (f"{i}b", second_half))
                queue.insert(0, (f"{i}a", first_half))
                time.sleep(30)
                continue
        if queue:
            time.sleep(30)

    log.info("Extracted %d total tool mentions across %d chunks.", len(all_mentions), len(chunks))

    # Phase 2: deterministic ranking in Python (no LLM needed)
    tools_log = _rank_tools_deterministic(all_mentions)
    log.info("Ranked %d tools deterministically by distinct-source count.", len(tools_log))

    # Phase 2b: LLM link fallback — fill "N/A" source_links via the LLM
    time.sleep(30)  # respect TPM limit before next LLM call
    log.info("LLM link fallback pass ...")
    try:
        tools_log = _llm_link_fallback(tools_log)
    except Exception:
        log.error("LLM link fallback failed after retries; continuing with existing links.")

    # Phase 2c: web search fallback — free DuckDuckGo search for remaining N/A links
    log.info("Web search fallback pass ...")
    try:
        tools_log = _web_search_link_fallback(tools_log)
    except Exception:
        log.error("Web search fallback failed; continuing with existing links.")

    # Build a fallback URL map from the enriched tools_log so that
    # _rank_field_tools_deterministic can use LLM-resolved links too.
    fallback_urls: dict[str, str] = {}
    for t in tools_log:
        link = t.get("source_link", "N/A")
        if link and link != "N/A":
            fallback_urls[t["tool_name"].lower()] = link

    # Phase 3: deterministic Python categorisation for field tools.
    # Categories per tool come from the per-mention categories (grounded
    # in the source use-cases by the extraction LLM); rank within each
    # category is by distinct positive source count, same as tools_log.
    log.info("Deterministic field_tools categorisation pass ...")
    field_tools = _rank_field_tools_deterministic(
        all_mentions, fallback_urls=fallback_urls,
    )
    log.info("Built %d field_tools entries across %d categories.",
             len(field_tools), len(config.CATEGORIES))

    return {"tools_log": tools_log, "field_tools": field_tools}


def _chunk_by_entries(entries: list[str], max_chars: int = 150_000) -> list[str]:
    """Pack entry strings into chunks, never splitting an entry across chunks.

    Each entry is one complete article or email (with its [Source: ...] header).
    This guarantees source attribution is always intact and no article is
    partially analysed.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for entry in entries:
        entry_len = len(entry)
        if entry_len > max_chars:
            # Single entry exceeds chunk size — flush current, then send it alone
            if current:
                chunks.append("\n".join(current))
                current, current_len = [], 0
            # Truncate as a last resort so we still analyse the beginning
            chunks.append(entry[:max_chars])
            log.warning("Single entry (%d chars) exceeds chunk limit; truncated.", entry_len)
            continue

        if current_len + entry_len + 1 > max_chars:
            # Current chunk is full — flush and start a new one
            chunks.append("\n".join(current))
            current, current_len = [], 0

        current.append(entry)
        current_len += entry_len + 1  # +1 for the joining newline

    if current:
        chunks.append("\n".join(current))

    return chunks


# ---------------------------------------------------------------
# 4a. LLM Extraction (structured JSON with source attribution)
# ---------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=10, max=60))
def _llm_extract_tools(text_chunk: str) -> tuple[list[dict], bool]:
    """Ask the LLM to list AI tools mentioned in a text chunk.

    Returns (tools, truncated) where *truncated* is True when the LLM
    response was cut off.  The caller can then split the chunk and retry
    the halves to avoid losing data.
    """
    _configure_gemini(_gemini_api_key_extract)
    categories_str = ", ".join(f'"{c}"' for c in config.CATEGORIES)
    model = genai.GenerativeModel(
        model_name=config.LLM_MODEL,
        system_instruction=textwrap.dedent(f"""            You are an AI-tools analyst. Given newsletter content, extract
            every AI tool explicitly mentioned in the provided text.

            CRITICAL RULES:
            - ONLY extract tools that are explicitly named in the text below.
            - Do NOT add any tools from your own knowledge or training data.
            - The "source" field MUST be copied exactly from the nearest
              [Source: ...] header above each article in the text.
            - If a tool is mentioned multiple times in the same source,
              include it only ONCE per source.
            - "categories" must be determined ONLY from what the source text
              says about the tool's use cases. Pick one or more from this list:
              [{categories_str}]
              If the text describes multiple use cases, include all matching
              categories. If none clearly match, use the closest one.
            - Do NOT invent or guess URLs that are not in the text.
            - Do NOT fabricate use cases — only report what the text states.

            For each tool return a JSON object with exactly these keys:
            - "tool_name": the exact name as it appears in the text
            - "source": the source identifier from the [Source: ...] header
            - "sentiment": "positive", "neutral", or "negative"
            - "use_case": what the source text says this tool is useful for
              (1 sentence, taken directly from the text; write "N/A" if the
              text gives no use case)
            - "categories": array of category strings from the list above
            - "description": 1-sentence summary of what the source says about it
            - "url": the tool's OFFICIAL HOMEPAGE URL if it is explicitly
              present in the text (e.g. "https://toolname.com"). Do NOT
              return links to newsletter posts, blog articles, Substack/
              Medium pages, news coverage, YouTube videos, tweets, or any
              other article-about-the-tool URL. If the text does not
              contain the tool's own website URL, return "N/A".

            Return a JSON array of objects. Nothing else.
        """),
        generation_config=genai.GenerationConfig(
            max_output_tokens=config.LLM_MAX_TOKENS,
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )
    response = model.generate_content(text_chunk)
    try:
        return json.loads(response.text, strict=False), False
    except json.JSONDecodeError:
        log.warning(
            "LLM response appears truncated (JSONDecodeError). "
            "Attempting partial recovery ..."
        )
        recovered = _recover_partial_json_array(response.text)
        if recovered:
            log.warning("Recovered %d tool mentions from truncated response.", len(recovered))
        else:
            log.warning("Could not recover any data from truncated response.")
        return recovered, True


# ---------------------------------------------------------------
# 4a-2. Homepage URL filter
# ---------------------------------------------------------------
# Hosting platforms / newsletter services that publish articles ABOUT tools
# rather than being the tool's own homepage.
_BLOG_HOSTS = (
    "substack.com", "medium.com", "beehiiv.com", "wordpress.com",
    "ghost.io", "ghost.org", "mailchi.mp", "buttondown.email",
    "convertkit.com", "blogspot.com", "tumblr.com", "hashnode.dev",
    "dev.to", "hackernoon.com", "techcrunch.com", "theverge.com",
    "venturebeat.com", "forbes.com", "nytimes.com", "wsj.com",
    "bloomberg.com", "reuters.com", "cnbc.com", "businessinsider.com",
    "wired.com", "arstechnica.com", "engadget.com", "mashable.com",
    "zdnet.com", "cnet.com", "theinformation.com", "axios.com",
    "semafor.com", "futurism.com", "technologyreview.com",
    "youtube.com", "youtu.be", "twitter.com", "x.com", "linkedin.com",
    "facebook.com", "instagram.com", "tiktok.com", "reddit.com",
    "github.io", "notion.site", "notion.so",
)

# URL path segments that indicate an article/blog post rather than a homepage.
_ARTICLE_PATH_MARKERS = (
    "/p/", "/post/", "/posts/", "/blog/", "/article/", "/articles/",
    "/newsletter/", "/news/", "/story/", "/stories/", "/i/",
    "/entry/", "/archive/", "/read/", "/issues/", "/issue-",
    "/2023/", "/2024/", "/2025/", "/2026/", "/@",
)


def _is_homepage_url(url: str) -> bool:
    """Return True if *url* looks like a tool homepage rather than an article.

    We reject URLs that live on known newsletter/blog platforms or whose path
    looks like an article slug.  This is heuristic but conservative: when in
    doubt we drop the URL and let the LLM fallback supply the real homepage.
    """
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url.strip())
    except Exception:
        return False
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return False

    host = parsed.netloc.lower().lstrip("www.")
    if any(bh in host for bh in _BLOG_HOSTS):
        return False

    path = parsed.path.lower()
    if any(marker in path for marker in _ARTICLE_PATH_MARKERS):
        return False

    # Long slug-like paths are almost certainly article URLs.
    slug = path.strip("/")
    if len(slug) > 50 or slug.count("-") >= 4:
        return False

    return True


# ---------------------------------------------------------------
# 4b. Deterministic Python ranking by distinct-source count
# ---------------------------------------------------------------
def _rank_tools_deterministic(mentions: list[dict]) -> list[dict]:
    """Rank tools by count of distinct sources with positive sentiment.

    Sorting: primary = distinct positive source count (descending),
             secondary = tool name alphabetical (ascending) for ties.
    Returns the top 25 tools.
    """
    tool_data: dict[str, dict] = defaultdict(lambda: {
        "sources": set(),
        "descriptions": [],
        "use_cases": [],
        "urls": [],
        "categories": [],
        "original_name": "",
    })

    for m in mentions:
        name = m.get("tool_name", "").strip()
        if not name:
            continue
        sentiment = m.get("sentiment", "").lower().strip()
        if sentiment != "positive":
            continue

        key = name.lower()
        source = _canonical_source(m.get("source", "unknown"))
        tool_data[key]["sources"].add(source)
        # Keep the first-seen original casing
        if not tool_data[key]["original_name"]:
            tool_data[key]["original_name"] = name
        if m.get("description"):
            tool_data[key]["descriptions"].append(m["description"])
        # Collect use cases from the text
        uc = m.get("use_case", "").strip()
        if uc and uc != "N/A" and uc not in tool_data[key]["use_cases"]:
            tool_data[key]["use_cases"].append(uc)
        if m.get("url") and m["url"] != "N/A" and _is_homepage_url(m["url"]):
            tool_data[key]["urls"].append(m["url"])
        # Collect categories from each mention, normalised to config list
        for cat in m.get("categories", []):
            normalised = _normalize_category(cat) if cat else None
            if normalised:
                tool_data[key]["categories"].append(normalised)

    # Sort: most distinct sources first, alphabetical for ties
    ranked = sorted(
        tool_data.items(),
        key=lambda x: (-len(x[1]["sources"]), x[0]),
    )

    result = []
    for key, data in ranked[:25]:
        # Deduplicate categories preserving order by frequency (most common first)
        seen = {}
        for cat in data["categories"]:
            seen[cat] = seen.get(cat, 0) + 1
        unique_cats = sorted(seen.keys(), key=lambda c: -seen[c])
        result.append({
            "tool_name": data["original_name"] or key,
            "category": ", ".join(unique_cats) if unique_cats else "",
            "mentions": len(data["sources"]),
            "use_cases": "; ".join(data["use_cases"][:5]) if data["use_cases"] else "N/A",
            "source_link": data["urls"][0] if data["urls"] else "N/A",
        })

    return result


# ---------------------------------------------------------------
# 4b-2. LLM link fallback for tools missing URLs
# ---------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=10, max=60))
def _llm_link_fallback(tools: list[dict]) -> list[dict]:
    """Ask the LLM to supply homepage URLs for tools that have no link.

    Only tools whose source_link is "N/A" are sent to the LLM.
    The LLM response is merged back; tools that already have a link are
    left untouched.  If the LLM cannot confidently determine a URL it
    should return "N/A" — we never want hallucinated links.
    """
    missing = [t for t in tools if t.get("source_link") in ("N/A", "", None)]
    if not missing:
        log.info("All tools already have links — skipping LLM link fallback.")
        return tools

    tool_names = [t["tool_name"] for t in missing]
    log.info("LLM link fallback: looking up URLs for %d tools …", len(tool_names))

    prompt = textwrap.dedent(f"""\
        For each AI tool listed below, provide the tool's OFFICIAL HOMEPAGE
        URL (the tool's own website — the page where a user would sign up
        for or download the product).

        RULES:
        - Return ONLY a JSON object mapping each tool name to its URL string.
        - The URL MUST be the tool's own website (e.g. "https://toolname.com"
          or "https://toolname.ai"), NOT a link to a review, article, blog
          post, newsletter issue, Wikipedia page, GitHub repo, YouTube
          video, tweet, or any third-party page ABOUT the tool.
        - Prefer the root domain (no long paths). Avoid URLs that contain
          "/blog/", "/post/", "/article/", "/news/" or year segments.
        - If you are NOT confident about the correct homepage, use "N/A".
        - Do NOT guess or fabricate URLs. Only provide URLs you are sure about.

        TOOLS:
        {json.dumps(tool_names)}
    """)

    model = genai.GenerativeModel(
        model_name=config.LLM_MODEL,
        system_instruction=(
            "Return only a valid JSON object mapping tool names to URL strings. "
            "No markdown. Use \"N/A\" when unsure."
        ),
        generation_config=genai.GenerationConfig(
            max_output_tokens=4096,
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )
    response = model.generate_content(prompt)
    url_map: dict[str, str] = json.loads(response.text)

    filled = 0
    for tool in tools:
        if tool.get("source_link") not in ("N/A", "", None):
            continue
        llm_url = url_map.get(tool["tool_name"], "N/A")
        if llm_url and llm_url != "N/A" and _is_homepage_url(llm_url):
            tool["source_link"] = llm_url
            filled += 1

    log.info("LLM link fallback filled %d / %d missing URLs.", filled, len(missing))
    return tools


# ---------------------------------------------------------------
# 4b-3. Web search fallback for tools still missing URLs
# ---------------------------------------------------------------
def _web_search_link_fallback(tools: list[dict]) -> list[dict]:
    """Search DuckDuckGo for homepage URLs of tools still missing links.

    Uses plain HTTP requests to DuckDuckGo HTML search — no API key,
    no LLM request, completely free.  Only fills in a URL when the
    top result passes the _is_homepage_url() filter.
    """
    import requests

    missing = [t for t in tools if t.get("source_link") in ("N/A", "", None)]
    if not missing:
        log.info("All tools already have links — skipping web search fallback.")
        return tools

    log.info("Web search fallback: looking up URLs for %d tools …", len(missing))
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }
    filled = 0
    for tool in missing:
        name = tool["tool_name"]
        query = quote_plus(f"{name} AI tool official website")
        try:
            resp = requests.get(
                f"https://html.duckduckgo.com/html/?q={query}",
                headers=headers,
                timeout=10,
            )
            resp.raise_for_status()
        except Exception as exc:
            log.debug("Web search failed for %s: %s", name, exc)
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        # DuckDuckGo HTML results have class "result__a" for the title links
        for link in soup.select("a.result__a"):
            href = link.get("href", "")
            if not href:
                continue
            # DuckDuckGo sometimes wraps URLs in a redirect; extract the real URL
            if "uddg=" in href:
                from urllib.parse import parse_qs, urlparse as _urlparse
                qs = parse_qs(_urlparse(href).query)
                href = qs.get("uddg", [href])[0]
            if _is_homepage_url(href):
                tool["source_link"] = href
                filled += 1
                log.debug("Web search found URL for %s: %s", name, href)
                break

        # Be polite — small delay between searches
        time.sleep(1)

    log.info("Web search fallback filled %d / %d missing URLs.", filled, len(missing))
    return tools


# ---------------------------------------------------------------
# 4c. Deterministic Python categorisation for field tools
# ---------------------------------------------------------------
def _rank_field_tools_deterministic(
    all_mentions: list[dict],
    fallback_urls: dict[str, str] | None = None,
) -> list[dict]:
    """Assign tools to categories deterministically in Python.

    Each tool's categories come from the per-mention `categories` field
    (which the extraction LLM derived ONLY from the use-cases stated in
    the scraped articles/emails). Ranking within each category is by the
    number of distinct positive sources that mentioned the tool *for that
    specific category*, so a tool's score in the "coding" category only
    counts sources that described it as useful for coding. The top 5 tools
    per category are returned.

    *fallback_urls* is an optional {tool_name_lower: url} map produced by the
    LLM link-fallback step; it supplements URLs that were missing from the
    original article/email content.
    """
    fallback_urls = fallback_urls or {}
    # Build a deduplicated summary of positive tools with per-category
    # source counts.  ``sources_by_cat`` tracks the distinct canonical
    # sources that mentioned each tool *for a specific category*, so the
    # ranking reflects how many sources praised the tool for that use-case.
    tool_info: dict[str, dict] = defaultdict(lambda: {
        "sources_by_cat": defaultdict(set),
        "descriptions": [], "use_cases": [],
        "urls": [], "categories": [], "original_name": "",
    })
    for m in all_mentions:
        name = m.get("tool_name", "").strip()
        if not name or m.get("sentiment", "").lower() != "positive":
            continue
        key = name.lower()
        source = _canonical_source(m.get("source", ""))
        if not tool_info[key]["original_name"]:
            tool_info[key]["original_name"] = name
        if m.get("description"):
            tool_info[key]["descriptions"].append(m["description"])
        uc = m.get("use_case", "").strip()
        if uc and uc != "N/A" and uc not in tool_info[key]["use_cases"]:
            tool_info[key]["use_cases"].append(uc)
        if m.get("url") and m["url"] != "N/A" and _is_homepage_url(m["url"]):
            tool_info[key]["urls"].append(m["url"])
        for cat in m.get("categories", []):
            normalised = _normalize_category(cat) if cat else None
            if normalised:
                tool_info[key]["categories"].append(normalised)
                tool_info[key]["sources_by_cat"][normalised].add(source)

    # For each category in the fixed list, collect tools whose extracted
    # categories include it, then rank by distinct positive sources
    # *for that specific category*.
    results: list[dict] = []
    for category in config.CATEGORIES:
        candidates = []
        for key, data in tool_info.items():
            if category not in set(data["categories"]):
                continue
            cat_count = len(data["sources_by_cat"].get(category, set()))
            candidates.append((key, data, cat_count))

        # Sort: most distinct positive sources first, alphabetical for ties
        candidates.sort(key=lambda x: (-x[2], x[0]))

        for rank, (key, data, count) in enumerate(candidates[:5], start=1):
            name = data["original_name"] or key
            url = data["urls"][0] if data["urls"] else fallback_urls.get(key, "N/A")
            # why_recommended is built from the use-case evidence extracted
            # from the source text (already grounded in the scraped JSON).
            if data["use_cases"]:
                why = "; ".join(data["use_cases"][:3])
            elif data["descriptions"]:
                why = data["descriptions"][0]
            else:
                why = "N/A"
            results.append({
                "field": category,
                "rank": rank,
                "tool_name": name,
                "why_recommended": why,
                "url": url,
            })

    return results


# ===================================================================
# 5. Google Sheets Output
# ===================================================================
def write_to_sheets(creds: Credentials, data: dict) -> None:
    """Write the analysed tool data to the two Google Sheets tabs."""
    spreadsheet_id = config.SPREADSHEET_ID or os.getenv("SPREADSHEET_ID", "")
    if not spreadsheet_id:
        raise ValueError(
            "SPREADSHEET_ID is not set. Set it in config.py or as an env var."
        )

    service = build("sheets", "v4", credentials=creds)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # --- Tab 1: AI Tools Log ---
    _ensure_tab_exists(service, spreadsheet_id, config.TAB_AI_TOOLS_LOG)

    rows_log = []
    for tool in data.get("tools_log", []):
        rows_log.append([
            today,
            tool.get("tool_name", ""),
            tool.get("category", ""),
            tool.get("mentions", 0),
            tool.get("use_cases", ""),
            tool.get("source_link", ""),
        ])

    if rows_log:
        _overwrite_rows(service, spreadsheet_id, config.TAB_AI_TOOLS_LOG,
                        ["Date Logged", "Tool Name", "Category",
                         "Positive Sources", "Use Cases", "Source Link"],
                        rows_log)
        log.info("Wrote %d rows to '%s'.", len(rows_log), config.TAB_AI_TOOLS_LOG)

    # --- Tab 2: Field Tools ---
    _ensure_tab_exists(service, spreadsheet_id, config.TAB_FIELD_TOOLS)

    rows_field = []
    for entry in data.get("field_tools", []):
        rows_field.append([
            entry.get("field", ""),
            entry.get("rank", ""),
            entry.get("tool_name", ""),
            entry.get("why_recommended", ""),
            entry.get("url", ""),
        ])

    if rows_field:
        _overwrite_rows(service, spreadsheet_id, config.TAB_FIELD_TOOLS,
                        ["Field/Action", "Rank", "Tool Name",
                         "Why it's Recommended", "URL"],
                        rows_field)
        log.info("Wrote %d rows to '%s'.", len(rows_field), config.TAB_FIELD_TOOLS)


def _ensure_tab_exists(service, spreadsheet_id: str, tab_name: str) -> None:
    """Create the tab if it does not already exist in the spreadsheet."""
    meta = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    existing = [s["properties"]["title"].strip() for s in meta.get("sheets", [])]
    if tab_name.strip() not in existing:
        service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"requests": [{"addSheet": {"properties": {"title": tab_name}}}]},
        ).execute()
        log.info("Created new sheet tab: '%s'", tab_name)


def _overwrite_rows(service, spreadsheet_id: str, tab_name: str,
                    headers: list[str], rows: list[list]) -> None:
    """Clear the tab and write headers + rows from scratch."""
    service.spreadsheets().values().clear(
        spreadsheetId=spreadsheet_id,
        range=f"'{tab_name}'!A:Z",
    ).execute()
    service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=f"'{tab_name}'!A1",
        valueInputOption="USER_ENTERED",
        body={"values": [headers] + rows},
    ).execute()


# ===================================================================
# 6. Main Orchestrator
# ===================================================================
def main() -> None:
    """Run the full extraction -> analysis -> output pipeline."""
    log.info("=== AI Tools Extraction Pipeline ===")

    if not _gemini_api_key_extract or not _gemini_api_key_fields:
        raise RuntimeError(
            "Gemini API key(s) not set. Set GEMINI_API_KEY (primary) and "
            "optionally GEMINI_API_KEY_EXTRACT / GEMINI_API_KEY_FIELDS."
        )

    # Step 1: Authenticate
    log.info("Authenticating with Google APIs ...")
    creds = get_google_credentials()

    # Step 2: Collect data from both sources
    log.info("Fetching emails from Gmail ...")
    emails = fetch_emails(creds)

    log.info("Scraping newsletter archives ...")
    articles = scrape_archives()

    if not articles and not emails:
        log.warning("No content collected from any source. Exiting.")
        return

    log.info(
        "Collected %d articles and %d emails. Sending to LLM for analysis ...",
        len(articles), len(emails),
    )

    # Step 3: LLM analysis + deterministic ranking
    data = analyze_content(articles, emails)

    # Step 4: Write results to Google Sheets
    log.info("Writing results to Google Sheets ...")
    write_to_sheets(creds, data)

    log.info("=== Pipeline complete ===")


if __name__ == "__main__":
    main()
