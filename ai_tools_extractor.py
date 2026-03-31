#!/usr/bin/env python3
"""
AI Tools Extraction Pipeline
=============================
Scrapes AI newsletter archives and Gmail, uses an LLM to extract / rank /
categorise AI tools, and writes the results to Google Sheets.

Usage:
    1. Place your Google OAuth `credentials.json` in the project root.
    2. Set SPREADSHEET_ID in config.py (or via env var SPREADSHEET_ID).
    3. Export your Gemini key:  export GEMINI_API_KEY="your-key-here"
    4. Run:  python ai_tools_extractor.py
"""

import base64
import json
import logging
import os
import re
import textwrap
import time
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
# Rate-limit helpers
# ---------------------------------------------------------------------------
_last_llm_call_time: float = 0.0   # timestamp of the most recent LLM call
_llm_calls_today: int = 0          # simple counter for RPD awareness


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return len(text) // 4


def _rate_limit_wait(estimated_input_tokens: int) -> None:
    """Sleep enough to respect RPM and TPM limits before the next LLM call.

    Strategy: after every request we must wait long enough so that the
    tokens from this request have "left" the 1-minute rolling window.
    We also enforce a minimum gap of ``60 / RPM_LIMIT`` seconds.
    """
    global _last_llm_call_time, _llm_calls_today

    estimated_total = estimated_input_tokens + config.LLM_MAX_TOKENS  # input + max output
    # Seconds of TPM budget this request consumes
    tpm_wait = (estimated_total / config.TPM_LIMIT) * 60
    # Minimum gap for RPM
    rpm_wait = 60 / config.RPM_LIMIT  # 6 s for RPM=10

    required_gap = max(tpm_wait, rpm_wait)

    now = time.time()
    elapsed = now - _last_llm_call_time if _last_llm_call_time else required_gap
    if elapsed < required_gap:
        sleep_for = required_gap - elapsed + 1  # +1 s safety margin
        log.info(
            "Rate-limit: sleeping %.1fs (est. %d tokens, TPM gap=%.1fs, RPM gap=%.1fs)",
            sleep_for, estimated_total, tpm_wait, rpm_wait,
        )
        time.sleep(sleep_for)

    _last_llm_call_time = time.time()
    _llm_calls_today += 1
    if _llm_calls_today > config.RPD_LIMIT * 0.8:
        log.warning("Approaching daily request limit: %d / %d RPD used.", _llm_calls_today, config.RPD_LIMIT)

# ---------------------------------------------------------------------------
# Google OAuth scopes
# ---------------------------------------------------------------------------
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/spreadsheets",
]


# ===================================================================
# 1. Google Authentication
# ===================================================================
def get_google_credentials() -> Credentials:
    """Authenticate with Google using OAuth2 and return credentials.

    Looks for a cached token in ``token.json``; if absent or expired,
    opens the browser-based OAuth consent flow using ``credentials.json``.
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

    Returns a list of dicts: ``{"subject": ..., "date": ..., "body": ...}``.
    Only emails from the last ``config.LOOKBACK_DAYS`` days are returned.
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
        log.warning("Gmail label '%s' not found — skipping email source.", config.GMAIL_LABEL)
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
            "body": body_text,
        })

    return results


def _extract_email_body(payload: dict) -> str:
    """Recursively extract plain-text (or decoded HTML) from a Gmail payload."""
    parts = payload.get("parts", [])
    if not parts:
        data = payload.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
        return ""

    # Prefer text/plain, fall back to text/html
    for mime in ("text/plain", "text/html"):
        for part in parts:
            if part.get("mimeType") == mime:
                data = part.get("body", {}).get("data", "")
                if data:
                    decoded = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
                    if mime == "text/html":
                        return BeautifulSoup(decoded, "html.parser").get_text(separator="\n")
                    return decoded
            # Handle nested multipart
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
    """Scrape all configured archive URLs for posts from the last 14 days.

    Returns a list of dicts: ``{"source": ..., "title": ..., "date": ...,
    "url": ..., "content": ...}``.
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
                log.exception("Failed to scrape %s — skipping.", archive_url)

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
    for _ in range(30):  # up to 30 scrolls (~30 days of articles)
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(2000)  # wait 2s for new content to load
        new_height = page.evaluate("document.body.scrollHeight")
        if new_height == prev_height:
            break  # no more content loading
        prev_height = new_height

    html = page.content()

    # Save rendered HTML for debugging (first archive only)
    debug_file = f"debug_{archive_url.split('/')[2]}.html"
    with open(debug_file, "w", encoding="utf-8") as f:
        f.write(html)
    log.info("  -> saved rendered HTML to %s", debug_file)

    soup = BeautifulSoup(html, "html.parser")
    articles: list[dict] = []

    link_candidates = _find_article_links(soup, archive_url)
    log.info("  -> found %d link candidates on archive page", len(link_candidates))
    for title, url, date_str in link_candidates[:5]:
        log.info("     sample: [%s] %s (%s)", date_str, title[:60], url[:80])

    for title, url, date_str in link_candidates[: config.MAX_ARTICLES_PER_SOURCE]:
        pub_date = _parse_date_safe(date_str)
        if pub_date and pub_date < cutoff:
            continue  # older than 14 days — skip

        try:
            page_html = _fetch_page(url, page)
            page_soup = BeautifulSoup(page_html, "html.parser")
            # Try <article>, then main, then body
            article_tag = (
                page_soup.find("article")
                or page_soup.find("main")
                or page_soup.find("body")
            )
            content = article_tag.get_text(separator="\n", strip=True) if article_tag else ""
        except Exception:
            log.warning("Could not fetch article: %s", url)
            content = title  # fall back to just the title

        articles.append({
            "source": archive_url,
            "title": title,
            "date": date_str,
            "url": url,
            "content": content[:8000],  # cap to avoid token explosion
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
        # Look for a sibling/parent time tag
        date_str = _find_nearby_date(a_tag)
        if title:
            results.append((title, href, date_str))

    # Strategy 2: Generic — any <a> whose href contains "/p/" or "/post/" or "/newsletter/"
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

    # Strategy 3: Broad fallback — grab all links that look like articles
    if not results:
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if not href.startswith("http"):
                href = base + href
            # Skip navigation / footer links
            if any(skip in href for skip in ["#", "javascript:", "/archive", "/login", "/subscribe"]):
                continue
            title = a_tag.get_text(strip=True)
            if title and len(title) > 15:
                date_str = _find_nearby_date(a_tag)
                results.append((title, href, date_str))

    return results


def _find_nearby_date(tag) -> str:
    """Search parent/sibling elements for a <time> tag or date-like text."""
    # Check for <time> in parent containers (up 3 levels)
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
def analyze_content(articles: list[dict], emails: list[dict]) -> dict:
    """Send collected content to the LLM and return structured tool data.

    Returns a dict with keys ``"tools_log"`` and ``"field_tools"``.
    """
    # Configure Gemini once for the whole analysis phase
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    # Build a combined text digest for the LLM
    digest_parts: list[str] = []

    for art in articles:
        digest_parts.append(
            f"[Source: {art['source']}]\nTitle: {art['title']}\n"
            f"Date: {art['date']}\nURL: {art['url']}\n"
            f"Content:\n{art['content']}\n{'---'}\n"
        )

    for email in emails:
        digest_parts.append(
            f"[Source: Gmail / {config.GMAIL_LABEL}]\n"
            f"Subject: {email['subject']}\nDate: {email['date']}\n"
            f"Content:\n{email['body'][:6000]}\n{'---'}\n"
        )

    full_digest = "\n".join(digest_parts)
    log.info(
        "Total digest size: %d chars ≈ %d tokens (from %d articles + %d emails)",
        len(full_digest), _estimate_tokens(full_digest), len(articles), len(emails),
    )

    # Split into chunks that fit within the context window while leaving room
    # for the system prompt and output.  800k chars ≈ 200k tokens; with 32k
    # output headroom and system prompt, this stays well inside the 1M context.
    chunks = _chunk_text(full_digest, max_chars=800_000)
    log.info("Split into %d chunk(s) for extraction.", len(chunks))

    # Phase 1: extract raw tool mentions from each chunk
    raw_mentions: list[str] = []
    for i, chunk in enumerate(chunks):
        log.info("LLM extraction pass %d/%d (%d chars) …", i + 1, len(chunks), len(chunk))
        _rate_limit_wait(_estimate_tokens(chunk))
        raw_mentions.append(_llm_extract_tools(chunk))

    # Phase 2: aggregate extracted mentions into a clean summary so the
    # ranking LLM gets a concise, pre-counted input (no tool is lost in
    # a wall of raw text).
    aggregated = _aggregate_mentions(raw_mentions)
    log.info(
        "Aggregated mentions: %d chars ≈ %d tokens",
        len(aggregated), _estimate_tokens(aggregated),
    )

    # Phase 3: rank and categorise
    log.info("LLM ranking & categorisation pass …")
    final_json = _llm_rank_and_categorise(aggregated)

    return final_json


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Split text into chunks of approximately max_chars."""
    if len(text) <= max_chars:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + max_chars
        # Try to split at a paragraph boundary
        boundary = text.rfind("\n---\n", start, end)
        if boundary > start:
            end = boundary + 5
        chunks.append(text[start:end])
        start = end
    return chunks


def _aggregate_mentions(raw_mentions: list[str]) -> str:
    """Parse extraction results and aggregate tool mentions.

    Returns a condensed summary with per-tool mention counts so the ranking
    LLM receives a clean, complete picture of every tool found.  Falls back
    to raw concatenation if JSON parsing fails for all chunks.
    """
    all_tools: dict[str, dict] = {}  # tool_name -> aggregated data
    unparsed_chunks: list[str] = []

    for raw in raw_mentions:
        # Strip markdown code fences that Gemini sometimes adds
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        try:
            tools = json.loads(cleaned)
            if not isinstance(tools, list):
                raise ValueError("Expected a JSON array")
        except (json.JSONDecodeError, ValueError):
            log.warning("Could not parse extraction chunk as JSON; keeping raw text.")
            unparsed_chunks.append(raw)
            continue

        for tool in tools:
            # Handle varying key names from the LLM
            name = (
                tool.get("Tool Name")
                or tool.get("tool_name")
                or tool.get("name")
                or "Unknown"
            ).strip()
            sentiment = (
                tool.get("Sentiment")
                or tool.get("sentiment")
                or "neutral"
            ).lower().strip()
            desc = (
                tool.get("Short description")
                or tool.get("description")
                or ""
            ).strip()
            url = (
                tool.get("Source URL")
                or tool.get("source_url")
                or tool.get("url")
                or "N/A"
            ).strip()

            if name not in all_tools:
                all_tools[name] = {
                    "positive": 0,
                    "neutral": 0,
                    "negative": 0,
                    "descriptions": [],
                    "urls": set(),
                }

            entry = all_tools[name]
            if sentiment in ("positive", "neutral", "negative"):
                entry[sentiment] += 1
            else:
                entry["neutral"] += 1
            if desc and desc not in entry["descriptions"]:
                entry["descriptions"].append(desc)
            if url and url != "N/A":
                entry["urls"].add(url)

    # Build the condensed summary
    lines: list[str] = []
    if all_tools:
        sorted_tools = sorted(
            all_tools.items(),
            key=lambda x: x[1]["positive"],
            reverse=True,
        )
        for name, data in sorted_tools:
            total = data["positive"] + data["neutral"] + data["negative"]
            desc = data["descriptions"][0] if data["descriptions"] else ""
            urls = ", ".join(sorted(data["urls"])) if data["urls"] else "N/A"
            lines.append(
                f"Tool: {name} | Positive mentions: {data['positive']} | "
                f"Neutral: {data['neutral']} | Negative: {data['negative']} | "
                f"Total: {total} | Description: {desc} | URLs: {urls}"
            )
        log.info("Aggregated %d unique tools from parsed JSON.", len(all_tools))

    # Append any unparsed chunks as fallback (so no content is lost)
    if unparsed_chunks:
        lines.append("\n--- RAW MENTIONS (could not parse as JSON) ---")
        lines.extend(unparsed_chunks)
        log.info("Included %d unparsed chunk(s) as raw text fallback.", len(unparsed_chunks))

    return "\n".join(lines)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30))
def _llm_extract_tools(text_chunk: str) -> str:
    """Ask the LLM to list AI tools mentioned in a text chunk.

    Uses ``response_mime_type="application/json"`` so the output is
    guaranteed to be parseable JSON, which enables reliable aggregation.
    """
    model = genai.GenerativeModel(
        model_name=config.LLM_MODEL,
        system_instruction=textwrap.dedent("""\
            You are an AI-tools analyst. Given newsletter content, extract
            every AI tool mentioned. For each tool output:
            - tool_name
            - sentiment  (one of: positive, neutral, negative)
            - description  (1 sentence)
            - source_url  (the tool's own URL if mentioned, else "N/A")
            Return the results as a JSON array of objects. Include EVERY
            tool you find — do not skip any.
        """),
        generation_config=genai.GenerationConfig(
            max_output_tokens=config.LLM_MAX_TOKENS,
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )
    response = model.generate_content(text_chunk)
    return response.text


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30))
def _llm_tools_log(mentions_text: str) -> list:
    """Ask the LLM for the top 25 tools ranked by mentions."""
    prompt = textwrap.dedent(f"""\
        Below are AI tool mentions extracted from multiple newsletter sources.
        The mention counts are already pre-aggregated. Return a JSON array of
        the top 25 tools sorted by positive mentions (descending).

        OUTPUT FORMAT (valid JSON array only, no markdown):
        [
          {{
            "tool_name": "...",
            "category": "...",
            "mentions": <int>,
            "description": "1 sentence.",
            "source_link": "https://..."
          }}
        ]

        RULES:
        - "mentions" = use the pre-counted positive mentions from the data.
        - "source_link" = the tool's own website, not the newsletter.
        - "description" = 1 sentence max.
        - Include exactly 25 tools (or fewer if fewer than 25 exist).

        MENTIONS DATA:
        {mentions_text}
    """)

    _rate_limit_wait(_estimate_tokens(prompt))
    model = genai.GenerativeModel(
        model_name=config.LLM_MODEL,
        system_instruction="Return only a valid JSON array. No markdown.",
        generation_config=genai.GenerationConfig(
            max_output_tokens=config.LLM_MAX_TOKENS,
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )
    response = model.generate_content(prompt)
    return json.loads(response.text)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30))
def _llm_field_tools(mentions_text: str) -> list:
    """Ask the LLM for the top 5 tools per category."""
    categories_str = "\n".join(f"- {c}" for c in config.CATEGORIES)

    prompt = textwrap.dedent(f"""\
        Below are AI tool mentions extracted from multiple newsletter sources.
        The mention counts are already pre-aggregated. For EACH of the
        {len(config.CATEGORIES)} categories below, return the top 5 tools
        ranked 1-5.

        CATEGORIES:
        {categories_str}

        OUTPUT FORMAT (valid JSON array only, no markdown):
        [
          {{
            "field": "<category name>",
            "rank": <1-5>,
            "tool_name": "...",
            "why_recommended": "1 sentence.",
            "url": "https://..."
          }}
        ]

        RULES:
        - rank 1 = best in category.
        - If fewer than 5 tools exist for a category, include as many as possible.
        - "url" = the tool's own website.
        - "why_recommended" = 1 sentence max.
        - You MUST include entries for ALL {len(config.CATEGORIES)} categories.

        MENTIONS DATA:
        {mentions_text}
    """)

    _rate_limit_wait(_estimate_tokens(prompt))
    model = genai.GenerativeModel(
        model_name=config.LLM_MODEL,
        system_instruction="Return only a valid JSON array. No markdown.",
        generation_config=genai.GenerationConfig(
            max_output_tokens=config.LLM_MAX_TOKENS,
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )
    response = model.generate_content(prompt)
    return json.loads(response.text)


def _llm_rank_and_categorise(mentions_text: str) -> dict:
    """Run two separate LLM calls for tools_log and field_tools.

    No ``@retry`` here — the individual LLM functions already retry
    internally.  A wrapper retry would re-run already-succeeded calls,
    wasting RPD budget.
    """
    log.info("LLM tools_log pass …")
    tools_log = _llm_tools_log(mentions_text)
    log.info("LLM field_tools pass …")
    field_tools = _llm_field_tools(mentions_text)
    return {"tools_log": tools_log, "field_tools": field_tools}


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
            tool.get("description", ""),
            tool.get("source_link", ""),
        ])

    if rows_log:
        _overwrite_rows(service, spreadsheet_id, config.TAB_AI_TOOLS_LOG,
                        ["Date Logged", "Tool Name", "Category", "Mentions", "Description", "Source Link"],
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
                        ["Field/Action", "Rank", "Tool Name", "Why it's Recommended", "URL"],
                        rows_field)
        log.info("Wrote %d rows to '%s'.", len(rows_field), config.TAB_FIELD_TOOLS)


def _ensure_tab_exists(service, spreadsheet_id: str, tab_name: str) -> None:
    """Create the tab if it doesn't already exist in the spreadsheet."""
    meta = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    existing = [s["properties"]["title"].strip() for s in meta.get("sheets", [])]
    if tab_name.strip() not in existing:
        service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"requests": [{"addSheet": {"properties": {"title": tab_name}}}]},
        ).execute()
        log.info("Created new sheet tab: '%s'", tab_name)


def _ensure_headers(service, spreadsheet_id: str, tab_name: str, headers: list[str]) -> None:
    """Write header row if the tab is empty."""
    result = (
        service.spreadsheets()
        .values()
        .get(spreadsheetId=spreadsheet_id, range=f"'{tab_name}'!A1:Z1")
        .execute()
    )
    if not result.get("values"):
        service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=f"'{tab_name}'!A1",
            valueInputOption="RAW",
            body={"values": [headers]},
        ).execute()


def _overwrite_rows(service, spreadsheet_id: str, tab_name: str, headers: list[str], rows: list[list]) -> None:
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
    """Run the full extraction → analysis → output pipeline."""
    log.info("=== AI Tools Extraction Pipeline ===")

    # Step 1: Authenticate
    log.info("Authenticating with Google APIs …")
    creds = get_google_credentials()

    # Step 2: Collect data from both sources
    log.info("Fetching emails from Gmail …")
    emails = fetch_emails(creds)

    log.info("Scraping newsletter archives …")
    articles = scrape_archives()

    if not articles and not emails:
        log.warning("No content collected from any source. Exiting.")
        return

    log.info(
        "Collected %d articles and %d emails. Sending to LLM for analysis …",
        len(articles), len(emails),
    )

    # Step 3: LLM analysis
    data = analyze_content(articles, emails)

    # Step 4: Write results to Google Sheets
    log.info("Writing results to Google Sheets …")
    write_to_sheets(creds, data)

    log.info("=== Pipeline complete ===")


if __name__ == "__main__":
    main()
