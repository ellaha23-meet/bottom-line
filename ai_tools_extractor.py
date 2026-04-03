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
    """Scrape all configured archive URLs for posts from the last LOOKBACK_DAYS days.

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
    soup = BeautifulSoup(html, "html.parser")
    articles: list[dict] = []

    link_candidates = _find_article_links(soup, archive_url)
    log.info("  -> found %d link candidates on archive page", len(link_candidates))
    for title, url, date_str in link_candidates[:5]:
        log.info("     sample: [%s] %s (%s)", date_str, title[:60], url[:80])

    for title, url, date_str in link_candidates[: config.MAX_ARTICLES_PER_SOURCE]:
        pub_date = _parse_date_safe(date_str)
        if pub_date and pub_date < cutoff:
            continue  # older than lookback window — skip

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

    seen_urls = {r[1] for r in results}

    # Strategy 2: Generic — any <a> whose href contains "/p/" or "/post/" or "/newsletter/"
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if any(seg in href for seg in ["/p/", "/post/", "/newsletter/", "/i/"]):
            if not href.startswith("http"):
                href = base + href
            title = a_tag.get_text(strip=True) or href.split("/")[-1]
            date_str = _find_nearby_date(a_tag)
            if title and href not in seen_urls:
                seen_urls.add(href)
                results.append((title, href, date_str))

    # Strategy 3: Broad fallback — only if strategies 1+2 found nothing
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
                if href not in seen_urls:
                    seen_urls.add(href)
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
# 4. LLM Analysis (extraction only) + Deterministic Ranking
# ===================================================================
def analyze_content(articles: list[dict], emails: list[dict]) -> dict:
    """Extract tool mentions via LLM, then rank deterministically in code.

    Phase 1 (LLM): Extract structured tool mentions with source attribution.
    Phase 2 (Code): Count distinct positive sources per tool and rank.

    Returns a dict with keys ``"tools_log"`` and ``"field_tools"``.
    """
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
            f"Content:\n{email['body']}\n{'---'}\n"
        )

    # Split into chunks for the LLM (≈800 000 chars ≈ 200k tokens)
    chunks = _chunk_text("\n".join(digest_parts), max_chars=800_000)

    # Phase 1: LLM extracts structured tool mentions from each chunk
    all_mentions: list[dict] = []
    for i, chunk in enumerate(chunks):
        log.info("LLM extraction pass %d/%d …", i + 1, len(chunks))
        raw_json = _llm_extract_tools(chunk)
        parsed = _parse_llm_json(raw_json)
        log.info("  -> extracted %d tool mentions", len(parsed))
        all_mentions.extend(parsed)
        if i < len(chunks) - 1:
            time.sleep(15)  # respect rate limit

    # Phase 2: deterministic aggregation and ranking (in Python, not LLM)
    log.info("Aggregating and ranking %d total tool mentions …", len(all_mentions))
    tools_log = _build_tools_log(all_mentions)
    field_tools = _build_field_tools(all_mentions)

    return {"tools_log": tools_log, "field_tools": field_tools}


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


# ---------------------------------------------------------------------------
# Phase 1 helpers: LLM extraction
# ---------------------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30))
def _llm_extract_tools(text_chunk: str) -> str:
    """Ask the LLM to list AI tools mentioned in a text chunk.

    Returns raw JSON text. Each mention includes the source newsletter
    name so that we can count distinct sources in Python.
    """
    categories_str = ", ".join(f'"{c}"' for c in config.CATEGORIES)

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(
        model_name=config.LLM_MODEL,
        system_instruction=textwrap.dedent(f"""\
            You are an AI-tools analyst.

            CRITICAL RULES:
            - Base your answers ONLY on the newsletter content provided below.
            - Do NOT use prior knowledge about these tools.
            - For descriptions and sentiment, use ONLY the provided text.
            - For tool_url, prefer URLs from the text but fall back to your
              knowledge of the tool's official website if not found.
            - Extract EVERY AI tool mentioned in the text.

            For each tool, return a JSON object with these exact keys:
            - "tool_name": the exact name of the tool as written in the text
            - "sentiment": one of "positive", "neutral", or "negative" — based
              on how the source describes the tool
            - "description": one sentence describing the tool, using ONLY what
              the newsletter says about it
            - "tool_url": the tool's website URL if mentioned in the text;
              if not found in the text, use your knowledge to provide the
              tool's official website URL
            - "source_name": the newsletter/source that mentioned this tool
              (copy from the [Source: ...] header above each article)
            - "category": the single best-fitting category from this list:
              [{categories_str}]

            Return ONLY a valid JSON array of objects. No markdown fences.
        """),
        generation_config=genai.GenerationConfig(
            max_output_tokens=config.LLM_MAX_TOKENS,
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )
    response = model.generate_content(text_chunk)
    return response.text


def _parse_llm_json(raw_text: str) -> list[dict]:
    """Parse a JSON array from LLM output, handling common formatting issues."""
    text = raw_text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        return [data]
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        log.warning("Could not parse LLM JSON response: %s…", text[:200])
        return []


# ---------------------------------------------------------------------------
# Phase 2 helpers: deterministic aggregation & ranking
# ---------------------------------------------------------------------------
def _normalize_tool_name(name: str) -> str:
    """Normalize a tool name for deduplication (lowercase, collapse whitespace)."""
    return re.sub(r"\s+", " ", name.strip().lower())


def _match_category(category: str) -> str | None:
    """Match a free-form category string to the closest config.CATEGORIES entry."""
    cat_lower = category.strip().lower()
    for c in config.CATEGORIES:
        if c.lower() == cat_lower:
            return c
    # Substring match as fallback
    for c in config.CATEGORIES:
        if cat_lower in c.lower() or c.lower() in cat_lower:
            return c
    return None


def _build_tools_log(mentions: list[dict]) -> list[dict]:
    """Deterministically rank tools by number of distinct positive sources.

    Returns the top 25 tools sorted by distinct-source count (descending),
    with alphabetical tool name as tie-breaker.
    """
    tool_data: dict[str, dict] = {}

    for m in mentions:
        if m.get("sentiment", "").lower() != "positive":
            continue

        norm = _normalize_tool_name(m.get("tool_name", ""))
        if not norm:
            continue

        if norm not in tool_data:
            tool_data[norm] = {
                "tool_name": m.get("tool_name", ""),
                "category": m.get("category", ""),
                "description": m.get("description", ""),
                "tool_url": m.get("tool_url", "N/A"),
                "sources": set(),
            }

        tool_data[norm]["sources"].add(m.get("source_name", "unknown"))

        # Keep the longer / more informative description
        if len(m.get("description", "")) > len(tool_data[norm]["description"]):
            tool_data[norm]["description"] = m["description"]
        if m.get("tool_url", "N/A") != "N/A":
            tool_data[norm]["tool_url"] = m["tool_url"]

    ranked = sorted(
        tool_data.values(),
        key=lambda t: (-len(t["sources"]), t["tool_name"].lower()),
    )

    result = []
    for tool in ranked[:25]:
        result.append({
            "tool_name": tool["tool_name"],
            "category": tool["category"],
            "mentions": len(tool["sources"]),
            "description": tool["description"],
            "source_link": tool["tool_url"],
        })

    return result


def _build_field_tools(mentions: list[dict]) -> list[dict]:
    """Deterministically rank top 5 tools per category by distinct positive sources.

    For each of the 12 categories, tools are sorted by how many distinct
    sources positively mentioned them.  Ties are broken alphabetically.
    """
    # {category: {normalized_name: {..., sources: set()}}}
    cat_tools: dict[str, dict[str, dict]] = {c: {} for c in config.CATEGORIES}

    for m in mentions:
        if m.get("sentiment", "").lower() != "positive":
            continue

        matched_cat = _match_category(m.get("category", ""))
        if not matched_cat:
            continue

        norm = _normalize_tool_name(m.get("tool_name", ""))
        if not norm:
            continue

        bucket = cat_tools[matched_cat]
        if norm not in bucket:
            bucket[norm] = {
                "tool_name": m.get("tool_name", ""),
                "why_recommended": m.get("description", ""),
                "url": m.get("tool_url", "N/A"),
                "sources": set(),
            }

        bucket[norm]["sources"].add(m.get("source_name", "unknown"))
        if m.get("tool_url", "N/A") != "N/A":
            bucket[norm]["url"] = m["tool_url"]

    result = []
    for category in config.CATEGORIES:
        ranked = sorted(
            cat_tools[category].values(),
            key=lambda t: (-len(t["sources"]), t["tool_name"].lower()),
        )
        for rank, tool in enumerate(ranked[:5], 1):
            result.append({
                "field": category,
                "rank": rank,
                "tool_name": tool["tool_name"],
                "why_recommended": tool["why_recommended"],
                "url": tool["url"],
            })

    return result


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
