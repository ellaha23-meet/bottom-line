#!/usr/bin/env python3
"""
AI Tools Extraction Pipeline
=============================
Scrapes AI newsletter archives and Gmail, uses an LLM to extract / rank /
categorise AI tools, and writes the results to Google Sheets.

Usage:
    1. Place your Google OAuth credentials.json in the project root.
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
# Configure Gemini API once at module level
# ---------------------------------------------------------------------------
_gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
if _gemini_api_key:
    genai.configure(api_key=_gemini_api_key)


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
    """Scrape all configured archive URLs for posts from the last 21 days.

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
            content = article_tag.get_text(separator="\n", strip=True) if article_tag else ""
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
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    return []


def _canonical_source(raw_identifier: str) -> str:
    """Map a raw domain or email address to its canonical provider name.

    Looks up config.SOURCE_MAPPING first.  Falls back to the raw identifier
    so new / unknown providers still get tracked (just without dedup).
    """
    # Direct lookup (covers archive domains and known email addresses)
    if raw_identifier in config.SOURCE_MAPPING:
        return config.SOURCE_MAPPING[raw_identifier]
    # Fallback: strip common prefixes like "www."
    stripped = raw_identifier.removeprefix("www.").lower()
    return config.SOURCE_MAPPING.get(stripped, raw_identifier)


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
    # With 30s waits between calls: ≤2 calls/min × ~100K tokens ≈ 200K TPM
    # (well under the 250K TPM limit and 10 RPM limit).
    chunks = _chunk_by_entries(entries, max_chars=150_000)

    # Phase 1: extract structured tool mentions from each chunk
    all_mentions: list[dict] = []
    for i, chunk in enumerate(chunks):
        log.info("LLM extraction pass %d/%d (%d chars) ...",
                 i + 1, len(chunks), len(chunk))
        extracted = _llm_extract_tools(chunk)
        if isinstance(extracted, list):
            all_mentions.extend(extracted)
        else:
            log.warning("Extraction pass %d returned non-list; skipping.", i + 1)
        if i < len(chunks) - 1:
            # 30s between calls: at ~100K tokens/call this keeps us under
            # 250K TPM (2 calls/min × 100K = 200K) and well under 10 RPM.
            time.sleep(30)

    log.info("Extracted %d total tool mentions across %d chunks.", len(all_mentions), len(chunks))

    # Phase 2: deterministic ranking in Python (no LLM needed)
    tools_log = _rank_tools_deterministic(all_mentions)
    log.info("Ranked %d tools deterministically by distinct-source count.", len(tools_log))

    # Phase 2b: LLM link fallback — fill "N/A" source_links via the LLM
    time.sleep(30)  # respect TPM limit before next LLM call
    log.info("LLM link fallback pass ...")
    tools_log = _llm_link_fallback(tools_log)

    # Build a fallback URL map from the enriched tools_log so that
    # _llm_field_tools can use LLM-resolved links too.
    fallback_urls: dict[str, str] = {}
    for t in tools_log:
        link = t.get("source_link", "N/A")
        if link and link != "N/A":
            fallback_urls[t["tool_name"].lower()] = link

    # Phase 3: LLM categorisation for field tools
    time.sleep(30)  # respect TPM limit before next LLM call
    log.info("LLM field_tools categorisation pass ...")
    field_tools = _llm_field_tools(all_mentions, fallback_urls=fallback_urls)

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
def _llm_extract_tools(text_chunk: str) -> list[dict]:
    """Ask the LLM to list AI tools mentioned in a text chunk.

    Returns a parsed list of dicts with keys:
    tool_name, source, sentiment, categories, description, url
    """
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
            - "url": URL of the tool if explicitly mentioned in the text,
              otherwise "N/A"

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
        return json.loads(response.text)
    except json.JSONDecodeError:
        log.warning(
            "LLM response appears truncated (JSONDecodeError). "
            "Attempting partial recovery ..."
        )
        recovered = _recover_partial_json_array(response.text)
        if recovered:
            log.warning("Recovered %d tool mentions from truncated response.", len(recovered))
            return recovered
        log.error("Could not recover any data from truncated response; retrying chunk.")
        raise  # let tenacity retry


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
        source = m.get("source", "unknown").strip()
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
        if m.get("url") and m["url"] != "N/A":
            tool_data[key]["urls"].append(m["url"])
        # Collect categories from each mention
        for cat in m.get("categories", []):
            if cat:
                tool_data[key]["categories"].append(cat)

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
        For each AI tool listed below, provide the official homepage URL.

        RULES:
        - Return ONLY a JSON object mapping each tool name to its URL string.
        - If you are NOT confident about the correct URL, use "N/A".
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
        if llm_url and llm_url != "N/A":
            tool["source_link"] = llm_url
            filled += 1

    log.info("LLM link fallback filled %d / %d missing URLs.", filled, len(missing))
    return tools


# ---------------------------------------------------------------
# 4c. LLM Categorisation for field tools
# ---------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=10, max=60))
def _llm_field_tools(
    all_mentions: list[dict],
    fallback_urls: dict[str, str] | None = None,
) -> list[dict]:
    """Assign tools to categories using the LLM.

    Builds a summary of all positively-mentioned tools (with distinct source
    counts) and asks the LLM to pick the top 5 per category.

    *fallback_urls* is an optional {tool_name_lower: url} map produced by the
    LLM link-fallback step; it supplements URLs that were missing from the
    original article/email content.
    """
    fallback_urls = fallback_urls or {}
    # Build a deduplicated summary of positive tools with source counts
    tool_info: dict[str, dict] = defaultdict(lambda: {
        "sources": set(), "descriptions": [], "use_cases": [],
        "urls": [], "categories": [], "original_name": "",
    })
    for m in all_mentions:
        name = m.get("tool_name", "").strip()
        if not name or m.get("sentiment", "").lower() != "positive":
            continue
        key = name.lower()
        tool_info[key]["sources"].add(m.get("source", ""))
        if not tool_info[key]["original_name"]:
            tool_info[key]["original_name"] = name
        if m.get("description"):
            tool_info[key]["descriptions"].append(m["description"])
        uc = m.get("use_case", "").strip()
        if uc and uc != "N/A" and uc not in tool_info[key]["use_cases"]:
            tool_info[key]["use_cases"].append(uc)
        if m.get("url") and m["url"] != "N/A":
            tool_info[key]["urls"].append(m["url"])
        for cat in m.get("categories", []):
            if cat:
                tool_info[key]["categories"].append(cat)

    # Build text list sorted by source count
    tool_lines = []
    for key, data in sorted(tool_info.items(), key=lambda x: -len(x[1]["sources"])):
        name = data["original_name"] or key
        count = len(data["sources"])
        desc = data["descriptions"][0] if data["descriptions"] else "No description"
        url = data["urls"][0] if data["urls"] else fallback_urls.get(key, "N/A")
        # Deduplicate categories
        seen_cats = dict.fromkeys(data["categories"])
        cats_str = "; ".join(seen_cats) if seen_cats else "uncategorized"
        use_cases_str = "; ".join(data["use_cases"][:5]) if data["use_cases"] else "N/A"
        tool_lines.append(
            f"- {name} | sources: {count} | url: {url} | categories: {cats_str} | {desc} | use cases: {use_cases_str}"
        )

    tools_text = "\n".join(tool_lines)
    categories_str = "\n".join(f"- {c}" for c in config.CATEGORIES)

    prompt = textwrap.dedent(f"""        Below is a list of AI tools extracted from newsletter articles and emails,
        along with how many distinct sources mentioned them positively.

        For EACH of the following 12 categories, select the top 5 tools ranked 1-5.

        CATEGORIES:
        {categories_str}

        CRITICAL RULES:
        - ONLY select tools from the TOOLS LIST below. Do NOT add any tools
          from your own knowledge.
        - Rank by how many distinct sources mentioned the tool positively
          (the "sources" count). Rank 1 = highest source count for that category.
        - If fewer than 5 tools fit a category from the list, include only
          those that fit. Do NOT invent tools to fill slots.
        - "url" MUST be copied from the TOOLS LIST below. Do NOT invent URLs.
        - "why_recommended" must be derived from the use-case evidence in the
          TOOLS LIST below, NOT from your own knowledge or opinion.

        OUTPUT FORMAT (valid JSON array only):
        [
          {{
            "field": "<category name>",
            "rank": <1-5>,
            "tool_name": "...",
            "why_recommended": "1 sentence from the sources.",
            "url": "https://..."
          }}
        ]

        TOOLS LIST:
        {tools_text}
    """)

    model = genai.GenerativeModel(
        model_name=config.LLM_MODEL,
        system_instruction=(
            "Return only a valid JSON array. No markdown. "
            "Only use tools from the provided TOOLS LIST."
        ),
        generation_config=genai.GenerationConfig(
            max_output_tokens=config.LLM_MAX_TOKENS,
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )
    response = model.generate_content(prompt)
    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        log.warning("field_tools response truncated; attempting partial recovery ...")
        recovered = _recover_partial_json_array(response.text)
        if recovered:
            log.warning("Recovered %d field_tools entries from truncated response.", len(recovered))
            return recovered
        log.error("Could not recover field_tools data; returning empty list.")
        return []


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

    if not _gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

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
