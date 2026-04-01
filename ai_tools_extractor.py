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
    # Build per-source entries with unique IDs so we can count distinct sources
    sources: list[dict] = []
    for i, art in enumerate(articles):
        sources.append({
            "source_id": f"art_{i}",
            "label": art["source"],
            "text": (
                f"Title: {art['title']}\nDate: {art['date']}\n"
                f"URL: {art['url']}\nContent:\n{art['content']}"
            ),
        })

    for i, email in enumerate(emails):
        sources.append({
            "source_id": f"email_{i}",
            "label": f"Gmail / {config.GMAIL_LABEL}",
            "text": (
                f"Subject: {email['subject']}\nDate: {email['date']}\n"
                f"Content:\n{email['body']}"
            ),
        })

    log.info("Total sources to analyse: %d articles + %d emails = %d",
             len(articles), len(emails), len(sources))

    # Build chunks of sources that fit within the LLM context window.
    # Each source is clearly delimited with its source_id so the LLM can tag
    # extracted tools back to the source they came from.
    chunks = _build_source_chunks(sources, max_chars=800_000)
    log.info("Split into %d chunk(s) for Phase 1.", len(chunks))

    # Phase 1: extract raw tool mentions from each chunk
    raw_mentions: list[str] = []
    for i, chunk in enumerate(chunks):
        log.info("LLM extraction pass %d/%d …", i + 1, len(chunks))
        raw_mentions.append(_llm_extract_tools(chunk))
        if i < len(chunks) - 1:
            time.sleep(15)  # 5 RPM limit: wait 15s between chunks

    # Phase 1.5: deterministic local counting & ranking
    log.info("Counting mentions per tool across sources locally …")
    pre_ranked = _parse_and_count_mentions(raw_mentions)
    log.info(
        "Local counting complete: %d tools found (top: %s with %d source mentions)",
        len(pre_ranked),
        pre_ranked[0]["tool_name"] if pre_ranked else "N/A",
        pre_ranked[0]["positive_mentions"] if pre_ranked else 0,
    )

    # Phase 2: LLM enrichment (description/category only) + field_tools
    log.info("LLM enrichment & categorisation pass …")
    time.sleep(15)  # wait before ranking call to respect rate limit
    final_json = _llm_rank_and_categorise(pre_ranked)

    return final_json


def _build_source_chunks(sources: list[dict], max_chars: int) -> list[str]:
    """Pack sources into chunks ≤ max_chars, each source clearly delimited."""
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for src in sources:
        entry = (
            f"=== SOURCE_ID: {src['source_id']} | {src['label']} ===\n"
            f"{src['text']}\n"
            f"=== END SOURCE_ID: {src['source_id']} ===\n"
        )
        if current_len + len(entry) > max_chars and current_parts:
            chunks.append("\n".join(current_parts))
            current_parts = []
            current_len = 0
        current_parts.append(entry)
        current_len += len(entry)

    if current_parts:
        chunks.append("\n".join(current_parts))

    return chunks


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30))
def _llm_extract_tools(text_chunk: str) -> str:
    """Ask the LLM to list AI tools mentioned in a text chunk.

    The chunk contains multiple sources delimited by SOURCE_ID markers.
    The LLM must tag each extracted tool with the source_id it came from.
    """
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(
        model_name=config.LLM_MODEL,
        system_instruction=textwrap.dedent("""\
            You are an AI-tools analyst. The input contains newsletter
            content from MULTIPLE sources, each delimited by
            === SOURCE_ID: <id> === ... === END SOURCE_ID: <id> ===

            For EVERY source, extract EVERY AI tool or product mentioned.
            Output one JSON object per tool-source pair:
            - "source_id": the SOURCE_ID the tool was found in (copy exactly)
            - "tool_name": the tool/product name (use official casing)
            - "sentiment": "positive", "neutral", or "negative"
            - "description": 1 sentence about what the tool IS
            - "use_case": 1 sentence about HOW the source says the tool
              is used or what task/problem it helps with. Quote or
              closely paraphrase the source. If no specific use case is
              described, write "General mention".
            - "source_url": the tool's URL if mentioned, else "N/A"

            CRITICAL RULES:
            - If the SAME tool appears in MULTIPLE sources, output a
              SEPARATE entry for each source — one row per source.
            - Do NOT skip sources. Process every single source in the input.
            - Do NOT deduplicate across sources.
            - Return a JSON array of objects.
        """),
        generation_config=genai.GenerationConfig(
            max_output_tokens=config.LLM_MAX_TOKENS,
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )
    response = model.generate_content(text_chunk)
    text = response.text

    # Detect truncated JSON (hit max_output_tokens before closing the array)
    stripped = text.rstrip()
    if stripped and not stripped.endswith("]"):
        log.warning(
            "Phase 1 output appears truncated (last char: %r). "
            "Consider reducing chunk size or increasing LLM_MAX_TOKENS.",
            stripped[-1] if stripped else "",
        )
        # Try to salvage: close the last complete object and the array
        last_brace = stripped.rfind("}")
        if last_brace > 0:
            text = stripped[: last_brace + 1] + "]"

    return text


def _parse_and_count_mentions(raw_mentions: list[str]) -> list[dict]:
    """Parse Phase 1 JSON outputs and deterministically count source mentions.

    "mentions" = number of distinct sources (articles / emails) in which
    a tool was mentioned with positive sentiment.  This is a real count
    that cannot be hallucinated.

    Returns a list of dicts sorted by mention count (descending):
        [{"tool_name": str, "positive_mentions": int,
          "total_mentions": int,
          "descriptions": list[str], "source_urls": list[str]}, ...]
    """
    # tool_key -> set of source_ids where it appeared positively
    tool_positive_sources: dict[str, set[str]] = {}
    # tool_key -> set of ALL source_ids where it appeared
    tool_all_sources: dict[str, set[str]] = {}
    # tool_key -> metadata
    tool_meta: dict[str, dict] = {}

    total_entries = 0

    for raw in raw_mentions:
        entries = _safe_parse_json_array(raw)
        if entries is None:
            continue

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            total_entries += 1

            name = (entry.get("tool_name") or entry.get("Tool Name") or "").strip()
            if not name:
                continue
            key = name.lower()

            source_id = (entry.get("source_id") or "unknown").strip()

            sentiment = str(
                entry.get("sentiment") or entry.get("Sentiment") or ""
            ).strip().lower()

            # Track unique sources
            tool_all_sources.setdefault(key, set()).add(source_id)
            if sentiment == "positive":
                tool_positive_sources.setdefault(key, set()).add(source_id)

            desc = (
                entry.get("description")
                or entry.get("Short description")
                or entry.get("short_description")
                or ""
            )
            url = (
                entry.get("source_url")
                or entry.get("Source URL")
                or entry.get("source_link")
                or "N/A"
            )

            use_case = (
                entry.get("use_case") or entry.get("Use Case") or ""
            ).strip()

            if key not in tool_meta:
                tool_meta[key] = {
                    "display_name": name, "descs": [], "urls": [],
                    "use_cases": [],
                }
            if desc:
                tool_meta[key]["descs"].append(desc.strip())
            if use_case and use_case.lower() != "general mention":
                tool_meta[key]["use_cases"].append(use_case)
            if url and url != "N/A":
                tool_meta[key]["urls"].append(url.strip())

    log.info("Phase 1 produced %d total extraction entries.", total_entries)

    # Build ranked list — sort by positive source count, then total source count
    ranked = []
    for key in tool_all_sources:
        pos_count = len(tool_positive_sources.get(key, set()))
        total_count = len(tool_all_sources[key])
        info = tool_meta.get(key, {})
        ranked.append({
            "tool_name": info.get("display_name", key),
            "positive_mentions": pos_count,
            "total_mentions": total_count,
            "descriptions": list(dict.fromkeys(info.get("descs", [])))[:3],
            "use_cases": list(dict.fromkeys(info.get("use_cases", []))),
            "source_urls": list(dict.fromkeys(info.get("urls", []))),
        })

    ranked.sort(key=lambda x: (x["positive_mentions"], x["total_mentions"]), reverse=True)
    return ranked


def _safe_parse_json_array(raw: str) -> list | None:
    """Best-effort parse of an LLM JSON array response."""
    text = raw.strip()
    # Strip markdown fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try to find a JSON array in the text
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    log.warning("Could not parse Phase 1 chunk as JSON array, skipping")
    return None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30))
def _llm_tools_log(pre_ranked: list[dict]) -> list:
    """Enrich pre-ranked tools with LLM-generated descriptions and categories.

    Accepts *pre_ranked*: a list of dicts already sorted by verified positive
    mention counts.  The LLM is only asked for a category and a polished
    one-sentence description — it does NOT decide the ranking or counts.
    """
    tools_summary = json.dumps(
        [
            {
                "tool_name": t["tool_name"],
                "positive_mentions": t["positive_mentions"],
                "total_mentions": t["total_mentions"],
                "sample_descriptions": t["descriptions"],
                "use_cases_from_sources": t["use_cases"],
                "source_urls": t["source_urls"],
            }
            for t in pre_ranked
        ],
        indent=2,
    )

    prompt = textwrap.dedent(f"""\
        Below is a pre-ranked list of AI tools sorted by verified positive
        mention count (descending).  The ranking and mention counts are
        already verified — do NOT change them.

        For each tool, add:
        - "category": a short category label (e.g. "LLM", "Image Generation").
          IMPORTANT: base the category on the "use_cases_from_sources" field —
          this shows how real newsletter articles described the tool being used.
          Do NOT guess from the tool name alone.
        - "description": a polished 1-sentence description based on the
          source descriptions and use cases provided.
        - "source_link": the tool's own website URL (not the newsletter).
          Use the source_urls provided when available; otherwise infer the
          official URL.

        Return a JSON array preserving the exact order and mention counts.

        OUTPUT FORMAT (valid JSON array only):
        [
          {{
            "tool_name": "...",
            "category": "...",
            "mentions": <int>,
            "description": "1 sentence.",
            "source_link": "https://..."
          }}
        ]

        PRE-RANKED TOOLS:
        {tools_summary}
    """)

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
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
    enriched = json.loads(response.text)

    # Enforce the verified counts/order in case the LLM mutated them
    tool_counts = {t["tool_name"].lower(): t["positive_mentions"] for t in pre_ranked}
    for item in enriched:
        key = item.get("tool_name", "").lower()
        if key in tool_counts:
            item["mentions"] = tool_counts[key]
    enriched.sort(key=lambda x: x.get("mentions", 0), reverse=True)

    return enriched


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30))
def _llm_field_tools(pre_ranked: list[dict]) -> list:
    """Ask the LLM for the top 5 tools per category.

    Uses the pre-ranked tool list so the LLM can only pick from tools
    that were actually found in the sources.
    """
    categories_str = "\n".join(f"- {c}" for c in config.CATEGORIES)

    tools_summary = json.dumps(
        [
            {
                "tool_name": t["tool_name"],
                "positive_mentions": t["positive_mentions"],
                "total_mentions": t["total_mentions"],
                "sample_descriptions": t["descriptions"],
                "use_cases_from_sources": t["use_cases"],
                "source_urls": t["source_urls"],
            }
            for t in pre_ranked
        ],
        indent=2,
    )

    prompt = textwrap.dedent(f"""\
        Below is a verified list of AI tools extracted from newsletter sources,
        ranked by number of source mentions.

        Each tool includes "use_cases_from_sources" — these are real quotes /
        paraphrases from newsletter articles describing HOW the tool is used.
        Use these to decide which category each tool fits into.

        For EACH of the categories below, pick the top 5 tools from this list
        that best fit the category based on their USE CASES, and rank them 1-5.

        CATEGORIES:
        {categories_str}

        VERIFIED TOOLS (only pick from these):
        {tools_summary}

        OUTPUT FORMAT (valid JSON array only, no markdown):
        [
          {{
            "field": "<category name>",
            "rank": <1-5>,
            "tool_name": "...",
            "why_recommended": "1 sentence based on actual use cases from the sources.",
            "url": "https://..."
          }}
        ]

        RULES:
        - rank 1 = best in category.
        - ONLY use tools from the VERIFIED TOOLS list above — do NOT invent tools.
        - Assign tools to categories based on their use_cases_from_sources,
          NOT based on what you generally know about the tool.
        - "why_recommended" must reference how the sources described the tool.
        - If fewer than 5 tools exist for a category, include as many as possible.
        - "url" = the tool's own website.
    """)

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
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
def _llm_rank_and_categorise(pre_ranked: list[dict]) -> dict:
    """Run two separate LLM calls for tools_log and field_tools."""
    log.info("LLM tools_log pass (top 25) …")
    tools_log = _llm_tools_log(pre_ranked[:25])
    time.sleep(15)
    log.info("LLM field_tools pass (all %d tools) …", len(pre_ranked))
    field_tools = _llm_field_tools(pre_ranked)
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
