#!/usr/bin/env python3
"""
AI Tools Extraction Pipeline
=============================
Scrapes AI newsletter archives and Gmail, uses an LLM to extract AI tools
from each source independently, deterministically counts positive mentions
across sources, and uses an LLM to analyse and categorise the ranked results.

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


# ===================================================================
# 1. Google Authentication
# ===================================================================
def get_google_credentials() -> Credentials:
    """Authenticate with Google using OAuth2 and return credentials."""
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
def _extract_sender_email(from_header: str) -> str:
    """Extract the bare email address from a From header value."""
    match = re.search(r"<([^>]+)>", from_header)
    if match:
        return match.group(1).lower().strip()
    return from_header.lower().strip()


def _map_sender_to_source_id(sender: str) -> str:
    """Map a sender email address to a known source_id.

    Checks exact match first, then falls back to domain-based matching.
    Returns the sender address itself if no known provider matches.
    """
    if sender in config.EMAIL_PROVIDERS:
        return config.EMAIL_PROVIDERS[sender]

    # Fuzzy match on domain parts
    for known_addr, source_id in config.EMAIL_PROVIDERS.items():
        known_domain = known_addr.split("@")[-1]
        sender_domain = sender.split("@")[-1]
        if known_domain == sender_domain:
            return source_id

    return f"email_{sender}"


def fetch_emails(creds: Credentials) -> list[dict]:
    """Fetch emails from Gmail with the configured newsletter label.

    Returns a list of dicts with keys: subject, date, body, source_id, sender.
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
        sender = _extract_sender_email(headers.get("From", ""))
        source_id = _map_sender_to_source_id(sender)
        results.append({
            "subject": headers.get("Subject", ""),
            "date": headers.get("Date", ""),
            "body": body_text,
            "sender": sender,
            "source_id": source_id,
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
def _archive_url_to_source_id(archive_url: str) -> str:
    """Derive a source_id from an archive URL (e.g. 'superhuman.ai')."""
    return archive_url.split("/")[2].replace("www.", "")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _fetch_page(url: str, page) -> str:
    """Navigate to a URL with Playwright and return the fully-rendered HTML."""
    page.goto(url, wait_until="domcontentloaded", timeout=15_000)
    return page.content()


def scrape_archives() -> list[dict]:
    """Scrape all configured archive URLs for posts from the lookback window.

    Returns a list of dicts with keys: source, source_id, title, date, url, content.
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
    source_id = _archive_url_to_source_id(archive_url)

    # Navigate to the archive page
    try:
        page.goto(archive_url, wait_until="networkidle", timeout=config.REQUEST_TIMEOUT * 1000)
    except PlaywrightTimeoutError:
        page.goto(archive_url, wait_until="domcontentloaded", timeout=config.REQUEST_TIMEOUT * 1000)

    # Scroll down repeatedly to trigger infinite scroll and load all articles
    prev_height = 0
    for _ in range(30):
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(2000)
        new_height = page.evaluate("document.body.scrollHeight")
        if new_height == prev_height:
            break
        prev_height = new_height

    html = page.content()

    # Save rendered HTML for debugging
    debug_file = f"debug_{archive_url.split('/')[2]}.html"
    with open(debug_file, "w", encoding="utf-8") as f:
        f.write(html)
    log.info("  -> saved rendered HTML to %s", debug_file)

    soup = BeautifulSoup(html, "html.parser")

    link_candidates = _find_article_links(soup, archive_url)
    log.info("  -> found %d link candidates on archive page", len(link_candidates))
    for title, url, date_str in link_candidates[:5]:
        log.info("     sample: [%s] %s (%s)", date_str, title[:60], url[:80])

    articles: list[dict] = []
    for title, url, date_str in link_candidates[: config.MAX_ARTICLES_PER_SOURCE]:
        pub_date = _parse_date_safe(date_str)
        if pub_date and pub_date < cutoff:
            continue

        try:
            page_html = _fetch_page(url, page)
            page_soup = BeautifulSoup(page_html, "html.parser")
            article_tag = (
                page_soup.find("article")
                or page_soup.find("main")
                or page_soup.find("body")
            )
            content = article_tag.get_text(separator="\n", strip=True) if article_tag else ""
        except Exception:
            log.warning("Could not fetch article: %s", url)
            content = title

        articles.append({
            "source": archive_url,
            "source_id": source_id,
            "title": title,
            "date": date_str,
            "url": url,
            "content": content[:8000],
        })

    return articles


def _find_article_links(soup: BeautifulSoup, archive_url: str) -> list[tuple[str, str, str]]:
    """Heuristically extract (title, url, date_string) tuples from an archive page."""
    base = "/".join(archive_url.split("/")[:3])
    results: list[tuple[str, str, str]] = []

    # Strategy 1: Substack-style archives
    for a_tag in soup.select("a[data-post-id], a.post-preview-title, a.post-preview"):
        href = a_tag.get("href", "")
        if not href.startswith("http"):
            href = base + href
        title = a_tag.get_text(strip=True)
        date_str = _find_nearby_date(a_tag)
        if title:
            results.append((title, href, date_str))

    # Strategy 2: Generic — /p/ /post/ /newsletter/ /i/ links
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

    # Strategy 3: Broad fallback
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
# 4. LLM-Based Per-Source Tool Extraction
# ===================================================================
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30))
def _llm_extract_tools_from_source(text: str, source_id: str, source_url: str) -> list[dict]:
    """Ask the LLM to extract AI tools from a single source's content.

    Returns a list of dicts with keys:
        source_id, tool_name, sentiment, description, use_case, source_url
    """
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(
        model_name=config.LLM_MODEL,
        system_instruction=textwrap.dedent("""\
            You are an AI-tools analyst. Given a single newsletter article or
            email, extract every distinct AI tool mentioned. For each tool return
            a JSON object with these exact keys:
            - "tool_name": the canonical name of the tool
            - "sentiment": one of "positive", "neutral", or "negative"
            - "description": 1-sentence description of the tool
            - "use_case": 1-sentence description of the primary use-case mentioned

            Return ONLY a valid JSON array. No markdown fences, no commentary.
            If no AI tools are mentioned, return an empty array [].
        """),
        generation_config=genai.GenerationConfig(
            max_output_tokens=config.LLM_MAX_TOKENS,
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )
    response = model.generate_content(text[:30000])  # cap per-source input
    raw_tools = json.loads(response.text)

    # Attach source metadata to each extracted tool
    for tool in raw_tools:
        tool["source_id"] = source_id
        tool["source_url"] = source_url

    return raw_tools


# ===================================================================
# 5. Deterministic Counting & Ranking
# ===================================================================
def _normalize_tool_name(name: str) -> str:
    """Normalize a tool name for deduplication (lowercase, strip whitespace)."""
    return re.sub(r"\s+", " ", name.strip().lower())


def analyze_content(articles: list[dict], emails: list[dict]) -> dict:
    """Extract tools per-source, deterministically count cross-source mentions,
    then send ranked data to the LLM for categorisation.

    Returns a dict with keys ``"tools_log"`` and ``"field_tools"``.
    """
    all_mentions: list[dict] = []

    # --- Process web archive articles ---
    total_web = len(articles)
    for i, art in enumerate(articles):
        log.info("LLM extraction: web article %d/%d [%s] %s",
                 i + 1, total_web, art["source_id"], art["title"][:50])
        text = (
            f"Title: {art['title']}\nDate: {art['date']}\n"
            f"URL: {art['url']}\nContent:\n{art['content']}"
        )
        try:
            tools = _llm_extract_tools_from_source(
                text, source_id=art["source_id"], source_url=art["url"]
            )
            all_mentions.extend(tools)
        except Exception:
            log.exception("Failed to extract tools from article: %s", art["url"])
        time.sleep(2)  # rate-limit spacing

    # --- Process emails (group by source_id, process each email) ---
    total_email = len(emails)
    for i, email in enumerate(emails):
        log.info("LLM extraction: email %d/%d [%s] %s",
                 i + 1, total_email, email["source_id"], email["subject"][:50])
        text = (
            f"Subject: {email['subject']}\nDate: {email['date']}\n"
            f"Sender: {email['sender']}\nContent:\n{email['body']}"
        )
        try:
            tools = _llm_extract_tools_from_source(
                text, source_id=email["source_id"], source_url=f"email://{email['sender']}"
            )
            all_mentions.extend(tools)
        except Exception:
            log.exception("Failed to extract tools from email: %s", email["subject"])
        time.sleep(2)

    log.info("Total raw tool mentions extracted: %d", len(all_mentions))

    # --- Deterministic counting ---
    ranked_tools, mention_details = _deterministic_rank(all_mentions)

    log.info("Unique tools found: %d", len(ranked_tools))
    for t in ranked_tools[:10]:
        log.info("  %s: %d positive source(s)", t["tool_name"], t["positive_source_count"])

    # --- LLM categorisation using the deterministic ranked data ---
    log.info("Sending ranked data to LLM for analysis & categorisation …")
    time.sleep(2)
    final = _llm_analyze_ranked(ranked_tools, mention_details)

    return final


def _deterministic_rank(all_mentions: list[dict]) -> tuple[list[dict], list[dict]]:
    """Deterministically count positive mentions per tool across unique sources.

    Returns:
        ranked_tools: list of dicts sorted by positive_source_count descending.
            Each dict has: tool_name, positive_source_count, sources, description, use_case
        mention_details: the full list of per-source mentions (the raw JSON).
    """
    # Group mentions by normalized tool name
    tool_data: dict[str, dict] = defaultdict(lambda: {
        "tool_name": "",
        "positive_sources": set(),
        "all_sources": set(),
        "descriptions": [],
        "use_cases": [],
        "source_urls": [],
        "sentiments": [],
    })

    for mention in all_mentions:
        name = mention.get("tool_name", "").strip()
        if not name:
            continue
        key = _normalize_tool_name(name)
        entry = tool_data[key]

        # Keep the most common casing (use first seen)
        if not entry["tool_name"]:
            entry["tool_name"] = name

        source_id = mention.get("source_id", "unknown")
        sentiment = mention.get("sentiment", "neutral").lower().strip()

        entry["all_sources"].add(source_id)
        entry["sentiments"].append(sentiment)

        if sentiment == "positive":
            entry["positive_sources"].add(source_id)

        if mention.get("description"):
            entry["descriptions"].append(mention["description"])
        if mention.get("use_case"):
            entry["use_cases"].append(mention["use_case"])
        if mention.get("source_url"):
            entry["source_urls"].append(mention["source_url"])

    # Build ranked list — include ALL extracted descriptions and use_cases
    # so the LLM categorisation is driven by actual source content
    ranked = []
    for key, entry in tool_data.items():
        # Deduplicate descriptions and use_cases while preserving order
        unique_descriptions = list(dict.fromkeys(d for d in entry["descriptions"] if d))
        unique_use_cases = list(dict.fromkeys(u for u in entry["use_cases"] if u))

        ranked.append({
            "tool_name": entry["tool_name"],
            "positive_source_count": len(entry["positive_sources"]),
            "total_source_count": len(entry["all_sources"]),
            "positive_sources": sorted(entry["positive_sources"]),
            "all_sources": sorted(entry["all_sources"]),
            "descriptions": unique_descriptions,
            "use_cases": unique_use_cases,
            "sample_source_urls": list(dict.fromkeys(entry["source_urls"]))[:5],
        })

    # Sort by positive_source_count descending, then by total_source_count
    ranked.sort(key=lambda x: (-x["positive_source_count"], -x["total_source_count"]))

    return ranked, all_mentions


# ===================================================================
# 6. LLM Analysis & Categorisation (using deterministic ranked data)
# ===================================================================
def _llm_analyze_ranked(ranked_tools: list[dict], mention_details: list[dict]) -> dict:
    """Send the deterministically-ranked tools to the LLM for categorisation.

    The LLM receives the ranked data (including ALL extracted descriptions and
    use_cases from source content) and produces:
    1. tools_log: top 25 tools with category and analysis
    2. field_tools: top 5 tools per category
    """
    # Prepare ranked summary for the LLM — include all extracted content
    # so categorisation is driven by what sources actually said
    ranked_summary = []
    for t in ranked_tools[:50]:  # send top 50 to LLM for analysis
        ranked_summary.append({
            "tool_name": t["tool_name"],
            "positive_mentions_across_sources": t["positive_source_count"],
            "total_sources_mentioned": t["total_source_count"],
            "sources": t["positive_sources"],
            "descriptions_from_sources": t["descriptions"],
            "use_cases_from_sources": t["use_cases"],
        })

    # --- Call 1: Tools Log ---
    log.info("LLM tools_log categorisation pass …")
    tools_log = _llm_tools_log_from_ranked(ranked_summary)
    time.sleep(15)

    # --- Call 2: Field Tools ---
    log.info("LLM field_tools categorisation pass …")
    field_tools = _llm_field_tools_from_ranked(ranked_summary)

    return {"tools_log": tools_log, "field_tools": field_tools}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30))
def _llm_tools_log_from_ranked(ranked_summary: list[dict]) -> list:
    """Ask the LLM to categorise and describe the top-ranked tools."""
    categories_str = "\n".join(f"- {c}" for c in config.CATEGORIES)

    prompt = textwrap.dedent(f"""\
        Below are AI tools ranked by how many independent sources mentioned
        them positively. The "positive_mentions_across_sources" count is
        DETERMINISTIC and MUST be preserved exactly in your output as the
        "mentions" field.

        Each tool includes "descriptions_from_sources" and "use_cases_from_sources"
        — these are the actual descriptions and use-cases extracted from the
        newsletter content. You MUST base your categorisation and description
        on this extracted content, NOT on your own knowledge of the tool.

        Your job is to:
        1. Read the extracted descriptions and use-cases for each tool.
        2. Based on that content, assign the tool to the single best-fitting
           category from the list below.
        3. Synthesise the extracted descriptions into 1 sentence.
        4. Provide the tool's own website URL.

        CATEGORIES:
        {categories_str}

        OUTPUT FORMAT (valid JSON array only, no markdown):
        [
          {{
            "tool_name": "...",
            "category": "<one of the categories above>",
            "mentions": <int — MUST equal positive_mentions_across_sources>,
            "description": "1 sentence synthesised from extracted descriptions.",
            "source_link": "https://tool-website.com"
          }}
        ]

        RULES:
        - Return the top 25 tools (or fewer if fewer exist).
        - "mentions" MUST exactly match the "positive_mentions_across_sources"
          value from the input. Do NOT re-count or estimate.
        - Keep tools sorted by mentions descending.
        - "category" MUST be chosen based on the extracted use_cases_from_sources
          and descriptions_from_sources, NOT your general knowledge.
        - "description" MUST summarise what the sources said, not your own knowledge.
        - "source_link" = the tool's own website, NOT the newsletter URL.

        RANKED TOOLS DATA:
        {json.dumps(ranked_summary, indent=2)}
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
def _llm_field_tools_from_ranked(ranked_summary: list[dict]) -> list:
    """Ask the LLM for the top 5 tools per category using ranked data."""
    categories_str = "\n".join(f"- {c}" for c in config.CATEGORIES)

    prompt = textwrap.dedent(f"""\
        Below are AI tools ranked by how many independent sources mentioned
        them positively. Each tool includes "descriptions_from_sources" and
        "use_cases_from_sources" — these are the actual descriptions and
        use-cases extracted from the newsletter content.

        Your job: for EACH of the 12 categories below, select the top 5 tools
        whose EXTRACTED use-cases and descriptions best fit that category.

        IMPORTANT: Assign tools to categories based ONLY on the extracted
        "use_cases_from_sources" and "descriptions_from_sources" content.
        Do NOT rely on your general knowledge of what a tool can do — only
        what the newsletter sources actually described.

        CATEGORIES:
        {categories_str}

        OUTPUT FORMAT (valid JSON array only, no markdown):
        [
          {{
            "field": "<category name>",
            "rank": <1-5>,
            "tool_name": "...",
            "why_recommended": "1 sentence based on extracted use-cases.",
            "url": "https://tool-website.com"
          }}
        ]

        RULES:
        - rank 1 = best in category (most relevant extracted use-cases +
          highest positive_mentions_across_sources count).
        - If fewer than 5 tools fit a category based on extracted content,
          include only those that actually fit.
        - "url" = the tool's own website.
        - "why_recommended" = 1 sentence summarising the extracted use-case,
          NOT your own knowledge.
        - Prefer tools with higher positive_mentions_across_sources counts
          when relevance is similar.

        RANKED TOOLS DATA:
        {json.dumps(ranked_summary, indent=2)}
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


# ===================================================================
# 7. Google Sheets Output
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
# 8. Main Orchestrator
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
        "Collected %d articles and %d emails. Starting per-source analysis …",
        len(articles), len(emails),
    )

    # Step 3: Per-source LLM extraction + deterministic ranking + LLM categorisation
    data = analyze_content(articles, emails)

    # Step 4: Write results to Google Sheets
    log.info("Writing results to Google Sheets …")
    write_to_sheets(creds, data)

    log.info("=== Pipeline complete ===")


if __name__ == "__main__":
    main()
