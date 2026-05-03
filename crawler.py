from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urljoin, urldefrag, urlparse

import requests
from bs4 import BeautifulSoup
from xml.etree import ElementTree as ET

from utils import clean_text


@dataclass(frozen=True)
class ScrapedPage:
    url: str
    title: str
    content: str


def _same_domain(url: str, base_netloc: str) -> bool:
    try:
        return urlparse(url).netloc.lower() == base_netloc.lower()
    except Exception:
        return False


def _normalize_url(url: str) -> str:
    u, _frag = urldefrag(url)
    return u.rstrip("/")


def _extract_meta_content(soup: BeautifulSoup) -> str:
    """Fallback for JS-rendered SPAs: extract title + meta description/keywords/og."""
    parts: list[str] = []
    if soup.title:
        t = clean_text(soup.title.get_text(strip=True))
        if t:
            parts.append(t)
    for attr, val in [("name", "description"), ("name", "keywords"), ("property", "og:description"), ("property", "og:title"), ("name", "twitter:description"), ("name", "twitter:title")]:
        tag = soup.find("meta", {attr: val})
        if tag and tag.get("content"):
            parts.append(clean_text(str(tag["content"])))
    return "\n".join(dict.fromkeys(parts)).strip()  # dedupe


def _extract_main_text(soup: BeautifulSoup) -> str:
    main = soup.find("main") or soup.find("article") or soup.find(attrs={"role": "main"})
    root = main if main else soup.body
    if not root:
        return ""
    # We'll accumulate extracted text blocks here.
    texts: list[str] = []

    # Remove scripts and styles
    for tag in root.find_all(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Extract phone numbers from tel: links (often not present as visible text).
    for a in root.find_all("a", href=True):
        href = (a.get("href") or "").strip().lower()
        if href.startswith("tel:"):
            num = clean_text(href[4:])
            if num and len(num) >= 8:
                texts.append(num)

    # Extract from paragraphs, headings, list items (many sites use divs/sections, not just p)
    for tag in root.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th"]):
        t = clean_text(tag.get_text(" ", strip=True))
        if t and len(t) >= 15:  # Lower threshold to catch headings and short blocks
            texts.append(t)
    # Fallback: if still empty, get all text from root
    if not texts:
        t = clean_text(root.get_text(separator=" ", strip=True))
        if t and len(t) >= 50:
            texts = [t]
    return "\n".join(texts).strip()


def _fetch_html_playwright(url: str, *, timeout_ms: int = 30000) -> str:
    """
    Fetch HTML after executing JavaScript.
    Required for SPA sites where requests/BeautifulSoup sees only a shell.
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        # Give client-side rendering a moment to populate the DOM.
        page.wait_for_timeout(3000)
        html = page.content()
        browser.close()
        return html


def _extract_links(soup: BeautifulSoup, *, base_url: str, base_netloc: str) -> Iterable[str]:
    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        if not href:
            continue
        if href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        abs_url = urljoin(base_url, href)
        abs_url = _normalize_url(abs_url)
        if not abs_url.startswith(("http://", "https://")):
            continue
        if _same_domain(abs_url, base_netloc):
            yield abs_url


def crawl(
    base_url: str,
    *,
    max_pages: int = 50,
    timeout_seconds: int = 15,
    user_agent: str = "Mozilla/5.0 (compatible; RAGCrawler/1.0)",
) -> list[ScrapedPage]:
    """
    Crawl internal pages only, avoiding duplicates/external domains.
    Returns a list of ScrapedPage with title + main paragraph content.
    """
    base_url = _normalize_url(base_url)
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("base_url must include scheme and domain, e.g. https://example.com")

    base_netloc = parsed.netloc
    headers = {"User-Agent": user_agent}

    visited: set[str] = set()

    # For SPA websites, internal links in HTML may be missing and body text can be empty.
    # sitemap.xml usually lists real pages that we can fetch and extract meta text from.
    sitemap_url = f"{base_url}/sitemap.xml"
    q_urls: list[str] = []
    try:
        sm = requests.get(sitemap_url, headers=headers, timeout=timeout_seconds)
        if sm.status_code == 200 and "xml" in (sm.headers.get("content-type") or "").lower():
            root = ET.fromstring(sm.text)
            ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
            q_urls = [loc.text.strip() for loc in root.findall("s:url/s:loc", ns) if loc is not None and loc.text]
    except Exception:
        q_urls = []

    q: deque[str] = deque(q_urls if q_urls else [base_url])
    out: list[ScrapedPage] = []

    while q and len(out) < max_pages:
        url = q.popleft()
        if url in visited:
            continue
        visited.add(url)

        try:
            resp = requests.get(url, headers=headers, timeout=timeout_seconds)
            resp.raise_for_status()
            ct = (resp.headers.get("content-type") or "").lower()
            if "text/html" not in ct:
                continue
        except requests.RequestException:
            continue

        try:
            soup = BeautifulSoup(resp.text, "html.parser")

            # If the response body is basically empty, it's likely a JS-rendered SPA.
            # Use Playwright to render the page and extract real visible text.
            body_text = clean_text(soup.body.get_text(" ", strip=True) if soup.body else "")
            needs_js = len(body_text) < 120
            if needs_js:
                rendered_html = _fetch_html_playwright(url)
                soup = BeautifulSoup(rendered_html, "html.parser")

            title = clean_text((soup.title.get_text(strip=True) if soup.title else "") or "")
            content = _extract_main_text(soup)
            if not content:
                # Fallback for JS-rendered SPAs (e.g. React/Vue): use meta tags
                content = _extract_meta_content(soup)
        except Exception:
            continue

        if content:
            out.append(ScrapedPage(url=url, title=title or url, content=content))

        try:
            for link in _extract_links(soup, base_url=url, base_netloc=base_netloc):
                if link not in visited:
                    q.append(link)
        except Exception:
            pass

    print(f"[crawler] pages_scraped={len(out)} visited={len(visited)} base={base_url}")
    return out
