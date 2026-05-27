"""Web search tool using DuckDuckGo."""
from typing import Any


def search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web and return list of {title, url, snippet}."""
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        return results
    except ImportError:
        return _fallback_search(query, max_results)
    except Exception as e:
        return [{"title": "Search error", "url": "", "snippet": str(e)}]


def _fallback_search(query: str, max_results: int) -> list[dict]:
    """Minimal fallback using DuckDuckGo HTML (no JS needed)."""
    import requests
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; NexusAgent/1.0)"}
        r = requests.get(
            "https://duckduckgo.com/html/",
            params={"q": query},
            headers=headers,
            timeout=10,
        )
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.text, "html.parser")
        results = []
        for result in soup.select(".result")[:max_results]:
            title_el = result.select_one(".result__title")
            url_el = result.select_one(".result__url")
            snippet_el = result.select_one(".result__snippet")
            results.append({
                "title": title_el.get_text(strip=True) if title_el else "",
                "url": url_el.get_text(strip=True) if url_el else "",
                "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
            })
        return results
    except Exception as e:
        return [{"title": "Search unavailable", "url": "", "snippet": str(e)}]
