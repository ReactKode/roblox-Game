"""Web scraping tool - fetches and cleans page content."""
import requests


def scrape(url: str, max_chars: int = 6000) -> str:
    """Fetch URL and return cleaned text content."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; NexusAgent/1.0)"}
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        return _clean_html(r.text, max_chars)
    except requests.exceptions.Timeout:
        return f"Error: Request to {url} timed out."
    except requests.exceptions.HTTPError as e:
        return f"Error: HTTP {e.response.status_code} for {url}"
    except Exception as e:
        return f"Error scraping {url}: {e}"


def _clean_html(html: str, max_chars: int) -> str:
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        lines = [line for line in text.splitlines() if len(line.strip()) > 2]
        result = "\n".join(lines)
        return result[:max_chars]
    except ImportError:
        # Minimal fallback without BS4
        import re
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]
