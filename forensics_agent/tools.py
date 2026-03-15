from __future__ import annotations

import os
from typing import Any, Dict, List
import dotenv

dotenv.load_dotenv()


USER_AGENT = "stego-forensics-agent/0.1"


def search_arxiv(query: str, max_results: int = 4) -> List[Dict[str, Any]]:
    """Search arXiv for academic papers related to the given query.

    Returns a list of dicts with keys: title, summary, url.
    """
    import requests

    url = "http://export.arxiv.org/api/query"
    params = {"search_query": f"all:{query}", "start": 0, "max_results": max_results}
    resp = requests.get(
        url, params=params, headers={"User-Agent": USER_AGENT}, timeout=20
    )
    resp.raise_for_status()
    entries = []
    for chunk in resp.text.split("<entry>")[1:]:
        title = (
            chunk.split("<title>")[1].split("</title>")[0].strip().replace("\n", " ")
        )
        summary = (
            chunk.split("<summary>")[1]
            .split("</summary>")[0]
            .strip()
            .replace("\n", " ")
        )
        link = chunk.split("<id>")[1].split("</id>")[0].strip()
        entries.append({"title": title, "summary": summary[:600], "url": link})
    return entries


def search_cve(query: str, max_results: int = 4) -> List[Dict[str, Any]]:
    """Search for CVEs and threat-intelligence reports using the Tavily API.

    Intended exclusively for CVE lookups related to the detected payload family.
    Requires the TAVILY_API_KEY environment variable to be set.

    Returns a list of dicts with keys: title, summary, url.
    """
    import requests

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY environment variable is not set. "
            "Export it before running: export TAVILY_API_KEY=tvly-..."
        )

    resp = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": query,
            "search_depth": "advanced",
            "max_results": max_results,
        },
        timeout=30,
    )
    resp.raise_for_status()
    results = []
    for item in resp.json().get("results", []):
        results.append(
            {
                "title": item.get("title", ""),
                "summary": item.get("content", "")[:600],
                "url": item.get("url", ""),
            }
        )
    return results
