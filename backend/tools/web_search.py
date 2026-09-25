import os
from typing import List, Dict, Any
from tavily import TavilyClient
from backend.config import TAVILY_API_KEY


class WebSearchTool:
    """Performs real-time web searches for updated educational news and exam dates."""

    def __init__(self):
        self.client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

    def search_educational_web(self, query: str, max_results: int = 4) -> List[Dict[str, Any]]:
        """Queries the web and formats search results."""
        if not self.client:
            return [{"title": "Search Unavailable", "snippet": "TAVILY_API_KEY missing."}]

        response = self.client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=True
        )

        results = []
        for res in response.get("results", []):
            results.append({
                "title": res.get("title"),
                "url": res.get("url"),
                "snippet": res.get("content")
            })

        return results


web_search_tool = WebSearchTool()