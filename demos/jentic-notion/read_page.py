#!/usr/bin/env python3
"""Read a Notion page via Jentic broker.

Usage:
    python demos/jentic-notion/read_page.py [page_id]
    
Environment:
    JENTIC_API_KEY
    JENTIC_PAGE_ID (optional, if not provided as arg)
"""

import os
import sys

import requests

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

from utils import (
    _normalize_notion_page_id,
    print_barrikade_headers,
    DEFAULT_JENTIC_BASE_URL,
    DEFAULT_NOTION_VERSION,
)

class Config:
    def __init__(self, api_key: str, base_url: str = DEFAULT_JENTIC_BASE_URL):
        self.api_key = api_key
        self.base_url = base_url

    @classmethod
    def from_env(cls) -> "Config":
        api_key = os.getenv("JENTIC_API_KEY", "")
        base_url = os.getenv("JENTIC_BASE_URL", DEFAULT_JENTIC_BASE_URL)
        return cls(api_key, base_url)


class JenticClient:
    def __init__(self, config: Config):
        self.config = config

    @property
    def _headers(self):
        return {
            "X-Jentic-API-Key": self.config.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _url(self, path: str) -> str:
        return f"{self.config.base_url.rstrip('/')}{path}"

    def get_page(self, page_id: str) -> requests.Response:
        headers = dict(self._headers)
        headers["Notion-Version"] = DEFAULT_NOTION_VERSION
        normalized = _normalize_notion_page_id(page_id)
        return requests.get(
            self._url(f"/api.notion.com/v1/pages/{normalized}"),
            headers=headers,
            timeout=20,
        )

    def get_page_blocks(self, block_id: str) -> requests.Response:
        headers = dict(self._headers)
        headers["Notion-Version"] = DEFAULT_NOTION_VERSION
        normalized = _normalize_notion_page_id(block_id)
        return requests.get(
            self._url(f"/api.notion.com/v1/blocks/{normalized}/children"),
            headers=headers,
            timeout=20,
        )


def main() -> int:
    if load_dotenv is not None:
        load_dotenv()

    if len(sys.argv) < 2 and not os.getenv("JENTIC_PAGE_ID"):
        print("Usage: python demos/jentic-notion/read_page.py [page_id]")
        print("Or set JENTIC_PAGE_ID environment variable")
        return 1

    page_id = sys.argv[1] if len(sys.argv) > 1 else os.getenv("JENTIC_PAGE_ID", "")

    config = Config.from_env()
    if not config.api_key:
        print("Error: JENTIC_API_KEY required")
        return 1

    client = JenticClient(config)

    print(f"\nReading Notion page: {page_id}")
    print("=" * 50)

    page_response = client.get_page(page_id)
    print(f"\nPage status: {page_response.status_code}")
    print_barrikade_headers(dict(page_response.headers))

    if page_response.status_code >= 400:
        print(f"FAILED: {page_response.status_code}")
        return 1

    page_body = page_response.json()
    title = "Untitled"
    properties = page_body.get("properties", {})
    if "title" in properties:
        title_arr = properties["title"].get("title", [])
        if title_arr:
            title = title_arr[0].get("plain_text", "Untitled")
    print(f"  Title: {title}")

    print(f"\nFetching blocks/children...")
    blocks_response = client.get_page_blocks(page_id)
    print(f"Blocks status: {blocks_response.status_code}")
    
    # Show ALL headers for debugging
    print("\n  All response headers:")
    for k, v in blocks_response.headers.items():
        print(f"    {k}: {v}")
    
    print_barrikade_headers(dict(blocks_response.headers))

    if blocks_response.status_code >= 400:
        print(f"BLOCKED: {blocks_response.status_code}")
        return 1

    blocks_body = blocks_response.json()
    results = blocks_body.get("results", [])
    print(f"  Found {len(results)} blocks")

    for idx, block in enumerate(results, 1):
        block_type = block.get("type", "unknown")
        content = ""
        if block_type == "paragraph":
            rich_text = block.get("paragraph", {}).get("rich_text", [])
            if rich_text:
                content = rich_text[0].get("text", {}).get("content", "")
        elif block_type in ("heading_1", "heading_2", "heading_3"):
            rich_text = block.get(block_type, {}).get("rich_text", [])
            if rich_text:
                content = rich_text[0].get("text", {}).get("content", "")
        if content:
            print(f"  Block {idx} ({block_type}): {content[:60]}")

    print("\n" + "=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())