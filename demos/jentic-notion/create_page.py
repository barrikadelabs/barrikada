#!/usr/bin/env python3
"""Create a Notion page via Jentic broker.

Usage:
    python demos/jentic-notion/create_page.py [title] [body]
    
Environment:
    JENTIC_API_KEY
    NOTION_PARENT_PAGE_ID or NOTION_PARENT_PAGE_URL
"""

import os
import sys

import requests

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

from utils import (
    _extract_notion_page_id,
    _normalize_notion_page_id,
    print_barrikade_headers,
    DEFAULT_JENTIC_BASE_URL,
    DEFAULT_NOTION_VERSION,
)


class Config:
    def __init__(self, api_key: str, parent_id: str, base_url: str = DEFAULT_JENTIC_BASE_URL):
        self.api_key = api_key
        self.parent_id = parent_id
        self.base_url = base_url

    @classmethod
    def from_env(cls) -> "Config":
        api_key = os.getenv("JENTIC_API_KEY", "")
        parent_id = os.getenv("NOTION_PARENT_PAGE_ID", "")
        parent_url = os.getenv("NOTION_PARENT_PAGE_URL", "")
        if not parent_id and parent_url:
            parent_id = _extract_notion_page_id(parent_url) or ""
        base_url = os.getenv("JENTIC_BASE_URL", DEFAULT_JENTIC_BASE_URL)
        return cls(api_key, parent_id, base_url)


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

    def create_page(self, parent_id: str, title: str, body_text: str) -> requests.Response:
        headers = dict(self._headers)
        headers["Notion-Version"] = DEFAULT_NOTION_VERSION
        payload = {
            "parent": {"page_id": parent_id},
            "properties": {
                "title": {
                    "title": [{"text": {"content": title}}]
                }
            },
            "children": [
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": body_text}}]
                    },
                }
            ],
        }
        return requests.post(
            self._url("/api.notion.com/v1/pages"),
            headers=headers,
            json=payload,
            timeout=20,
        )


DEFAULT_TITLE = "Barrikade x Jentic Demo Test"
DEFAULT_BODY = "This is a test page with benign content."


def main() -> int:
    if load_dotenv is not None:
        load_dotenv()

    title = os.getenv("NOTION_PAGE_TITLE", DEFAULT_TITLE)
    body_text = os.getenv("NOTION_PAGE_BODY", DEFAULT_BODY)

    if len(sys.argv) > 1:
        title = sys.argv[1]
    if len(sys.argv) > 2:
        body_text = sys.argv[2]

    config = Config.from_env()
    if not config.api_key:
        print("Error: JENTIC_API_KEY required")
        return 1
    if not config.parent_id:
        print("Error: NOTION_PARENT_PAGE_ID or NOTION_PARENT_PAGE_URL required")
        return 1

    client = JenticClient(config)

    print(f"\nCreating Notion page:")
    print(f"  Title: {title}")
    print(f"  Body: {body_text}")
    print("=" * 50)

    response = client.create_page(config.parent_id, title, body_text)
    print(f"\nStatus: {response.status_code}")
    print_barrikade_headers(dict(response.headers))

    if response.status_code >= 400:
        body = response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
        if isinstance(body, dict):
            print(f"Message: {body.get('message', 'N/A')}")
        else:
            print(f"Error: {body}")
        return 1

    page_body = response.json()
    page_id = page_body.get("id", "unknown")
    print(f"\nSUCCESS!")
    print(f"  Page ID: {page_id}")
    print(f"  URL: https://www.notion.so/{page_id.replace('-', '')}")
    print("\n" + "=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())