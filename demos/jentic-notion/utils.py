import re
from urllib.parse import urlparse

DEFAULT_JENTIC_BASE_URL = "http://localhost:8900"
DEFAULT_NOTION_VERSION = "2022-06-28"

_NOTION_ID_RE = re.compile(r"([0-9a-fA-F]{32}|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})")


def _normalize_notion_page_id(raw: str) -> str | None:
    candidate = raw.strip()
    if len(candidate) == 32 and all(ch in "0123456789abcdefABCDEF" for ch in candidate):
        return (
            f"{candidate[0:8]}-"
            f"{candidate[8:12]}-"
            f"{candidate[12:16]}-"
            f"{candidate[16:20]}-"
            f"{candidate[20:32]}"
        ).lower()
    if len(candidate) == 36 and _NOTION_ID_RE.fullmatch(candidate):
        return candidate.lower()
    return None


def _extract_notion_page_id(url_or_id: str) -> str | None:
    normalized = _normalize_notion_page_id(url_or_id)
    if normalized:
        return normalized
    parsed = urlparse(url_or_id)
    if not parsed.scheme or not parsed.netloc:
        return None
    path = parsed.path or ""
    match = _NOTION_ID_RE.search(path)
    if not match:
        return None
    return _normalize_notion_page_id(match.group(1))


def print_barrikade_headers(headers: dict, prefix: str = "  ") -> None:
    barrikade_headers = [
        "X-Barrikade-Checked",
        "X-Barrikade-Verdict",
        "X-Barrikade-Bypass",
        "X-Barrikade-Egress-Checked",
        "X-Barrikade-Egress-Verdict",
        "X-Barrikade-Egress-Bypass",
    ]
    for key in barrikade_headers:
        value = headers.get(key)
        if value is not None:
            print(f"{prefix}{key}: {value}")