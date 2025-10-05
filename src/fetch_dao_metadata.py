from __future__ import annotations

import os
import time
import json
from typing import Any, Dict, Optional

import requests
import pandas as pd


def _now_unix() -> int:
    return int(time.time())


def _request_with_retry(url: str, payload: Dict[str, Any], headers: Dict[str, str], max_attempts: int = 5, timeout: int = 60) -> Dict[str, Any]:
    for attempt in range(max_attempts):
        try:
            resp = requests.post(url, headers=headers,
                                 json=payload, timeout=timeout)
            # handle snapshot rate limiting
            if resp.status_code == 429:
                wait_s = min(60, 2 ** attempt * 2)
                time.sleep(wait_s)
                continue
            resp.raise_for_status()
            data = resp.json() if resp.content else {}
            if isinstance(data, dict) and data.get("errors"):
                # graphql style error container
                raise RuntimeError(str(data.get("errors")))
            return data.get("data", data) if isinstance(data, dict) else {}
        except Exception:
            if attempt == max_attempts - 1:
                raise
            wait_s = min(60, 2 ** attempt * 2)
            time.sleep(wait_s)
    return {}


def _fetch_snapshot_space(space_id: str, endpoint: Optional[str] = None) -> Dict[str, Any]:
    """fetch snapshot space metadata for a given space id"""
    url = endpoint or "https://hub.snapshot.org/graphql"
    headers = {"Content-Type": "application/json"}
    query = """
    query Space($id: String!) {
      space(id: $id) {
        id
        name
        about
        network
        symbol
        website
        twitter
        github
        avatar
        cover
        created
      }
    }
    """
    variables = {"id": str(space_id)}
    data = _request_with_retry(url=url, payload={
                               "query": query, "variables": variables}, headers=headers, timeout=60)
    space = (data.get("space") if isinstance(data, dict) else None) or {}
    # flatten to single row record
    record = {
        "source": "snapshot",
        "identifier": str(space_id),
        "fetched_at": _now_unix(),
        **{k: space.get(k) for k in [
            "id", "name", "about", "network", "symbol", "website", "twitter",
            "github", "avatar", "cover", "created"
        ]},
    }
    return record


def _fetch_tally_org(identifier: str, is_slug: Optional[bool] = None, api_key: Optional[str] = None, endpoint: Optional[str] = None) -> Dict[str, Any]:
    """fetch tally organization metadata for the given id or slug"""
    key = api_key or os.environ.get("TALLY_API_KEY")
    if not key:
        raise RuntimeError(
            "TALLY_API_KEY not set; required to query Tally API")
    url = endpoint or "https://api.tally.xyz/query"
    headers = {"Api-Key": key, "Content-Type": "application/json"}

    # infer slug versus id when not specified
    use_slug = is_slug if is_slug is not None else (
        not str(identifier).isdigit())

    query = """
    query Organization($input: OrganizationInput!) {
      organization(input: $input) {
        id
        name
        slug
        governorIds
      }
    }
    """
    variables = {"input": ({"slug": str(identifier)}
                           if use_slug else {"id": str(identifier)})}
    data = _request_with_retry(url=url, payload={
                               "query": query, "variables": variables}, headers=headers, timeout=60)
    org = (data.get("organization") if isinstance(data, dict) else None) or {}
    record = {
        "source": "tally",
        "identifier": str(identifier),
        "identifier_type": "slug" if use_slug else "id",
        "fetched_at": _now_unix(),
        **{k: org.get(k) for k in ["id", "name", "slug"]},
        "governor_ids": json.dumps(org.get("governorIds") or []),
    }
    return record


def export_metadata(source: str, identifier: str, output_path: str, is_tally_slug: Optional[bool] = None) -> Dict[str, Any]:
    """fetch dao metadata from snapshot or tally and write parquet"""
    src_norm = (source or "").strip().lower()
    if src_norm not in {"snapshot", "tally"}:
        raise ValueError("source must be 'snapshot' or 'tally'")

    if src_norm == "snapshot":
        record = _fetch_snapshot_space(identifier)
    else:
        record = _fetch_tally_org(identifier, is_slug=is_tally_slug)

    df = pd.DataFrame([record])
    # ensure directory exists
    import os as _os
    _os.makedirs(_os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    return record


__all__ = ["export_metadata"]
