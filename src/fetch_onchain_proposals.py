"""
Tally GraphQL API client for fetching on-chain proposal data.

This module provides functionality to interact with Tally's public GraphQL API
to fetch proposal data, votes, and related governance information.

Usage:
    from src.fetch_onchain_proposals import TallyClient

    client = TallyClient(api_key="your_api_key")
    proposals = client.get_proposals(filters=ProposalFilters(governor_id="eip155:1:0x1234...abcd"))

Intended output locations for downstream scripts: `data/<dao_slug>/governance_data/onchain_*`.
"""

import requests
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv
import logging
import time
import backoff

# Module-level logger
logger = logging.getLogger(__name__)

# Load environment from detected project root reliably (prefer folder with requirements.txt)
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR
for _cand in (_THIS_DIR, *_THIS_DIR.parents):
    if (_cand / "requirements.txt").exists():
        _PROJECT_ROOT = _cand
        break
for _candidate in (_PROJECT_ROOT / ".env", _PROJECT_ROOT / "env"):
    if _candidate.exists():
        load_dotenv(dotenv_path=_candidate, override=False)


class ProposalStatus(Enum):
    """Enum for proposal status values."""
    ACTIVE = "active"
    QUEUED = "queued"
    SUCCEEDED = "succeeded"
    EXECUTED = "executed"
    DEFEATED = "defeated"
    VETOED = "vetoed"
    CROSSCHAINEXECUTED = "crosschainexecuted"
    VETOVOTEOPEN = "vetoVoteOpen"


@dataclass
class ProposalFilters:
    """Filters for proposal queries."""
    governor_id: Optional[str] = None
    organization_id: Optional[str] = None
    status: Optional[ProposalStatus] = None
    proposer: Optional[str] = None
    archived: Optional[bool] = None
    draft: Optional[bool] = None


@dataclass
class PaginationInput:
    """Pagination parameters for queries."""
    limit: int = 20
    after_cursor: Optional[str] = None


@dataclass
class SortInput:
    """Sort parameters for queries."""
    sort_by: str = "id"
    is_descending: bool = True


class TallyClient:
    """Client for interacting with Tally's GraphQL API."""

    BASE_URL = "https://api.tally.xyz/query"
    MAX_BACKOFF_TIME = 120  # seconds
    REQS_PER_MIN = 30       # default client-side rate limit (tunable)

    def __init__(self, api_key: str, timeout: int = 30, reqs_per_min: int | None = None):
        """
        Initialize the Tally client.

        Args:
            api_key: Your Tally API key from User Settings
            timeout: Request timeout in seconds
            reqs_per_min: Optional per-minute request limit (client-side)
        """
        self.api_key = api_key
        self.timeout = timeout
        self.headers = {"Api-Key": api_key, "Content-Type": "application/json"}
        self._last_request_time = 0.0
        # Allow tuning via env var TALLY_REQS_PER_MIN when parameter is not provided
        if reqs_per_min is None:
            try:
                _env_rpm = int(os.environ.get("TALLY_REQS_PER_MIN", "0"))
            except Exception:
                _env_rpm = 0
            reqs_per_min = _env_rpm if _env_rpm > 0 else None

        self.reqs_per_min = reqs_per_min if reqs_per_min and reqs_per_min > 0 else self.REQS_PER_MIN
        if reqs_per_min and reqs_per_min > 0:
            logger.info(
                f"TallyClient request rate set to {self.reqs_per_min} reqs/min")

    @backoff.on_exception(
        backoff.expo,
        (requests.HTTPError, requests.RequestException),
        max_time=MAX_BACKOFF_TIME,
        giveup=lambda e: hasattr(
            e, 'response') and e.response is not None and e.response.status_code not in (429, 502, 503, 504),
        jitter=backoff.full_jitter,
    )
    def _make_request(self, query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a GraphQL request to the Tally API.

        Args:
            query: GraphQL query string
            variables: Query variables

        Returns:
            JSON response from the API

        Raises:
            requests.RequestException: If the request fails
        """
        # simple client-side rate limiting
        elapsed = time.time() - self._last_request_time
        min_interval = 60.0 / float(self.reqs_per_min)
        if elapsed < min_interval:
            wait_s = max(0.0, min_interval - elapsed)
            if wait_s >= 0.001:
                logger.debug(
                    f"Client rate limit: sleeping for {wait_s:.3f}s (rpm={self.reqs_per_min})")
            time.sleep(wait_s)

        payload = {"query": query, "variables": variables}

        response = requests.post(
            self.BASE_URL,
            headers=self.headers,
            json=payload,
            timeout=self.timeout,
        )
        self._last_request_time = time.time()
        if response.status_code != 200:
            try:
                body = response.text
            except Exception:
                body = "<unreadable body>"
            logger.error(
                f"[TALLY HTTP ERROR] status={response.status_code} body={body[:800]}")
        try:
            response.raise_for_status()
        except requests.RequestException as e:
            # Already printed the body above; re-raise
            raise e

        try:
            data = response.json()
        except Exception:
            logger.error(
                f"[TALLY JSON ERROR] Non-JSON response: {response.text[:800]}")
            raise

        # Log GraphQL errors if present
        if isinstance(data, dict) and data.get("errors"):
            from json import dumps as _dumps
            logger.error(
                f"[TALLY GQL ERRORS] { _dumps(data['errors'])[:1000] }")

        return data

    def get_proposal_by_id(self, proposal_id: str) -> Dict[str, Any]:
        """Fetch a single proposal by its Tally ID."""
        query = """
        query GetProposal($id: ID!) {
          proposal(id: $id) {
            id
            onchainId
            status
            start {
              ... on Block { number timestamp }
              ... on BlocklessTimestamp { timestamp }
            }
            end {
              ... on Block { number timestamp }
              ... on BlocklessTimestamp { timestamp }
            }
            block { number timestamp }
            creator { address }
            proposer { address }
            governor { id name organization { id name } }
            organization { id name }
            quorum
            voteStats { type votesCount votersCount percent }
            events { type block { number timestamp } txHash createdAt }
            executableCalls { target value calldata signature }
            metadata { title description eta ipfsHash discourseURL snapshotURL }
          }
        }
        """
        variables = {"id": proposal_id}
        return self._make_request(query, variables)

    def get_proposal_by_onchain_id(self, governor_id: str, onchain_id: str) -> Dict[str, Any]:
        """Fetch a single proposal by governor ID and on-chain proposal ID."""
        query = """
        query GetProposalByOnchainId($governorId: ID!, $onchainId: String!) {
          proposal(governorId: $governorId, onchainId: $onchainId) {
            id
            onchainId
            status
            start {
              ... on Block { number timestamp }
              ... on BlocklessTimestamp { timestamp }
            }
            end {
              ... on Block { number timestamp }
              ... on BlocklessTimestamp { timestamp }
            }
            block { number timestamp }
            creator { address }
            proposer { address }
            governor { id name organization { id name } }
            organization { id name }
            quorum
            voteStats { type votesCount votersCount percent }
            events { type block { number timestamp } txHash createdAt }
            executableCalls { target value calldata signature }
            metadata { title description eta ipfsHash discourseURL snapshotURL }
          }
        }
        """
        variables = {"governorId": governor_id, "onchainId": onchain_id}
        return self._make_request(query, variables)

    def get_votes(
        self,
        proposal_ids: Optional[List[str]] = None,
        voter_addresses: Optional[List[str]] = None,
        pagination: Optional[PaginationInput] = None,
    ) -> Dict[str, Any]:
        """Fetch votes for specific proposals or voters."""
        query = """
        query GetVotes($input: VotesInput!) {
          votes(input: $input) {
            nodes {
              id
              choice
              weight
              voter { address }
              proposal { id onchainId metadata { title } }
              block { number timestamp }
              txHash
              reason
            }
            pageInfo { lastCursor }
          }
        }
        """
        input_vars: Dict[str, Any] = {}
        if proposal_ids:
            input_vars["proposalIds"] = proposal_ids
        if voter_addresses:
            input_vars["voterAddresses"] = voter_addresses
        if pagination:
            page_dict: Dict[str, Any] = {"limit": pagination.limit}
            if pagination.after_cursor:
                page_dict["afterCursor"] = pagination.after_cursor
            input_vars["page"] = page_dict
        else:
            input_vars["page"] = {"limit": 20}
        variables = {"input": input_vars}
        return self._make_request(query, variables)

    def get_live_proposals(self, governor_id: str, limit: int = 20) -> Dict[str, Any]:
        """Convenience method to get live (active) proposals for a specific governor."""
        filters = ProposalFilters(
            governor_id=governor_id, status=ProposalStatus.ACTIVE)
        pagination = PaginationInput(limit=limit)
        return self.get_proposals(filters=filters, pagination=pagination)

    def get_recent_proposals(self, governor_id: str, limit: int = 20) -> Dict[str, Any]:
        """Convenience method to get the most recent proposals for a specific governor."""
        filters = ProposalFilters(governor_id=governor_id)
        pagination = PaginationInput(limit=limit)
        sort = SortInput(sort_by="id", is_descending=True)
        # Note: Tally may ignore sort; we keep API for compatibility
        return self.get_proposals(filters=filters, pagination=pagination, sort=sort)

    def get_proposals(
        self,
        filters: Optional[ProposalFilters] = None,
        pagination: Optional[PaginationInput] = None,
        sort: Optional[SortInput] = None
    ) -> Dict[str, Any]:
        """
        Fetch proposals with optional filters, pagination, and sorting.

        Args:
            filters: Filters to apply to the query
            pagination: Pagination parameters
            sort: Sort parameters

        Returns:
            Dictionary containing proposals data and pagination info
        """
        query = """
        query GetProposals($input: ProposalsInput!) {
          proposals(input: $input) {
            nodes {
              ... on Proposal {
                id
                onchainId
                status
                start {
                  ... on Block { number timestamp }
                  ... on BlocklessTimestamp { timestamp }
                }
                end {
                  ... on Block { number timestamp }
                  ... on BlocklessTimestamp { timestamp }
                }
                block { number timestamp }
                creator { address }
                proposer { address }
                governor { id name organization { id name } }
                organization { id name }
                quorum
                voteStats { type votesCount votersCount percent }
                events { type block { number timestamp } txHash }
                executableCalls { target value calldata signature }
                metadata { title description eta ipfsHash discourseURL snapshotURL }
              }
            }
            pageInfo { firstCursor lastCursor count }
          }
        }
        """

        # Build input variables
        input_vars = {}

        if filters:
            filter_dict = {}
            if filters.governor_id:
                filter_dict["governorId"] = filters.governor_id
            if filters.organization_id:
                filter_dict["organizationId"] = filters.organization_id
            if filters.status:
                filter_dict["status"] = filters.status.value
            if filters.proposer:
                filter_dict["proposer"] = filters.proposer
            # Avoid undocumented fields that may cause validation errors

            if filter_dict:
                input_vars["filters"] = filter_dict

        if pagination:
            page_dict = {"limit": pagination.limit}
            if pagination.after_cursor:
                page_dict["afterCursor"] = pagination.after_cursor
            input_vars["page"] = page_dict
        else:
            input_vars["page"] = {"limit": 20}

        # Omit sort for compatibility
        variables = {"input": input_vars}

        return self._make_request(query, variables)

    def get_organization(self, slug: Optional[str] = None, org_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Look up a Tally organization by slug or id.

        One of slug or org_id must be provided.
        Returns minimal organization info including its id and governorIds.
        """
        if not slug and not org_id:
            raise ValueError("Either 'slug' or 'org_id' must be provided")

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

        if slug:
            variables = {"input": {"slug": slug}}
        else:
            variables = {"input": {"id": org_id}}

        return self._make_request(query, variables)


def _flatten_proposal_node(node: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a Tally proposals node to a tabular-friendly dict with metadata JSON."""
    def _extract_block(ts_obj: Optional[Dict[str, Any]]) -> Tuple[Optional[int], Optional[int]]:
        if not ts_obj:
            return None, None
        # Union types: may contain 'number' and/or 'timestamp'
        number = ts_obj.get("number")
        timestamp = ts_obj.get("timestamp")
        return number, timestamp

    start_num, start_ts = _extract_block(node.get("start"))
    end_num, end_ts = _extract_block(node.get("end"))
    block_num, block_ts = _extract_block(node.get("block"))

    flattened = {
        "id": node.get("id"),
        "onchain_id": node.get("onchainId"),
        "status": node.get("status"),
        "creator_address": (node.get("creator") or {}).get("address"),
        "proposer_address": (node.get("proposer") or {}).get("address"),
        "governor_id": (node.get("governor") or {}).get("id"),
        "governor_name": (node.get("governor") or {}).get("name"),
        "organization_id": (node.get("organization") or {}).get("id"),
        "organization_name": (node.get("organization") or {}).get("name"),
        "quorum": node.get("quorum"),
        "start_block": start_num,
        "start_timestamp": start_ts,
        "end_block": end_num,
        "end_timestamp": end_ts,
        "block_number": block_num,
        "block_timestamp": block_ts,
    }

    metadata = node.get("metadata") or {}
    flattened["title"] = metadata.get("title")
    flattened["description"] = metadata.get("description")
    flattened["eta"] = metadata.get("eta")
    flattened["ipfs_hash"] = metadata.get("ipfsHash")
    flattened["discourse_url"] = metadata.get("discourseURL")
    flattened["snapshot_url"] = metadata.get("snapshotURL")

    # Also keep raw metadata as JSON to preserve all details
    flattened["metadata_json"] = json.dumps(metadata, ensure_ascii=False)

    # Vote stats: array of types (for, against, abstain)
    vote_stats = node.get("voteStats") or []
    # Summaries by type
    try:
        for vs in vote_stats:
            vs_type = vs.get("type")
            if not vs_type:
                continue
            flattened[f"votes_{vs_type}_count"] = vs.get("votesCount")
            flattened[f"votes_{vs_type}_voters"] = vs.get("votersCount")
            flattened[f"votes_{vs_type}_percent"] = vs.get("percent")
    except Exception:
        pass

    return flattened


def write_proposals_to_csv(nodes: List[Dict[str, Any]], output_path: str) -> None:
    if not nodes:
        return
    flattened = [_flatten_proposal_node(n) for n in nodes]
    df = pd.DataFrame(flattened)
    # Column order preference (subset)
    preferred = [
        "id", "onchain_id", "status", "title", "organization_id", "organization_name",
        "governor_id", "governor_name", "creator_address", "proposer_address",
        "start_block", "start_timestamp", "end_block", "end_timestamp",
        "block_number", "block_timestamp", "quorum", "metadata_json",
    ]
    cols = [c for c in preferred if c in df.columns] + \
        [c for c in df.columns if c not in preferred]
    df = df[cols]
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


def write_proposals_to_parquet(nodes: List[Dict[str, Any]], output_path: str) -> None:
    if not nodes:
        return
    flattened = [_flatten_proposal_node(n) for n in nodes]
    df = pd.DataFrame(flattened)
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)


def fetch_all_proposals_for_organization(
    client: TallyClient,
    organization: str,
    is_slug: bool = True,
    page_limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Fetch all proposals for a DAO organization. If is_slug is True, resolves the org id first.
    Returns a list of proposals nodes.
    """
    # Resolve organization id if needed
    if is_slug:
        org_resp = client.get_organization(slug=organization)
        org = (org_resp.get("data") or {}).get("organization") or {}
        org_id = org.get("id")
        if not org_id:
            raise RuntimeError(
                f"Organization not found for slug '{organization}'")
    else:
        org_id = organization

    # Clamp to Tally's hard limit of 20
    effective_limit = min(page_limit, 20)

    all_nodes: List[Dict[str, Any]] = []
    after_cursor: Optional[str] = None
    page_num = 0
    while True:
        filters = ProposalFilters(organization_id=str(org_id))
        pagination = PaginationInput(
            limit=effective_limit, after_cursor=after_cursor)
        resp = client.get_proposals(filters=filters, pagination=pagination)
        data = resp.get("data", {})
        proposals = data.get("proposals", {})
        nodes = proposals.get("nodes", []) or []
        page_info = proposals.get("pageInfo", {})
        all_nodes.extend(nodes)
        page_num += 1
        # Visual progress bar similar to offchain votes module
        filled = min(20, page_num)
        progress_bar = "█" * filled + "░" * max(0, 20 - filled)
        print(
            f"\r[PROGRESS] [{progress_bar}] Page {page_num}: {len(all_nodes):,} proposals",
            end="",
            flush=True,
        )
        new_cursor = page_info.get("lastCursor")
        if not new_cursor or new_cursor == after_cursor or not nodes:
            break
        after_cursor = new_cursor
    # Newline after progress bar to avoid overwriting next prints
    try:
        print()
    except Exception:
        pass
    return all_nodes


def export_proposals_for_organization(
    client: TallyClient,
    organization: str,
    output_dir: str,
    is_slug: bool = True,
    page_limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Fetch all proposals for a DAO organization and write CSV/Parquet files to output_dir.

    - If is_slug is True, resolves the organization id from the provided slug.
    - Writes onchain_proposals.csv and onchain_proposals.parquet under output_dir.
    - Returns the list of proposal nodes.
    """
    nodes = fetch_all_proposals_for_organization(
        client=client,
        organization=organization,
        is_slug=is_slug,
        page_limit=page_limit,
    )
    csv_path = os.path.join(output_dir, "onchain_proposals.csv")
    parquet_path = os.path.join(output_dir, "onchain_proposals.parquet")
    write_proposals_to_csv(nodes, csv_path)
    write_proposals_to_parquet(nodes, parquet_path)
    return nodes


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Fetch Tally on-chain proposals for a DAO organization")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--org-slug", dest="org_slug",
                       help="Organization slug on Tally (e.g., 'uniswap')")
    group.add_argument("--org-id", dest="org_id",
                       help="Organization ID on Tally (e.g., '2206072050458560434')")
    parser.add_argument("--output", dest="output_dir",
                        default=".", help="Output directory for CSV/Parquet")
    parser.add_argument("--api-key", dest="api_key",
                        default=os.environ.get("TALLY_API_KEY"), help="Tally API key")
    parser.add_argument("--limit", dest="limit", type=int,
                        default=100, help="Page size (default: 100)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging (DEBUG)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Reduce logging output (ERROR)")
    parser.add_argument("--reqs-per-min", dest="reqs_per_min", type=int, default=None,
                        help="Client-side request rate limit (per minute)")

    args = parser.parse_args()
    if args.verbose:
        _level = logging.DEBUG
    elif args.quiet:
        _level = logging.ERROR
    else:
        _level = logging.INFO
    logging.basicConfig(level=_level)
    if not args.api_key:
        raise SystemExit(
            "TALLY_API_KEY is required (provide --api-key or set env var)")

    client = TallyClient(api_key=args.api_key, reqs_per_min=args.reqs_per_min)
    is_slug = args.org_slug is not None
    organization = args.org_slug or args.org_id

    os.makedirs(args.output_dir, exist_ok=True)
    nodes = export_proposals_for_organization(
        client=client,
        organization=organization,
        output_dir=args.output_dir,
        is_slug=is_slug,
        page_limit=args.limit,
    )
    print(f"Exported {len(nodes)} proposals to {args.output_dir}")
