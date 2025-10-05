"""
Tally GraphQL API client for fetching on-chain vote data.

This module provides functionality to fetch individual on-chain ballots that
token-holders cast through Tally, with comprehensive filtering, pagination,
and analysis capabilities.

Usage:
    from src.fetch_onchain_votes import TallyVotesClient

    client = TallyVotesClient(api_key="your_api_key")
    votes_df = client.fetch_votes_for_proposal(proposal_id=41024)
"""

import requests
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from typing import Literal
import argparse
from dataclasses import dataclass
from enum import Enum
import logging
import time
import backoff
from pathlib import Path
import csv
import os
import sys
import tempfile
from decimal import Decimal, InvalidOperation
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class VoteType(Enum):
    """Enum for vote type values."""
    FOR = "for"
    AGAINST = "against"
    ABSTAIN = "abstain"
    PENDING_FOR = "pendingFor"
    PENDING_AGAINST = "pendingAgainst"
    PENDING_ABSTAIN = "pendingAbstain"


@dataclass
class VoteFilters:
    """Filters for vote queries."""
    proposal_id: Optional[int] = None
    proposal_ids: Optional[List[int]] = None
    voter: Optional[str] = None
    vote_type: Optional[VoteType] = None
    include_pending_votes: bool = False
    is_veto_vote: bool = False
    chain_id: Optional[str] = None
    has_reason: Optional[bool] = None


@dataclass
class VotePagination:
    """Pagination parameters for vote queries."""
    limit: int = 100
    after_cursor: Optional[str] = None


@dataclass
class VoteSort:
    """Sort parameters for vote queries."""
    sort_by: str = "id"  # "id" or "amount"
    is_descending: bool = False


class TallyVotesClient:
    """Client for fetching on-chain votes from Tally's GraphQL API."""

    BASE_URL = "https://api.tally.xyz/query"
    MAX_BACKOFF_TIME = 120  # Maximum time for exponential backoff
    REQS_PER_MIN = 30  # Very conservative rate limit to avoid 429s

    def __init__(self, api_key: str, timeout: int = 30, show_progress: Optional[bool] = None):
        """
        Initialize the Tally votes client.

        Args:
            api_key: Your Tally API key from User Settings
            timeout: Request timeout in seconds
            show_progress: Whether to print progress bars; defaults to True when stdout is a TTY
        """
        self.api_key = api_key
        self.timeout = timeout
        self.headers = {
            "Api-Key": api_key,
            "Content-Type": "application/json",
        }
        # Reuse HTTP connections
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        # Progress display control
        self.show_progress = show_progress if show_progress is not None else sys.stdout.isatty()
        self.last_request_time = 0

    @backoff.on_exception(
        backoff.expo,
        (requests.HTTPError, requests.RequestException),
        max_time=MAX_BACKOFF_TIME,
        giveup=lambda e: hasattr(
            e, 'response') and e.response.status_code not in (429, 502, 503, 504),
        jitter=backoff.full_jitter
    )
    def _make_request(self, query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a GraphQL request to the Tally API with automatic backoff on retryable errors.

        Args:
            query: GraphQL query string
            variables: Query variables

        Returns:
            JSON response from the API

        Raises:
            requests.RequestException: If the request fails
        """
        # Rate limiting
        elapsed = time.time() - self.last_request_time
        min_interval = 60.0 / self.REQS_PER_MIN
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        payload = {
            "query": query,
            "variables": variables
        }

        response = self.session.post(
            self.BASE_URL,
            json=payload,
            timeout=self.timeout
        )

        self.last_request_time = time.time()

        if response.status_code != 200:
            # If rate limited and server tells us when to retry, respect it
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        sleep_for = float(retry_after)
                        logger.warning(
                            f"429 Too Many Requests. Sleeping for {sleep_for}s per Retry-After header.")
                        time.sleep(sleep_for)
                    except ValueError:
                        pass
            logger.error(f"HTTP {response.status_code}: {response.text}")

        response.raise_for_status()

        data = response.json()
        if 'errors' in data:
            logger.error(f"GraphQL errors: {data['errors']}")
            raise Exception(f"GraphQL errors: {data['errors']}")

        return data

    def get_votes(
        self,
        filters: Optional[VoteFilters] = None,
        pagination: Optional[VotePagination] = None,
        sort: Optional[VoteSort] = None
    ) -> Dict[str, Any]:
        """
        Fetch votes with optional filters, pagination, and sorting.

        Args:
            filters: Filters to apply to the query
            pagination: Pagination parameters
            sort: Sort parameters

        Returns:
            Dictionary containing votes data and pagination info
        """
        query = """
        query GetVotes($input: VotesInput!) {
          votes(input: $input) {
            nodes {
              ... on OnchainVote {
                id
                amount
                type
                reason
                voter {
                  address
                  ens
                  name
                }
                proposal {
                  id
                  onchainId
                  metadata { title }
                }
                block { number timestamp }
                txHash
              }
              ... on VetoVote {
                id
                amount
                type
                voter {
                  address
                  ens
                  name
                }
                proposal {
                  id
                  onchainId
                  metadata { title }
                }
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
            if filters.proposal_id is not None:
                # API expects list under proposalIds
                try:
                    filter_dict["proposalIds"] = [int(filters.proposal_id)]
                except Exception:
                    filter_dict["proposalIds"] = [filters.proposal_id]
            if filters.proposal_ids:
                try:
                    filter_dict["proposalIds"] = [
                        int(x) for x in filters.proposal_ids]
                except Exception:
                    filter_dict["proposalIds"] = list(filters.proposal_ids)
            if filters.voter:
                filter_dict["voterAddresses"] = [filters.voter]
            if filters.vote_type:
                filter_dict["type"] = filters.vote_type.value
            if filters.include_pending_votes:
                filter_dict["includePendingVotes"] = filters.include_pending_votes
            if filters.is_veto_vote:
                filter_dict["isVetoVote"] = filters.is_veto_vote
            if filters.chain_id:
                filter_dict["chainId"] = filters.chain_id
            if filters.has_reason is not None:
                filter_dict["hasReason"] = filters.has_reason

            if filter_dict:
                input_vars["filters"] = filter_dict

        if pagination:
            # Tally caps limit at 20
            page_limit = min(20, pagination.limit)
            page_dict = {"limit": page_limit}
            if pagination.after_cursor:
                # Use afterCursor parameter name as expected by Tally API
                page_dict["afterCursor"] = pagination.after_cursor
            input_vars["page"] = page_dict
        else:
            input_vars["page"] = {"limit": 20}

        if sort:
            allowed_sort_fields = {"id", "amount"}
            sort_by_value = sort.sort_by if sort.sort_by in allowed_sort_fields else "id"
            if sort.sort_by not in allowed_sort_fields:
                logger.warning(
                    f"Invalid sort_by '{sort.sort_by}', defaulting to 'id'")
            input_vars["sort"] = {
                "sortBy": sort_by_value,
                "isDescending": sort.is_descending
            }
        else:
            input_vars["sort"] = {"sortBy": "id", "isDescending": False}

        variables = {"input": input_vars}

        # Debug logging to see what we're sending
        logger.debug(f"GraphQL request variables: {variables}")

        return self._make_request(query, variables)

    def fetch_votes_for_proposal(
        self,
        proposal_id: int,
        include_pending_votes: bool = False,
        token_decimals: int = 18,
        batch_size: int = 100
    ) -> pd.DataFrame:
        """
        Fetch all votes for a single proposal and return as DataFrame.

        Args:
            proposal_id: Tally proposal ID
            include_pending_votes: Whether to include pending votes
            token_decimals: Number of decimals for token conversion (default 18)
            batch_size: Number of votes to fetch per request

        Returns:
            DataFrame with vote data
        """
        start_time = time.time()
        logger.info(f"Fetching votes for proposal {proposal_id}")
        if self.show_progress:
            print(f"[STARTING] Fetching votes for proposal {proposal_id}...")

        filters = VoteFilters(
            proposal_id=proposal_id,
            include_pending_votes=include_pending_votes
        )
        pagination = VotePagination(limit=batch_size)
        sort = VoteSort(sort_by="id", is_descending=False)

        all_votes = []
        cursor = None
        page_count = 0
        total_count = None

        while True:
            page_count += 1
            logger.debug(f"Fetching page {page_count} (cursor: {cursor})")

            pagination.after_cursor = cursor

            try:
                response = self.get_votes(
                    filters=filters, pagination=pagination, sort=sort)

                votes_data = response["data"]["votes"]
                nodes = votes_data["nodes"]

                if not nodes:
                    logger.info("No more votes to fetch")
                    break

                all_votes.extend(nodes)

                # Capture total count if provided
                if total_count is None:
                    try:
                        total_count = int(votes_data.get(
                            "pageInfo", {}).get("count"))
                    except Exception:
                        total_count = None

                # Progress reporting
                if self.show_progress:
                    elapsed = time.time() - start_time
                    rate = len(all_votes) / elapsed if elapsed > 0 else 0
                    eta_str = ""

                    if total_count and len(all_votes) > 0 and rate > 0:
                        remaining = max(0, total_count - len(all_votes))
                        eta_seconds = remaining / rate
                        if eta_seconds > 0:
                            eta_str = f" (ETA: {int(eta_seconds)}s)"

                    # Match offchain visual style: solid blocks for progress, light blocks for remaining
                    progress_bar = "█" * \
                        min(20, page_count) + "░" * max(0, 20 - page_count)
                    print(
                        f"\r[PROGRESS] [{progress_bar}] Page {page_count}: {len(all_votes):,} votes | {elapsed:.1f}s | {rate:.1f} votes/s{eta_str}",
                        end="", flush=True
                    )

                logger.debug(
                    f"Fetched {len(nodes)} votes (total: {len(all_votes)})")

                page_info = votes_data["pageInfo"]
                # Break if we got less than requested (respect 20 cap)
                requested_limit = min(20, pagination.limit)
                if len(nodes) < requested_limit:
                    logger.info("Final page detected by size < limit")
                    break

                new_cursor = page_info.get("lastCursor")
                # Stop if no cursor or cursor hasn't changed (no more pages)
                if not new_cursor or new_cursor == cursor:
                    logger.info(
                        f"Reached final page (cursor: {new_cursor}, prev: {cursor})")
                    break

                cursor = new_cursor

            except Exception as e:
                logger.error(f"Error fetching votes: {e}")
                raise

        elapsed = time.time() - start_time
        rate = len(all_votes) / elapsed if elapsed > 0 else 0

        if self.show_progress:
            print(
                f"\n[COMPLETE] Fetched {len(all_votes):,} votes in {elapsed:.1f}s ({rate:.1f} votes/s)"
            )
        logger.info(f"Fetched {len(all_votes)} total votes in {elapsed:.2f}s")

        # Validate votes
        validation = self.validate_votes(all_votes)
        if validation['duplicate_count'] > 0:
            logger.warning(
                f"Found {validation['duplicate_count']} duplicate votes")
        if validation['has_missing_data']:
            logger.warning("Some votes have missing critical data")
        logger.info(f"Validation results: {validation}")

        if not all_votes:
            return pd.DataFrame()

        # Convert to DataFrame and process
        df = pd.DataFrame(all_votes)

        # Extract nested voter information
        if 'voter' in df.columns:
            voter_df = pd.json_normalize(df['voter'])
            voter_df.columns = [f'voter_{col}' for col in voter_df.columns]
            df = pd.concat([df.drop('voter', axis=1), voter_df], axis=1)

        # Extract nested proposal information
        if 'proposal' in df.columns:
            proposal_df = pd.json_normalize(df['proposal'])
            proposal_df.columns = [
                f'proposal_{col}' for col in proposal_df.columns]
            df = pd.concat([df.drop('proposal', axis=1), proposal_df], axis=1)

        # Extract nested block information
        if 'block' in df.columns:
            block_df = pd.json_normalize(df['block'])
            block_df.columns = [f'block_{col}' for col in block_df.columns]
            df = pd.concat([df.drop('block', axis=1), block_df], axis=1)

        # Ensure key columns exist even for union variants
        canonical_cols = [
            'id', 'type', 'amount', 'createdAt', 'reason',
            'voter_address', 'voter_ens', 'voter_name',
            'proposal_id', 'proposal_onchainId', 'proposal_metadata.title',
            'block_number', 'block_timestamp', 'txHash'
        ]
        for col in canonical_cols:
            if col not in df.columns:
                df[col] = pd.NA

        # Address normalization
        if 'voter_address' in df.columns:
            df['voter_address'] = df['voter_address'].astype(
                'string').str.lower()

        # Convert amount to human-readable votes and keep raw
        if 'amount' in df.columns:
            df['amount_raw'] = df['amount']
            df['votes'] = pd.to_numeric(
                df['amount'], errors='coerce') / (10 ** token_decimals)

        # Convert timestamps
        if 'createdAt' in df.columns:
            df['createdAt'] = pd.to_datetime(
                df['createdAt'], errors='coerce', utc=True)

        if 'block_timestamp' in df.columns:
            df['block_timestamp'] = pd.to_datetime(
                df['block_timestamp'], errors='coerce', utc=True)

        # Dtypes for stability
        for str_col in ['id', 'type', 'voter_address', 'voter_ens', 'voter_name', 'txHash']:
            if str_col in df.columns:
                df[str_col] = df[str_col].astype('string')
        for int_col in ['block_number']:
            if int_col in df.columns:
                df[int_col] = pd.to_numeric(
                    df[int_col], errors='coerce').astype('Int64')

        # Reorder columns for better readability
        column_order = [
            'id', 'type', 'votes', 'amount_raw', 'voter_address', 'voter_ens', 'voter_name',
            'createdAt', 'reason', 'proposal_id', 'proposal_onchainId',
            'proposal_metadata.title', 'block_number', 'block_timestamp', 'txHash'
        ]

        existing_columns = [col for col in column_order if col in df.columns]
        remaining_columns = [
            col for col in df.columns if col not in existing_columns]
        df = df[existing_columns + remaining_columns]

        return df

    def fetch_votes_for_multiple_proposals(
        self,
        proposal_ids: List[int],
        include_pending_votes: bool = False,
        token_decimals: int = 18,
        batch_size: int = 100,
        max_workers: int = 1
    ) -> pd.DataFrame:
        """
        Fetch votes for multiple proposals and return as combined DataFrame.

        Args:
            proposal_ids: List of Tally proposal IDs
            include_pending_votes: Whether to include pending votes
            token_decimals: Number of decimals for token conversion
            batch_size: Number of votes to fetch per request

        Returns:
            Combined DataFrame with vote data from all proposals
        """
        all_dfs = []

        if max_workers <= 1:
            for proposal_id in proposal_ids:
                logger.info(f"Fetching votes for proposal {proposal_id}")
                df = self.fetch_votes_for_proposal(
                    proposal_id=proposal_id,
                    include_pending_votes=include_pending_votes,
                    token_decimals=token_decimals,
                    batch_size=batch_size
                )
                if not df.empty:
                    all_dfs.append(df)
        else:
            # Bounded concurrency; beware of API rate limits
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_pid = {
                    executor.submit(
                        self.fetch_votes_for_proposal,
                        proposal_id=pid,
                        include_pending_votes=include_pending_votes,
                        token_decimals=token_decimals,
                        batch_size=batch_size
                    ): pid for pid in proposal_ids
                }
                for future in as_completed(future_to_pid):
                    pid = future_to_pid[future]
                    try:
                        df = future.result()
                        if not df.empty:
                            all_dfs.append(df)
                    except Exception as e:
                        logger.error(
                            f"Failed to fetch votes for proposal {pid}: {e}")

        if not all_dfs:
            return pd.DataFrame()

        combined_df = pd.concat(all_dfs, ignore_index=True)
        return combined_df

    def fetch_votes_by_voter(
        self,
        voter_address: str,
        include_pending_votes: bool = False,
        token_decimals: int = 18,
        batch_size: int = 100
    ) -> pd.DataFrame:
        """
        Fetch all votes cast by a specific voter.

        Args:
            voter_address: Ethereum address of the voter
            include_pending_votes: Whether to include pending votes
            token_decimals: Number of decimals for token conversion
            batch_size: Number of votes to fetch per request

        Returns:
            DataFrame with vote data for the voter
        """
        start_time = time.time()
        logger.info(f"Fetching votes for voter {voter_address}")
        if self.show_progress:
            print(
                f"[STARTING] Fetching votes for voter {voter_address[:10]}...")

        filters = VoteFilters(
            voter=voter_address,
            include_pending_votes=include_pending_votes
        )
        pagination = VotePagination(limit=batch_size)
        sort = VoteSort(sort_by="id", is_descending=True)

        all_votes = []
        cursor = None
        page_count = 0

        while True:
            page_count += 1
            pagination.after_cursor = cursor

            try:
                response = self.get_votes(
                    filters=filters, pagination=pagination, sort=sort)

                votes_data = response["data"]["votes"]
                nodes = votes_data["nodes"]

                if not nodes:
                    break

                all_votes.extend(nodes)

                # Progress reporting
                if self.show_progress:
                    elapsed = time.time() - start_time
                    rate = len(all_votes) / elapsed if elapsed > 0 else 0
                    progress_bar = "█" * \
                        min(20, page_count) + "░" * max(0, 20 - page_count)
                    print(
                        f"\r[PROGRESS] [{progress_bar}] Page {page_count}: {len(all_votes):,} votes | {elapsed:.1f}s | {rate:.1f} votes/s",
                        end="", flush=True
                    )

                page_info = votes_data["pageInfo"]
                new_cursor = page_info.get("lastCursor")

                # Stop if no cursor or cursor hasn't changed (no more pages)
                if not new_cursor or new_cursor == cursor:
                    logger.debug(
                        f"Reached final page (cursor: {new_cursor}, prev: {cursor})")
                    break

                cursor = new_cursor

            except Exception as e:
                logger.error(f"Error fetching votes: {e}")
                raise

        elapsed = time.time() - start_time
        rate = len(all_votes) / elapsed if elapsed > 0 else 0

        if self.show_progress:
            print(
                f"\n[COMPLETE] Fetched {len(all_votes):,} votes in {elapsed:.1f}s ({rate:.1f} votes/s)"
            )
        logger.info(f"Fetched {len(all_votes)} total votes in {elapsed:.2f}s")

        if not all_votes:
            return pd.DataFrame()

        # Process similar to fetch_votes_for_proposal
        df = pd.DataFrame(all_votes)

        if 'voter' in df.columns:
            voter_df = pd.json_normalize(df['voter'])
            voter_df.columns = [f'voter_{col}' for col in voter_df.columns]
            df = pd.concat([df.drop('voter', axis=1), voter_df], axis=1)

        if 'proposal' in df.columns:
            proposal_df = pd.json_normalize(df['proposal'])
            proposal_df.columns = [
                f'proposal_{col}' for col in proposal_df.columns]
            df = pd.concat([df.drop('proposal', axis=1), proposal_df], axis=1)

        if 'block' in df.columns:
            block_df = pd.json_normalize(df['block'])
            block_df.columns = [f'block_{col}' for col in block_df.columns]
            df = pd.concat([df.drop('block', axis=1), block_df], axis=1)

        # Ensure key columns exist even for union variants
        canonical_cols = [
            'id', 'type', 'amount', 'createdAt', 'reason',
            'voter_address', 'voter_ens', 'voter_name',
            'proposal_id', 'proposal_onchainId', 'proposal_metadata.title',
            'block_number', 'block_timestamp', 'txHash'
        ]
        for col in canonical_cols:
            if col not in df.columns:
                df[col] = pd.NA

        if 'voter_address' in df.columns:
            df['voter_address'] = df['voter_address'].astype(
                'string').str.lower()

        if 'amount' in df.columns:
            df['amount_raw'] = df['amount']
            df['votes'] = pd.to_numeric(
                df['amount'], errors='coerce') / (10 ** token_decimals)

        if 'createdAt' in df.columns:
            df['createdAt'] = pd.to_datetime(
                df['createdAt'], errors='coerce', utc=True)

        if 'block_timestamp' in df.columns:
            df['block_timestamp'] = pd.to_datetime(
                df['block_timestamp'], errors='coerce', utc=True)

        return df

    def validate_votes(self, votes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate fetched votes for completeness and integrity.

        Args:
            votes: List of vote dictionaries

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'total_votes': len(votes),
            'unique_votes': len(set(v.get('id', '') for v in votes)),
            'duplicate_count': len(votes) - len(set(v.get('id', '') for v in votes)),
            'total_vp': 0,
            'voters': len(set(v.get('voter', {}).get('address', '') for v in votes)),
            'timestamp_order_valid': True,
            'has_missing_data': False,
            'invalid_types': 0
        }

        # Sum amounts using Decimal to avoid precision loss
        total_vp = Decimal(0)
        for vote in votes:
            try:
                total_vp += Decimal(str(vote.get('amount', '0')))
            except (InvalidOperation, TypeError):
                pass
        validation_results['total_vp'] = float(total_vp)

        # Check for missing critical data
        for vote in votes:
            if not vote.get('id') or not vote.get('voter', {}).get('address'):
                validation_results['has_missing_data'] = True
                logger.warning(
                    f"Vote missing critical data: {vote.get('id', 'unknown')}")

        # Check vote types and timestamp ordering per proposal
        by_proposal: Dict[str, List[Dict[str, Any]]] = {}
        for vote in votes:
            vtype = vote.get('type')
            if vtype and vtype not in {v.value for v in VoteType}:
                validation_results['invalid_types'] += 1
            pid = None
            proposal = vote.get('proposal') or {}
            pid = proposal.get('id')
            by_proposal.setdefault(pid, []).append(vote)

        out_of_order_count = 0
        total_with_ts = 0
        for pid, pvotes in by_proposal.items():
            prev_created = 0
            for vote in pvotes:
                if 'createdAt' in vote and vote['createdAt']:
                    try:
                        created_ts = pd.to_datetime(
                            vote['createdAt']).timestamp()
                        total_with_ts += 1
                        if created_ts < prev_created:
                            out_of_order_count += 1
                        prev_created = max(prev_created, created_ts)
                    except Exception:
                        continue

        # Some out-of-order timestamps are normal due to blockchain finality
        # More than 10% out of order
        if total_with_ts > 0 and out_of_order_count > len(votes) * 0.1:
            validation_results['timestamp_order_valid'] = False

        return validation_results

    def analyze_vote_patterns(self, votes_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze voting patterns from a votes DataFrame.

        Args:
            votes_df: DataFrame with vote data

        Returns:
            Dictionary with analysis results
        """
        if votes_df.empty:
            return {"error": "No votes to analyze"}

        analysis = {}

        # Basic statistics
        analysis["total_votes"] = len(votes_df)
        analysis["unique_voters"] = votes_df['voter_address'].nunique(
        ) if 'voter_address' in votes_df.columns else 0

        # Vote type distribution
        if 'type' in votes_df.columns:
            analysis["vote_type_distribution"] = votes_df['type'].value_counts(
            ).to_dict()

        # Voting power distribution
        if 'votes' in votes_df.columns:
            analysis["total_voting_power"] = votes_df['votes'].sum()
            analysis["avg_voting_power"] = votes_df['votes'].mean()
            analysis["median_voting_power"] = votes_df['votes'].median()
            analysis["top_10_voters"] = votes_df.nlargest(
                10, 'votes')[['voter_address', 'votes']].to_dict('records')

        # Time-based analysis
        if 'createdAt' in votes_df.columns:
            analysis["voting_timeline"] = {
                "first_vote": votes_df['createdAt'].min().isoformat(),
                "last_vote": votes_df['createdAt'].max().isoformat(),
                "voting_duration_hours": (votes_df['createdAt'].max() - votes_df['createdAt'].min()).total_seconds() / 3600
            }

        # Reason analysis
        if 'reason' in votes_df.columns:
            votes_with_reason = votes_df[votes_df['reason'].notna()]
            analysis["votes_with_reason"] = len(votes_with_reason)
            analysis["reason_percentage"] = (
                len(votes_with_reason) / len(votes_df)) * 100

        return analysis

    def bulk_fetch_dao_votes(
        self,
        dao_name: str,
        proposals_csv_path: str,
        output_base_dir: str = "output",
        resume: bool = False,
        delay: float = 1.0,
        include_pending_votes: bool = False,
        token_decimals: int = 18
    ) -> Dict[str, Any]:
        """
        Fetch votes for all proposals in a DAO from proposals CSV.

        Args:
            dao_name: Name of the DAO (for output directory)
            proposals_csv_path: Path to CSV file containing proposal data
            output_base_dir: Base directory for output files
            resume: Whether to skip already fetched proposals
            delay: Delay between proposal fetches (seconds)
            include_pending_votes: Whether to include pending votes
            token_decimals: Number of decimals for token conversion

        Returns:
            Dictionary with fetch results and statistics
        """
        proposals_path = Path(proposals_csv_path)
        if not proposals_path.exists():
            raise FileNotFoundError(
                f"Proposals CSV not found: {proposals_path}")

        # Read proposals CSV
        print(f"[READING] Reading proposals from: {proposals_path}")
        proposals = []

        with open(proposals_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Extract proposal ID from the 'id' column
                if 'id' in row and row['id']:
                    proposals.append({
                        'id': row['id'],
                        'onchain_id': row.get('onchainId', ''),
                        'title': row.get('metadata.title', '')
                    })

        print(f"[FOUND] Found {len(proposals)} proposals for {dao_name}")

        # Setup progress tracking
        results = {
            'total_proposals': len(proposals),
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'total_votes': 0,
            'start_time': time.time(),
            'failed_proposals': []
        }

        # Create output and temp directories (use unique per-run temp dir to avoid clashes)
        output_dir = Path(output_base_dir) / dao_name
        output_dir.mkdir(parents=True, exist_ok=True)
        run_suffix = f"{int(time.time())}_{os.getpid()}"
        temp_dir = Path(tempfile.mkdtemp(
            prefix=f"_tmp_votes_{run_suffix}_", dir=str(output_dir)))

        # Process each proposal (write per-proposal parquet into temp dir)

        for i, proposal in enumerate(proposals, 1):
            proposal_id = proposal['id']

            # Check if already exists (for resume functionality)
            proposal_parquet_path = temp_dir / f"{proposal_id}.parquet"
            if resume and proposal_parquet_path.exists():
                results['skipped'] += 1
                print(
                    f"[SKIP] [{i:3d}/{len(proposals)}] Skipping proposal {proposal_id} (already exists)")
                # Count towards totals for verification
                try:
                    existing_df = pd.read_parquet(proposal_parquet_path)
                    results['total_votes'] += len(existing_df)
                except Exception as e:
                    logger.warning(
                        f"Could not load existing parquet {proposal_parquet_path}: {e}")
                continue

            try:
                print(
                    f"\n[PROCESSING] [{i:3d}/{len(proposals)}] Processing proposal {proposal_id}")
                if proposal['title']:
                    print(f"   Title: {proposal['title'][:60]}...")

                # Fetch votes for this proposal
                votes_df = self.fetch_votes_for_proposal(
                    proposal_id=int(proposal_id),
                    include_pending_votes=include_pending_votes,
                    token_decimals=token_decimals
                )

                if not votes_df.empty:
                    # Add proposal metadata
                    votes_df['proposal_dao'] = dao_name
                    votes_df['proposal_title'] = proposal['title']

                    # Save per-proposal parquet in temp dir atomically
                    proposal_parquet_path.parent.mkdir(
                        parents=True, exist_ok=True)
                    tmp_file = proposal_parquet_path.with_suffix(
                        '.parquet.tmp')
                    votes_df.to_parquet(
                        tmp_file, index=False, engine='pyarrow', compression='snappy')
                    os.replace(tmp_file, proposal_parquet_path)
                    print(
                        f"   [SAVED] {len(votes_df)} votes to {proposal_parquet_path.name}")

                    results['total_votes'] += len(votes_df)
                else:
                    print(
                        f"   [INFO] No votes found for proposal {proposal_id}")

                results['processed'] += 1

                # Rate limiting delay
                if delay > 0 and i < len(proposals):
                    time.sleep(delay)

            except Exception as e:
                results['failed'] += 1
                results['failed_proposals'].append({
                    'proposal_id': proposal_id,
                    'title': proposal['title'],
                    'error': str(e)
                })
                print(
                    f"[FAILED] Failed to process proposal {proposal_id}: {e}")
                logger.error(
                    f"Failed to process proposal {proposal_id}: {e}", exc_info=True)
                continue

        # Merge all votes from temp dir into consolidated CSV and Parquet
        merged_parts = []
        for part in sorted(temp_dir.glob("*.parquet")):
            try:
                merged_parts.append(pd.read_parquet(part))
            except Exception as e:
                logger.warning(f"Failed to read {part}: {e}")

        if merged_parts:
            merged_df = pd.concat(merged_parts, ignore_index=True)
            print(
                f"\n[MERGING] Merged {len(merged_df)} votes from {len(merged_parts)} files")
            merged_csv_path = output_dir / "onchain_votes.csv"
            merged_df.to_csv(merged_csv_path, index=False)
            print(f"[SAVED] Merged CSV -> {merged_csv_path}")

            merged_parquet_path = output_dir / "onchain_votes.parquet"
            try:
                tmp_merged = merged_parquet_path.with_suffix('.parquet.tmp')
                merged_df.to_parquet(
                    tmp_merged, index=False, engine='pyarrow', compression='snappy')
                os.replace(tmp_merged, merged_parquet_path)
                print(f"[SAVED] Merged parquet -> {merged_parquet_path}")
            except Exception as e:
                logger.warning(
                    f"Failed to write parquet {merged_parquet_path}: {e}")

            # Verify and cleanup temp dir
            validation = self._verify_merged_votes(
                merged_df, results['total_votes'])
            if validation['is_valid']:
                print(f"[VERIFY] ✅ Merged file verified successfully!")
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                    print(f"[CLEANUP] ✅ Deleted temp directory {temp_dir}")
                except Exception as e:
                    logger.warning(
                        f"Failed to delete temp directory {temp_dir}: {e}")
            else:
                print(
                    f"[VERIFY] ❌ Merged file validation failed: {validation['errors']}")
                print(f"[CLEANUP] ⚠️ Keeping temp files in {temp_dir}")

        # Final summary
        elapsed = time.time() - results['start_time']
        success_rate = (results['processed'] / results['total_proposals']
                        ) * 100 if results['total_proposals'] > 0 else 0

        print(f"\n[COMPLETE] Bulk processing complete!")
        print(
            f"   [STATS] Processed: {results['processed']}/{results['total_proposals']} ({success_rate:.1f}%)")
        print(f"   [SKIP] Skipped: {results['skipped']}")
        print(f"   [FAIL] Failed: {results['failed']}")
        print(f"   [VOTES] Total votes: {results['total_votes']:,}")
        print(f"   [TIME] Time: {elapsed:.1f}s ({elapsed/60:.1f}m)")

        if results['failed_proposals']:
            print(f"\n[FAILED] Failed proposals:")
            for failed in results['failed_proposals'][:5]:
                print(f"   {failed['proposal_id']}: {failed['error']}")
                if failed['title']:
                    print(f"      Title: {failed['title'][:50]}...")
            if len(results['failed_proposals']) > 5:
                print(
                    f"   ... and {len(results['failed_proposals']) - 5} more")

        return results

    def _verify_merged_votes(self, df: pd.DataFrame, expected_votes: int) -> Dict[str, Any]:
        """Verify that the merged votes DataFrame is valid and complete."""
        validation = {
            'is_valid': True,
            'errors': []
        }

        # Check vote count
        if len(df) != expected_votes:
            validation['is_valid'] = False
            validation['errors'].append(
                f"Vote count mismatch. Expected: {expected_votes}, Got: {len(df)}")

        # Check for required columns
        required_columns = ['id', 'voter_address', 'type', 'proposal_id']
        missing_columns = [
            col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation['is_valid'] = False
            validation['errors'].append(
                f"Missing required columns: {missing_columns}")

        # Check for duplicate votes
        if 'id' in df.columns:
            duplicate_count = len(df) - df['id'].nunique()
            if duplicate_count > 0:
                validation['is_valid'] = False
                validation['errors'].append(
                    f"Found {duplicate_count} duplicate vote IDs")

        # Basic data integrity checks
        if 'voter_address' in df.columns:
            null_voters = df['voter_address'].isnull().sum()
            if null_voters > 0:
                validation['errors'].append(
                    f"Found {null_voters} votes with null voter addresses")

        return validation


def fetch_votes_simple(api_key: str, proposal_id: int, limit: int = 100, start_cursor: str = None) -> pd.DataFrame:
    """
    Simple helper function to fetch all votes for a proposal.
    Uses the TallyVotesClient for improved error handling and progress reporting.

    Args:
        api_key: Tally API key
        proposal_id: Proposal ID to fetch votes for
        limit: Batch size for pagination
        start_cursor: Starting cursor for pagination (not used, kept for compatibility)

    Returns:
        DataFrame with vote data
    """
    client = TallyVotesClient(api_key=api_key)
    return client.fetch_votes_for_proposal(
        proposal_id=proposal_id,
        batch_size=limit
    )


def example_usage():
    """Example usage of the Tally votes client."""
    # Initialize client with your API key
    API_KEY = "YOUR_TALLY_API_KEY"  # Replace with your actual API key
    client = TallyVotesClient(api_key=API_KEY)

    # Example proposal ID
    PROPOSAL_ID = 41024  # Replace with actual proposal ID

    try:
        # Fetch all votes for a proposal
        print(f"Fetching votes for proposal {PROPOSAL_ID}...")
        votes_df = client.fetch_votes_for_proposal(PROPOSAL_ID)

        if not votes_df.empty:
            print(f"Fetched {len(votes_df)} votes")
            print("\nFirst 5 votes:")
            print(votes_df.head())

            # Analyze voting patterns
            print("\nVoting pattern analysis:")
            analysis = client.analyze_vote_patterns(votes_df)
            for key, value in analysis.items():
                print(f"{key}: {value}")

            # Save to CSV
            votes_df.to_csv(f"proposal_{PROPOSAL_ID}_votes.csv", index=False)
            print(f"Votes saved to proposal_{PROPOSAL_ID}_votes.csv")

        # Example: Fetch votes for a specific voter
        # Replace with actual address
        VOTER_ADDRESS = "0x1234567890123456789012345678901234567890"
        print(f"\nFetching votes for voter {VOTER_ADDRESS}...")
        voter_votes_df = client.fetch_votes_by_voter(VOTER_ADDRESS)

        if not voter_votes_df.empty:
            print(f"Voter has cast {len(voter_votes_df)} votes")
            print(
                voter_votes_df[['proposal_id', 'type', 'votes', 'createdAt']].head())

        # Example using the simple helper function
        print(f"\nUsing simple helper function...")
        simple_votes_df = fetch_votes_simple(API_KEY, PROPOSAL_ID)
        print(f"Simple function fetched {len(simple_votes_df)} votes")

    except requests.RequestException as e:
        print(f"Error making request: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch on-chain votes (unified flags)")
    parser.add_argument('dao_id', help='DAO name (used for output path)')
    parser.add_argument('--proposal-id', dest='proposal_id', type=int,
                        default=None, help='Tally proposal ID (fetch only this proposal)')
    parser.add_argument('--voter', dest='voter', type=str, default=None,
                        help='Fetch all votes cast by a voter (address)')
    parser.add_argument('--test', action='store_true',
                        help='Test mode: pick first proposal with transfers and fetch only it')
    parser.add_argument('--api-key', default=os.getenv('TALLY_API_KEY'),
                        help='Tally API key (or set env TALLY_API_KEY)')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Delay between proposals (bulk)')
    parser.add_argument('--resume', action='store_true',
                        help='Skip existing files (bulk)')
    parser.add_argument('--proposals-csv', dest='proposals_csv', default=None,
                        help='Path to proposals CSV (bulk mode)')
    parser.add_argument('--output-dir', dest='output_dir', default='data',
                        help='Base output directory')
    parser.add_argument('--include-pending', action='store_true',
                        help='Include pending votes where applicable')
    parser.add_argument('--token-decimals', type=int, default=18,
                        help='Token decimals for human-readable votes')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for pagination (capped by API)')
    parser.add_argument('--chain-id', dest='chain_id', default=None,
                        help='Optional chainId filter')
    parser.add_argument('--vote-type', dest='vote_type', default=None,
                        choices=[v.value for v in VoteType], help='Optional vote type filter')
    parser.add_argument('--has-reason', dest='has_reason', action='store_true',
                        help='Filter votes that include a reason')
    parser.add_argument('--sort-by', dest='sort_by', default='id', choices=['id', 'amount'],
                        help='Sort field')
    parser.add_argument('--sort-desc', dest='sort_desc', action='store_true',
                        help='Sort descending')
    parser.add_argument('--no-progress', action='store_true',
                        help='Disable progress output')
    parser.add_argument('--verbose', '-v',
                        action='store_true', help='Enable verbose logging (DEBUG)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Reduce logging output (ERROR)')

    args = parser.parse_args()
    if args.verbose:
        _lvl = logging.DEBUG
    elif args.quiet:
        _lvl = logging.ERROR
    else:
        _lvl = logging.INFO
    logging.basicConfig(level=_lvl)

    if not args.api_key:
        print("[ERROR] Missing Tally API key. Pass --api-key or set TALLY_API_KEY.")
        raise SystemExit(2)

    client = TallyVotesClient(api_key=args.api_key,
                              show_progress=not args.no_progress)

    # Helper to pick first proposal with transfers (expects offchain-style ids)
    def find_first_transfer_proposal(dao_id: str) -> int | None:
        try:
            base = Path(__file__).resolve().parents[2]
            path = base / 'data' / dao_id / 'governance_transfer_data' / \
                'offchain_matched_transfers.csv'
            if not path.exists():
                return None
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pid = row.get('proposal_id')
                    if pid and pid.isdigit():
                        return int(pid)
        except Exception:
            return None
        return None

    # Single proposal mode
    if args.proposal_id is not None:
        df = client.fetch_votes_for_proposal(
            args.proposal_id,
            include_pending_votes=args.include_pending,
            token_decimals=args.token_decimals,
            batch_size=args.batch_size,
        )
        out_dir = Path(args.output_dir) / args.dao_id / 'governance_data'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'onchain_proposal_{args.proposal_id}_votes.csv'
        df.to_csv(out_path, index=False)
        print(f"[SAVED] {len(df)} votes -> {out_path}")
        return

    # Voter mode
    if args.voter:
        df = client.fetch_votes_by_voter(
            args.voter,
            include_pending_votes=args.include_pending,
            token_decimals=args.token_decimals,
            batch_size=args.batch_size,
        )
        out_dir = Path(args.output_dir) / args.dao_id / 'governance_data'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'onchain_voter_{args.voter[:8]}_votes.csv'
        df.to_csv(out_path, index=False)
        print(f"[SAVED] {len(df)} votes -> {out_path}")
        return

    # Test mode -> pick first proposal with transfers
    if args.test:
        target = find_first_transfer_proposal(args.dao_id)
        if target is None:
            print(
                '[WARNING] No transfers found or non-numeric proposal ids; provide --proposal-id')
            return
        df = client.fetch_votes_for_proposal(target)
        out_dir = Path('data') / args.dao_id / 'governance_data'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'onchain_proposal_{target}_votes.csv'
        df.to_csv(out_path, index=False)
        print(f"[SAVED] {len(df)} votes -> {out_path}")
        return

    # Bulk mode
    if args.proposals_csv:
        results = client.bulk_fetch_dao_votes(
            dao_name=args.dao_id,
            proposals_csv_path=args.proposals_csv,
            output_base_dir=args.output_dir,
            resume=args.resume,
            delay=args.delay,
            include_pending_votes=args.include_pending,
            token_decimals=args.token_decimals,
        )
        # Exit code non-zero on failures
        if results.get('failed', 0) > 0:
            raise SystemExit(1)
        return

    print('[INFO] No action requested. Provide --proposal-id, --voter, or --proposals-csv for bulk mode.')


if __name__ == "__main__":
    main()
