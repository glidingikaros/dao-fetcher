from __future__ import annotations
import httpx
import backoff
import time
import logging
import json
import csv
import argparse
import sys
import pandas as pd
from pathlib import Path
import os
from typing import Optional, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pyarrow as pa
import pyarrow.parquet as pq

"""unified snapshot vote fetching module with csv and parquet export"""

# configuration
ENDPOINT = "https://hub.snapshot.org/graphql"
HEADERS = {"Content-Type": "application/json",
           "User-Agent": "dao-data-collector/1.0"}  # client.headers will be used
PAGE_SIZE = 1000
# keep small buffer below 60 rpm cap
REQS_PER_MIN = 58
MAX_BACKOFF_TIME = 120

logger = logging.getLogger(__name__)

# graphql queries for fetching votes
VOTES_QUERY_CREATED = """
query VotesPageCreated($proposal: String!, $after: Int!) {
  votes(
    first: 1000
    where: {proposal: $proposal, created_gt: $after}
    orderBy: "created"
    orderDirection: asc
  ) {
    id
    voter
    vp
    vp_by_strategy
    vp_state
    created
    choice
    reason
    app
    metadata
    ipfs
    proposal {
      id
      space { id }
    }
  }
}
"""

VOTES_QUERY_OFFSET = """
query VotesPageOffset($proposal: String!, $skip: Int!) {
  votes(
    first: 1000
    skip: $skip
    where: {proposal: $proposal}
    orderBy: "created"
    orderDirection: asc
  ) {
    id
    voter
    vp
    vp_by_strategy
    vp_state
    created
    choice
    reason
    app
    metadata
    ipfs
    proposal {
      id
      space { id }
    }
  }
}
"""

# query for checking proposal state
PROPOSAL_STATE_QUERY = """
query ProposalState($id: String!) {
  proposal(id: $id) {
    id
    state
    votes
    space {
      id
    }
  }
}
"""

# base columns for csv/parquet; choice.* added dynamically
BASE_COLUMNS = [
    'app',
    'choice',
    'created',
    'id',
    'ipfs',
    'metadata',
    'metadata.app',
    'proposal',
    'proposalId',
    'reason',
    'space',
    'voter',
    'vp',
    'vp_by_strategy',
    'vp_state'
]


@dataclass
class Vote:
    """snapshot vote data model"""
    id: str
    voter: str
    vp: float
    vp_by_strategy: Optional[list[float]] = None
    vp_state: Optional[str] = None
    created: int = 0
    choice: Any = None
    reason: Optional[str] = None
    app: Optional[str] = None
    metadata: Optional[dict] = field(default_factory=dict)
    ipfs: Optional[str] = None
    proposal: Optional[str] = None
    proposalId: Optional[str] = None
    space: Optional[str] = None

    def to_csv_row(self) -> dict:
        """convert vote to csv row with flattened fields"""
        row = {
            'app': self.app,
            'choice': json.dumps(self.choice) if isinstance(self.choice, (dict, list)) else str(self.choice),
            'created': self.created,
            'id': self.id,
            'ipfs': self.ipfs,
            'metadata': json.dumps(self.metadata) if self.metadata else None,
            'metadata.app': self.metadata.get('app') if isinstance(self.metadata, dict) else None,
            'proposal': self.proposal,
            'proposalId': self.proposalId,
            'reason': self.reason,
            'space': self.space,
            'voter': self.voter,
            'vp': self.vp,
            'vp_by_strategy': json.dumps(self.vp_by_strategy) if self.vp_by_strategy else None,
            'vp_state': self.vp_state
        }

        # handle dynamic choice fields
        # only include present choice indices; column set finalized in exporter
        if isinstance(self.choice, dict):
            for key, value in self.choice.items():
                row[f'choice.{key}'] = value
        elif isinstance(self.choice, (int, float)):
            try:
                idx = int(self.choice)
                row[f'choice.{idx}'] = 1.0
            except Exception:
                pass

        return row


class SnapshotVoteFetcher:
    """snapshot vote fetcher with retry handling"""

    def __init__(self, api_key: Optional[str] = None, endpoint: str = ENDPOINT):
        self.headers = HEADERS.copy()
        self.client = httpx.Client(
            http2=True, headers=self.headers, timeout=30)
        effective_api_key = api_key or os.getenv('SNAPSHOT_API_KEY')
        if effective_api_key:
            # use canonical casing; headers are case-insensitive
            self.client.headers['X-Api-Key'] = effective_api_key
        self.reqs_per_min = REQS_PER_MIN
        self.last_request_time = 0
        self.endpoint = endpoint
        self._proposal_cache: dict[str, dict[str, Any]] = {}

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

    def __enter__(self) -> "SnapshotVoteFetcher":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    class RetryableGraphQLError(Exception):
        pass

    @backoff.on_exception(
        backoff.expo,
        (httpx.HTTPStatusError, httpx.RequestError, RetryableGraphQLError),
        max_time=MAX_BACKOFF_TIME,
        giveup=lambda e: (not isinstance(e, SnapshotVoteFetcher.RetryableGraphQLError)) and hasattr(
            e, 'response') and e.response is not None and e.response.status_code not in (429, 500, 502, 503, 504),
        jitter=backoff.full_jitter
    )
    def _gql_request(self, query: str, variables: dict[str, Any]) -> dict:
        """Execute GraphQL request with automatic backoff on retryable errors."""
        # rate limiting
        elapsed = time.time() - self.last_request_time
        min_interval = 60.0 / self.reqs_per_min
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        try:
            response = self.client.post(
                self.endpoint,
                json={"query": query, "variables": variables},
            )

            # if rate limited, honor retry-after then raise to trigger backoff
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                try:
                    time.sleep(float(retry_after) if retry_after else 1.0)
                except Exception:
                    time.sleep(1.0)
                raise httpx.HTTPStatusError(
                    "Rate limited", request=response.request, response=response)

            response.raise_for_status()

            data = response.json()
            if 'errors' in data:
                errs = data['errors']
                err_text = str(errs).lower()
                if any(term in err_text for term in ["rate limit", "too many requests", "timeout", "timed out"]):
                    logger.warning(f"GraphQL retryable errors: {errs}")
                    raise SnapshotVoteFetcher.RetryableGraphQLError(str(errs))
                logger.error(f"GraphQL errors: {errs}")
                raise Exception(f"GraphQL errors: {errs}")

            self.last_request_time = time.time()
            return data['data']

        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP {e.response.status_code}: {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.warning(f"Request error: {e}")
            raise

    def _build_vote(self, vote_data: dict, proposal_id: str) -> Vote:
        return Vote(
            id=vote_data['id'],
            voter=vote_data['voter'],
            vp=vote_data.get('vp', 0),
            vp_by_strategy=vote_data.get('vp_by_strategy'),
            vp_state=vote_data.get('vp_state'),
            created=vote_data.get('created', 0),
            choice=vote_data.get('choice'),
            reason=vote_data.get('reason'),
            app=vote_data.get('app'),
            metadata=vote_data.get('metadata', {}),
            ipfs=vote_data.get('ipfs'),
            proposal=proposal_id,
            proposalId=proposal_id,
            space=(
                vote_data.get('proposal', {})
                or {}
            ).get('space', {}).get('id') or vote_data.get('space', {}).get('id')
        )

    def _fetch_votes_created(self, proposal_id: str, start_after: int = 0) -> list[Vote]:
        logger.info("MODE: Using created-cursor pagination")
        cursor = start_after
        votes: list[Vote] = []
        seen_ids: set[str] = set()
        page_num = 0
        start_time = time.time()

        while True:
            page_num += 1
            logger.debug(
                f"Fetching page {page_num} with created cursor: {cursor}")
            data = self._gql_request(VOTES_QUERY_CREATED, {
                                     "proposal": proposal_id, "after": cursor})
            page_votes = data.get('votes', [])

            new_added = 0
            for vote_data in page_votes:
                vid = vote_data.get('id')
                if vid in seen_ids:
                    continue
                seen_ids.add(vid)
                votes.append(self._build_vote(vote_data, proposal_id))
                new_added += 1

            elapsed = time.time() - start_time
            rate = len(votes) / elapsed if elapsed > 0 else 0
            eta_str = ""
            if page_num > 1 and len(page_votes) == PAGE_SIZE:
                estimated_total = len(votes) * 1.2
                remaining = max(0, estimated_total - len(votes))
                eta_seconds = remaining / rate if rate > 0 else 0
                if eta_seconds > 0:
                    eta_str = f" (ETA: {int(eta_seconds)}s)"

            progress_bar = "█" * min(20, page_num) + \
                "░" * max(0, 20 - page_num)
            print(
                f"\r[PROGRESS] [{progress_bar}] Page {page_num}: {len(votes):,} votes | {elapsed:.1f}s | {rate:.1f} votes/s{eta_str}", end="", flush=True)
            logger.info(
                f"Page {page_num}: fetched {len(page_votes)} votes (total: {len(votes)})")

            if len(page_votes) < PAGE_SIZE:
                break
            # overlap protection: roll back 1s to include any equal-timestamp spillover next page
            last_created = page_votes[-1]['created'] if page_votes else cursor
            cursor = max(0, (last_created or 0) - 1)
            # safety: if no new votes were added on a full page, stop to avoid potential loop
            if new_added == 0 and len(page_votes) == PAGE_SIZE:
                logger.warning(
                    "No new votes added on a full page with created-cursor; stopping to avoid loop")
                break

        return votes

    def _fetch_votes_offset(self, proposal_id: str) -> list[Vote]:
        logger.info("MODE: Using offset pagination")
        skip = 0
        votes: list[Vote] = []
        seen_ids: set[str] = set()
        page_num = 0
        start_time = time.time()

        while True:
            page_num += 1
            logger.debug(f"Fetching page {page_num} with skip: {skip}")
            data = self._gql_request(VOTES_QUERY_OFFSET, {
                                     "proposal": proposal_id, "skip": skip})
            page_votes = data.get('votes', [])

            for vote_data in page_votes:
                vid = vote_data.get('id')
                if vid in seen_ids:
                    continue
                seen_ids.add(vid)
                votes.append(self._build_vote(vote_data, proposal_id))

            elapsed = time.time() - start_time
            rate = len(votes) / elapsed if elapsed > 0 else 0
            eta_str = ""
            if page_num > 1 and len(page_votes) == PAGE_SIZE:
                estimated_total = len(votes) * 1.2
                remaining = max(0, estimated_total - len(votes))
                eta_seconds = remaining / rate if rate > 0 else 0
                if eta_seconds > 0:
                    eta_str = f" (ETA: {int(eta_seconds)}s)"

            progress_bar = "█" * min(20, page_num) + \
                "░" * max(0, 20 - page_num)
            print(
                f"\r[PROGRESS] [{progress_bar}] Page {page_num}: {len(votes):,} votes | {elapsed:.1f}s | {rate:.1f} votes/s{eta_str}", end="", flush=True)
            logger.info(
                f"Page {page_num}: fetched {len(page_votes)} votes (total: {len(votes)})")

            if len(page_votes) < PAGE_SIZE:
                break
            skip += PAGE_SIZE

        return votes

    def fetch_all_votes(self, proposal_id: str, start_after_timestamp: int = 0) -> list[Vote]:
        """Fetch all votes for a given proposal using created-cursor pagination with documented offset fallback."""
        start_time = time.time()
        logger.info(
            f"Starting vote fetch for proposal: {proposal_id} via created-cursor")
        print(f"STARTING: Fetching votes for proposal: {proposal_id[:10]}...")

        use_offset = False
        votes: list[Vote] = []

        # try created cursor first
        try:
            votes = self._fetch_votes_created(
                proposal_id, start_after=start_after_timestamp)
        except Exception as e:
            emsg = str(e)
            if (
                "Unknown argument" in emsg
                or "Cannot query field" in emsg
                or "GraphQL error" in emsg and "created_gt" in emsg
            ):
                use_offset = True
                logger.info(
                    "Falling back to offset pagination due to GraphQL validation error")
                votes = self._fetch_votes_offset(proposal_id)
            else:
                raise

        elapsed = time.time() - start_time
        rate = len(votes) / elapsed if elapsed > 0 else 0
        print(
            f"\n[COMPLETE] Completed! Fetched {len(votes):,} votes in {elapsed:.1f}s ({rate:.1f} votes/s)")
        logger.info(
            f"Completed fetching {len(votes)} votes for proposal {proposal_id} in {elapsed:.2f}s")

        # if proposal closed verify against expected and retry with offset if needed
        try:
            state, expected_count = self.get_proposal_state(proposal_id)
            if state != "active" and expected_count and len(votes) != expected_count:
                logger.warning(
                    f"Vote count mismatch: fetched {len(votes)}, expected {expected_count}. Retrying with alternate pagination"
                )
                alt_votes = self._fetch_votes_offset(proposal_id) if not use_offset else self._fetch_votes_created(
                    proposal_id, start_after=start_after_timestamp)
                # deduplicate by id
                dedup: dict[str, Vote] = {v.id: v for v in votes}
                for v in alt_votes:
                    dedup[v.id] = v
                merged = list(dedup.values())
                if len(merged) >= len(votes):
                    votes = merged
        except Exception:
            # non fatal
            pass

        return votes

    def get_proposal_info(self, proposal_id: str) -> dict[str, Any]:
        """cache and return proposal info"""
        if proposal_id in self._proposal_cache:
            return self._proposal_cache[proposal_id]
        data = self._gql_request(PROPOSAL_STATE_QUERY, {"id": proposal_id})
        proposal = data.get('proposal', {}) or {}
        info = {
            'state': proposal.get('state', 'unknown'),
            'votes': proposal.get('votes', 0),
            'space_id': (proposal.get('space') or {}).get('id', 'unknown'),
        }
        self._proposal_cache[proposal_id] = info
        return info

    def get_proposal_state(self, proposal_id: str) -> tuple[str, int]:
        """fetch proposal state and vote count"""
        info = self.get_proposal_info(proposal_id)
        return info['state'], info['votes']

    def get_proposal_space(self, proposal_id: str) -> str:
        """Get the space name for a proposal."""
        info = self.get_proposal_info(proposal_id)
        return info.get('space_id', 'unknown')

    def crawl_until_closed(self, proposal_id: str, poll_interval: int = 15) -> list[Vote]:
        """Continuously fetch votes until proposal is no longer active."""
        all_votes = []
        monitor_start = time.time()

        print(
            f"MONITORING: Starting proposal monitoring for: {proposal_id[:10]}...")

        while True:
            all_votes = self.fetch_all_votes(proposal_id)
            state, expected_count = self.get_proposal_state(proposal_id)

            total_elapsed = time.time() - monitor_start

            if state != "active":
                print(
                    f"\n[FINISHED] Proposal {state}! Final count: {len(all_votes):,} votes (monitored for {total_elapsed:.1f}s)")
                logger.info(
                    f"Proposal {proposal_id} is {state} with {len(all_votes)} votes")

                # validate completeness
                if len(all_votes) != expected_count:
                    print(
                        f"WARNING: Vote count mismatch: fetched {len(all_votes)}, expected {expected_count}. Re-fetching...")
                    logger.warning(
                        f"Vote count mismatch: fetched {len(all_votes)}, "
                        f"expected {expected_count}. Re-fetching..."
                    )
                    time.sleep(5)  # brief pause before retry
                    all_votes = self.fetch_all_votes(proposal_id)

                break

            next_check = datetime.now() + timedelta(seconds=poll_interval)
            print(
                f"\n[WAITING] Proposal still active ({len(all_votes):,} votes), next check at {next_check.strftime('%H:%M:%S')}")
            logger.info(
                f"Proposal still active ({len(all_votes)} votes so far), "
                f"waiting {poll_interval}s for new votes..."
            )
            time.sleep(poll_interval)

        return all_votes

    def validate_votes(self, votes: list[Vote]) -> dict[str, Any]:
        """validate vote completeness for logging"""
        validation_results = {
            'total_votes': len(votes),
            'unique_votes': len(set(v.id for v in votes)),
            'duplicate_count': len(votes) - len(set(v.id for v in votes)),
            'total_vp': sum(v.vp for v in votes),
            'voters': len(set(v.voter for v in votes)),
            'cursor_order_valid': True,
            'timestamp_order_valid': True
        }

        # check cursor ordering (timestamps should not regress; equality allowed)
        prev_created = 0
        for vote in votes:
            if vote.created < prev_created:
                validation_results['cursor_order_valid'] = False
                logger.error(
                    f"Cursor order violation: {prev_created} -> {vote.created}")
                break
            prev_created = vote.created

        # check timestamp ordering (should generally increase)
        prev_created = 0
        out_of_order_count = 0
        for vote in votes:
            if vote.created < prev_created:
                out_of_order_count += 1
            prev_created = vote.created

        # some out-of-order timestamps are normal due to chain finality
        if out_of_order_count > len(votes) * 0.1:  # more than 10% out of order
            validation_results['timestamp_order_valid'] = False

        return validation_results

    def bulk_fetch_space_votes(self, space_name: str, proposals_csv_path: str,
                               output_base_dir: str = "output",
                               resume: bool = False, delay: float = 1.0,
                               batch_size: int = 0) -> dict[str, Any]:
        """Fetch votes for all proposals in a space from proposals CSV."""
        proposals_path = Path(proposals_csv_path)
        if not proposals_path.exists():
            raise FileNotFoundError(
                f"Proposals CSV not found: {proposals_path}")

        # read proposals csv using built in csv module
        print(f"READING: Reading proposals from: {proposals_path}")
        proposal_ids = []

        with open(proposals_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                proposal_ids.append(row['id'])

        print(f"FOUND: Found {len(proposal_ids)} proposals in {space_name}")

        # setup progress tracking
        results = {
            'total_proposals': len(proposal_ids),
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'total_votes': 0,
            'start_time': time.time(),
            'failed_proposals': []
        }

        output_dir = Path(output_base_dir) / space_name

        for i, proposal_id in enumerate(proposal_ids, 1):
            # check if already exists (resume support)
            proposal_parquet_path = output_dir / f"{proposal_id}.parquet"
            if resume and proposal_parquet_path.exists():
                results['skipped'] += 1
                print(
                    f"SKIP: [{i:3d}/{len(proposal_ids)}] Skipping {proposal_id[:10]}... (already exists)")
                continue

            try:
                print(
                    f"\nPROCESSING: [{i:3d}/{len(proposal_ids)}] Processing {proposal_id[:10]}...")

                # fetch votes for this proposal
                votes = self.fetch_all_votes(proposal_id)
                results['total_votes'] += len(votes)

                # export votes
                exporter = VoteCSVExporter()
                if batch_size > 0:
                    # use batch export with proposal id as prefix
                    batch_output_dir = output_dir / proposal_id[:10]
                    exporter.export_batch_to_csv(
                        votes, batch_output_dir, batch_size, f"votes_{proposal_id[:10]}")
                    # also export parquet batches
                    exporter.export_batch_to_parquet(
                        votes, batch_output_dir, batch_size, f"votes_{proposal_id[:10]}")
                else:
                    # single parquet per proposal (no individual csv)
                    parquet_path = output_dir / f"{proposal_id}.parquet"
                    exporter.export_to_parquet(votes, parquet_path)

                results['processed'] += 1

                # rate limiting delay to avoid overwhelming the api
                if delay > 0 and i < len(proposal_ids):
                    time.sleep(delay)

            except Exception as e:
                results['failed'] += 1
                results['failed_proposals'].append(
                    {'proposal_id': proposal_id, 'error': str(e)})
                print(f"FAILED: Failed to process {proposal_id[:10]}: {e}")
                logger.error(f"Failed to process {proposal_id}: {e}")
                continue

        # final summary
        elapsed = time.time() - results['start_time']
        success_rate = (results['processed'] / results['total_proposals']
                        ) * 100 if results['total_proposals'] > 0 else 0

        print(f"\n[COMPLETE] Bulk processing complete!")
        print(
            f"   [STATS] Processed: {results['processed']}/{results['total_proposals']} ({success_rate:.1f}%)")
        print(f"   [SKIP] Skipped: {results['skipped']}")
        print(f"   [FAIL] Failed: {results['failed']}")
        print(f"   [VOTES] Total votes: {results['total_votes']:,}")
        print(f"   [TIME] Time: {elapsed:.1f}s")

        if results['failed_proposals']:
            print(f"\n[FAILED] Failed proposals:")
            for failed in results['failed_proposals'][:5]:  # show first 5 failures
                print(f"   {failed['proposal_id'][:10]}: {failed['error']}")
            if len(results['failed_proposals']) > 5:
                print(
                    f"   ... and {len(results['failed_proposals']) - 5} more")

        return results

    def _verify_consolidated_file(self, consolidated_path: Path, expected_votes: int, proposal_ids: list[str]) -> bool:
        """Verify that the consolidated offchain_votes.csv file is valid and complete."""
        try:
            if not consolidated_path.exists():
                print(
                    f"[VERIFY] Consolidated file does not exist: {consolidated_path}")
                return False

            # check file size
            file_size = consolidated_path.stat().st_size
            if file_size == 0:
                print(
                    f"[VERIFY] Consolidated file is empty: {consolidated_path}")
                return False

            # verify csv structure and count votes
            with open(consolidated_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                # check headers
                if not reader.fieldnames:
                    print(
                        f"[VERIFY] Missing CSV headers.")
                    return False
                # ensure base columns exist while allowing extra choice columns
                headers_set = set(reader.fieldnames)
                if not set(BASE_COLUMNS).issubset(headers_set):
                    print(
                        f"[VERIFY] Invalid CSV headers. Missing required base columns.")
                    return False

                # count votes and verify integrity
                vote_count = 0
                for row in reader:
                    vote_count += 1
                    # basic validation of required fields
                    if not row.get('id') or not row.get('voter'):
                        print(
                            f"[VERIFY] Invalid vote data at row {vote_count}")
                        return False

                if vote_count != expected_votes:
                    print(
                        f"[VERIFY] Vote count mismatch. Expected: {expected_votes}, Got: {vote_count}")
                    return False

            print(f"[VERIFY] Consolidated file verified successfully!")
            print(f"   Total votes: {vote_count:,}")
            print(f"   File size: {file_size:,} bytes")
            return True

        except Exception as e:
            print(f"[VERIFY] Verification failed: {e}")
            return False

    def _cleanup_temp_dir(self, temp_dir: Path) -> None:
        """Delete the temporary directory containing per-proposal parquet files."""
        try:
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)
                print(f"[CLEANUP] Deleted temp directory {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to delete temp directory {temp_dir}: {e}")

    def _consolidate_proposal_csvs(self, source_dir: Path, consolidated_path: Path, proposal_ids: list[str]) -> None:
        """Consolidate individual proposal parquet files from a temp dir into a single offchain_votes.csv and .parquet.
        Choice columns are computed dynamically as the union across inputs. Uses streaming Parquet writer to avoid high memory usage.
        """
        consolidated_start = time.time()
        total_consolidated_votes = 0

        # create consolidated parquet path
        consolidated_parquet_path = consolidated_path.with_suffix('.parquet')

        # first pass: determine dynamic choice columns union using parquet schema
        dynamic_choice_cols: set[str] = set()
        file_paths: list[Path] = []
        for proposal_id in proposal_ids:
            p = source_dir / f"{proposal_id}.parquet"
            if p.exists():
                file_paths.append(p)
                try:
                    schema = pq.read_schema(p)
                    for name in schema.names:
                        if isinstance(name, str) and name.startswith('choice.'):
                            dynamic_choice_cols.add(name)
                except Exception as e:
                    logger.warning(f"Failed reading schema from {p}: {e}")
                    continue

        # build final columns
        sorted_choice_cols = sorted(
            dynamic_choice_cols,
            key=lambda c: int(c.split('.', 1)[1]) if c.split(
                '.', 1)[1].isdigit() else 10**9
        )
        fieldnames = BASE_COLUMNS + sorted_choice_cols

        # build a stable target schema to avoid parquet schema mismatch
        # base column types stay permissive yet consistent
        base_schema_types: dict[str, pa.DataType] = {
            'app': pa.string(),
            'choice': pa.string(),
            'created': pa.int64(),
            'id': pa.string(),
            'ipfs': pa.string(),
            'metadata': pa.string(),
            'metadata.app': pa.string(),
            'proposal': pa.string(),
            'proposalId': pa.string(),
            'reason': pa.string(),
            'space': pa.string(),
            'voter': pa.string(),
            'vp': pa.float64(),
            'vp_by_strategy': pa.string(),
            'vp_state': pa.string(),
        }
        schema_fields: list[pa.Field] = []
        for col in BASE_COLUMNS:
            schema_fields.append(
                pa.field(col, base_schema_types.get(col, pa.string())))
        for col in sorted_choice_cols:
            schema_fields.append(pa.field(col, pa.float64()))
        target_schema = pa.schema(schema_fields)

        # second pass: write csv line by line and stream parquet via writer
        parquet_writer: pq.ParquetWriter | None = None
        try:
            with open(consolidated_path, 'w', newline='', encoding='utf-8') as consolidated_file:
                writer = csv.DictWriter(
                    consolidated_file, fieldnames=fieldnames)
                writer.writeheader()

                for i, p in enumerate(file_paths, 1):
                    try:
                        df = pd.read_parquet(p)
                        # normalize missing columns and order columns
                        missing_cols = [
                            c for c in fieldnames if c not in df.columns]
                        for c in missing_cols:
                            if c.startswith('choice.'):
                                # ensure numeric dtype with nan for arrow stability
                                df[c] = pd.Series(
                                    [float('nan')] * len(df), dtype='float64')
                            else:
                                df[c] = None
                        # coerce dynamic choice columns to float64
                        for c in sorted_choice_cols:
                            df[c] = pd.to_numeric(
                                df[c], errors='coerce').astype('float64')
                        # coerce vp to float where present
                        if 'vp' in df.columns:
                            df['vp'] = pd.to_numeric(
                                df['vp'], errors='coerce').astype('float64')
                        # reorder columns
                        df = df[fieldnames]

                        # stream parquet first to ensure schema consistency before csv append
                        table = pa.Table.from_pandas(df, preserve_index=False)
                        table = table.cast(target_schema)
                        if parquet_writer is None:
                            parquet_writer = pq.ParquetWriter(
                                consolidated_parquet_path, table.schema)
                        parquet_writer.write_table(table)

                        # append csv after successful parquet write to keep counts in sync
                        df.to_csv(consolidated_file, header=False, index=False)

                        votes_count = len(df)
                        total_consolidated_votes += votes_count
                        print(
                            f"\r[CONSOLIDATE] [{i:3d}/{len(file_paths)}] Consolidated {votes_count:,} votes from {p.stem[:10]}...", end="", flush=True)
                    except Exception as e:
                        logger.warning(f"Failed to consolidate {p.name}: {e}")
                        print(
                            f"\nWARNING: Failed to consolidate {p.stem[:10]}: {e}")
                        continue
        finally:
            if parquet_writer is not None:
                parquet_writer.close()

        consolidation_time = time.time() - consolidated_start
        print(
            f"\n[SUCCESS] Consolidation complete! {total_consolidated_votes:,} total votes in {consolidation_time:.1f}s")
        logger.info(
            f"Consolidated {total_consolidated_votes} votes from {len(file_paths)} proposals in {consolidation_time:.2f}s")

        # verify the consolidated file
        print(f"\n[VERIFY] Verifying consolidated file...")
        if self._verify_consolidated_file(consolidated_path, total_consolidated_votes, proposal_ids):
            self._cleanup_temp_dir(source_dir)
        else:
            print(f"[CLEANUP] Skipping cleanup due to verification failure")
            logger.warning(
                "Skipping cleanup of individual files due to verification failure")

    def fetch_votes_by_space(self, space_id: str, proposals_csv_path: str,
                             output_dir: str = None, resume: bool = False,
                             delay: float = 1.0, batch_size: int = 0,
                             start_from: int = 1, max_choices: int = 64,
                             start_after_timestamp: int = 0) -> dict[str, Any]:
        """fetch votes for all proposals from a proposals csv"""
        proposals_path = Path(proposals_csv_path)
        if not proposals_path.exists():
            raise FileNotFoundError(
                f"Proposals CSV not found: {proposals_path}")

        # read proposals csv to extract unique proposal ids
        print(f"READING: Reading proposals from: {proposals_path}")
        proposal_ids = set()  # ensure uniqueness

        with open(proposals_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                proposal_ids.add(row['id'])

        proposal_ids = list(proposal_ids)  # convert back to list
        print(
            f"FOUND: Found {len(proposal_ids)} unique proposals in {space_id}")

        # setup progress tracking
        results = {
            'total_proposals': len(proposal_ids),
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'total_votes': 0,
            'start_time': time.time(),
            'failed_proposals': []
        }

        # set default output directory if not provided
        if output_dir is None:
            output_dir = f"data/{space_id}/governance_data"

        output_path = Path(output_dir)
        # use a unique per-run temp dir to avoid clashes
        run_suffix = f"{int(time.time())}_{os.getpid()}"
        temp_dir = output_path / f"_tmp_votes_{run_suffix}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # determine starting index for processing
        start_index = max(0, start_from - 1)
        cancelled = False
        for i in range(start_index, len(proposal_ids)):
            proposal_id = proposal_ids[i]
            # check if already exists (resume support)
            proposal_parquet_path = temp_dir / f"{proposal_id}.parquet"
            if resume and proposal_parquet_path.exists():
                results['skipped'] += 1
                print(
                    f"SKIP: [{i+1:3d}/{len(proposal_ids)}] Skipping {proposal_id[:10]}... (already exists)")
                continue

            try:
                print(
                    f"\nPROCESSING: [{i+1:3d}/{len(proposal_ids)}] Processing {proposal_id[:10]}...")

                # fetch votes for this proposal
                votes = self.fetch_all_votes(
                    proposal_id, start_after_timestamp=start_after_timestamp)
                results['total_votes'] += len(votes)

                # export votes
                exporter = VoteCSVExporter(max_choices=max_choices)
                if batch_size > 0:
                    # use batch export with proposal id as prefix
                    batch_output_dir = output_path / proposal_id[:10]
                    exporter.export_batch_to_csv(
                        votes, batch_output_dir, batch_size, f"votes_{proposal_id[:10]}")
                    # also export parquet batches
                    exporter.export_batch_to_parquet(
                        votes, batch_output_dir, batch_size, f"votes_{proposal_id[:10]}")
                else:
                    # single parquet file per proposal (no individual csv)
                    parquet_path = temp_dir / f"{proposal_id}.parquet"
                    # ensure parent directory exists if removed externally
                    parquet_path.parent.mkdir(parents=True, exist_ok=True)
                    exporter.export_to_parquet(votes, parquet_path)

                results['processed'] += 1

                # rate limiting delay to avoid overwhelming the api
                # only delay if not last iteration
                if delay > 0 and i < len(proposal_ids) - 1:
                    time.sleep(delay)

            except KeyboardInterrupt:
                print("\n[CANCELLED] Vote fetching interrupted by user")
                logger.info(
                    "Vote fetching interrupted by user via KeyboardInterrupt")
                cancelled = True
                break
            except Exception as e:
                results['failed'] += 1
                results['failed_proposals'].append(
                    {'proposal_id': proposal_id, 'error': str(e)})
                print(f"FAILED: Failed to process {proposal_id[:10]}: {e}")
                logger.error(f"Failed to process {proposal_id}: {e}")
                continue

        # final summary
        elapsed = time.time() - results['start_time']
        success_rate = (results['processed'] / results['total_proposals']
                        ) * 100 if results['total_proposals'] > 0 else 0

        print(f"\n[COMPLETE] Space processing complete!")
        print(
            f"   [STATS] Processed: {results['processed']}/{results['total_proposals']} ({success_rate:.1f}%)")
        print(f"   [SKIP] Skipped: {results['skipped']}")
        print(f"   [FAIL] Failed: {results['failed']}")
        print(f"   [VOTES] Total votes: {results['total_votes']:,}")
        print(f"   [TIME] Time: {elapsed:.1f}s")
        if cancelled:
            print(
                "   [STATUS] Run cancelled; partial results saved where applicable")

        if results['failed_proposals']:
            print(f"\n[FAILED] Failed proposals:")
            for failed in results['failed_proposals'][:5]:  # show first 5 failures
                print(f"   {failed['proposal_id'][:10]}: {failed['error']}")
            if len(results['failed_proposals']) > 5:
                print(
                    f"   ... and {len(results['failed_proposals']) - 5} more")

        # consolidate individual proposal csv files
        if results['processed'] > 0:
            print(
                f"\n[CONSOLIDATE] Consolidating {results['processed']} proposal files into votes.csv...")
            consolidated_votes_path = output_path / "offchain_votes.csv"
            self._consolidate_proposal_csvs(
                temp_dir, consolidated_votes_path, proposal_ids)
            print(
                f"[SUCCESS] Consolidated votes saved to: {consolidated_votes_path}")

        return results


class VoteCSVExporter:
    """snapshot vote exporter with choice column normalization"""

    def __init__(self, max_choices: int = 64):
        self.max_choices = max_choices

    def _derive_choice_columns(self, votes: List[Vote]) -> List[str]:
        keys: set[int] = set()
        for v in votes:
            c = v.choice
            if isinstance(c, dict):
                for k in c.keys():
                    try:
                        keys.add(int(k))
                    except Exception:
                        continue
            elif isinstance(c, (int, float)):
                try:
                    keys.add(int(c))
                except Exception:
                    continue
        sorted_keys = sorted(keys)
        if self.max_choices > 0:
            sorted_keys = sorted_keys[: self.max_choices]
        return [f"choice.{k}" for k in sorted_keys]

    def _build_columns(self, votes: List[Vote]) -> List[str]:
        choice_cols = self._derive_choice_columns(votes)
        return BASE_COLUMNS + choice_cols

    def get_output_path(self, space_name: str, filename: str = "offchain_votes.csv", base_dir: str = "output") -> Path:
        """build standardized output path"""
        # sanitize space name for filesystem
        safe_space = space_name.replace('/', '_').replace('\\', '_')
        return Path(base_dir) / safe_space / filename

    def export_to_csv(self, votes: List[Vote], output_path: str | Path) -> None:
        """export votes to csv"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        export_start = time.time()
        print(
            f"\n[EXPORT] Exporting {len(votes):,} votes to {output_path.name}...")
        logger.info(f"Exporting {len(votes)} votes to {output_path}")

        columns = self._build_columns(votes)
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()

            for vote in votes:
                csv_row = vote.to_csv_row()
                # ensure all required columns are present
                row_data = {col: csv_row.get(col) for col in columns}
                writer.writerow(row_data)

        export_time = time.time() - export_start
        print(f"[SUCCESS] Export complete! ({export_time:.1f}s)")
        logger.info(
            f"Successfully exported {len(votes)} votes to {output_path}")

    def export_batch_to_csv(self, votes: List[Vote], output_dir: str | Path,
                            batch_size: int = 10000, prefix: str = "votes") -> List[Path]:
        """export votes to csv in batches"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        batch_start = time.time()
        num_batches = (len(votes) + batch_size - 1) // batch_size
        print(
            f"\n[BATCH] Exporting {len(votes):,} votes in {num_batches} batches to {output_dir}...")

        output_files = []

        for i in range(0, len(votes), batch_size):
            batch = votes[i:i + batch_size]
            batch_num = i // batch_size + 1
            output_file = output_dir / f"{prefix}_batch_{batch_num:04d}.csv"

            batch_progress = f"[{batch_num}/{num_batches}]"
            print(
                f"\r[EXPORT] {batch_progress} Exporting batch {batch_num}...", end="", flush=True)

            # temporarily disable the individual export progress for batch mode
            original_print = print

            def silent_print(*args, **kwargs):
                if not (args and '[EXPORT]' in str(args[0])):
                    original_print(*args, **kwargs)

            import builtins
            builtins.print = silent_print
            self.export_to_csv(batch, output_file)
            builtins.print = original_print

            output_files.append(output_file)

        batch_time = time.time() - batch_start
        print(
            f"\n[SUCCESS] Batch export complete! {len(output_files)} files created in {batch_time:.1f}s")
        logger.info(
            f"Exported {len(votes)} votes across {len(output_files)} batch files")
        return output_files

    def export_to_parquet(self, votes: List[Vote], output_path: str | Path) -> None:
        """export votes to parquet"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        export_start = time.time()
        print(
            f"\n[EXPORT] Exporting {len(votes):,} votes to {output_path.name}...")
        logger.info(f"Exporting {len(votes)} votes to {output_path}")

        df = pd.DataFrame([vote.to_csv_row() for vote in votes])
        df.to_parquet(output_path, index=False, engine='pyarrow')

        export_time = time.time() - export_start
        print(f"[SUCCESS] Export complete! ({export_time:.1f}s)")
        logger.info(
            f"Successfully exported {len(votes)} votes to {output_path}")

    def export_batch_to_parquet(self, votes: List[Vote], output_dir: str | Path,
                                batch_size: int = 10000, prefix: str = "votes") -> List[Path]:
        """export votes to parquet in batches"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        batch_start = time.time()
        num_batches = (len(votes) + batch_size - 1) // batch_size
        print(
            f"\n[BATCH] Exporting {len(votes):,} votes in {num_batches} batches to {output_dir}...")

        output_files = []

        for i in range(0, len(votes), batch_size):
            batch = votes[i:i + batch_size]
            batch_num = i // batch_size + 1
            output_file = output_dir / \
                f"{prefix}_batch_{batch_num:04d}.parquet"

            batch_progress = f"[{batch_num}/{num_batches}]"
            print(
                f"\r[EXPORT] {batch_progress} Exporting batch {batch_num}...", end="", flush=True)

            # temporarily disable the individual export progress for batch mode
            original_print = print

            def silent_print(*args, **kwargs):
                if not (args and '[EXPORT]' in str(args[0])):
                    original_print(*args, **kwargs)

            import builtins
            builtins.print = silent_print
            self.export_to_parquet(batch, output_file)
            builtins.print = original_print

            output_files.append(output_file)

        batch_time = time.time() - batch_start
        print(
            f"\n[SUCCESS] Batch export complete! {len(output_files)} files created in {batch_time:.1f}s")
        logger.info(
            f"Exported {len(votes)} votes across {len(output_files)} batch files")
        return output_files


def setup_logging(verbose: bool = False):
    """configure logging output"""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def main():
    """cli entry point for snapshot vote fetcher"""
    parser = argparse.ArgumentParser(
        description="Fetch Snapshot votes (unified flags)")
    parser.add_argument(
        'dao_id', help='DAO/space ID (e.g., uniswapgovernance.eth)')
    parser.add_argument('--proposal-id', dest='proposal_id', default=None,
                        help='Snapshot proposal ID (fetch only this proposal)')
    parser.add_argument('--test', action='store_true',
                        help='Test mode: pick first proposal with transfers and fetch only it')
    parser.add_argument('--output', '-o', default=None,
                        help='Output CSV file path (default: data/{dao}/governance_data/offchain_votes.csv)')
    parser.add_argument('--batch-size', type=int, default=0,
                        help='Split output into batches (0 = single file)')
    parser.add_argument('--resume', action='store_true',
                        help='Skip proposals already processed (bulk)')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Delay between proposals in seconds (bulk)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--api-key', type=str,
                        help='Snapshot API key for higher rate limits')
    parser.add_argument('--start-from-timestamp', type=int, default=0,
                        help='Seed the created cursor with a unix timestamp (created_gt)')
    parser.add_argument('--max-choices', type=int, default=64,
                        help='Maximum number of dynamic choice.* columns to emit (0 = unlimited)')

    args = parser.parse_args()

    setup_logging(args.verbose)

    # initialize fetcher
    exporter = VoteCSVExporter(max_choices=args.max_choices)

    try:
        with SnapshotVoteFetcher(api_key=args.api_key) as fetcher:
            # unified flow
            # 1) If proposal_id provided, fetch that single proposal
            if args.proposal_id:
                votes = fetcher.fetch_all_votes(
                    args.proposal_id, start_after_timestamp=args.start_from_timestamp)
            # determine output path
                if args.output is None:
                    space_name = fetcher.get_proposal_space(args.proposal_id)
                    output_path = exporter.get_output_path(space_name)
                    print(f"[AUTO_SAVE] Auto-saving to: {output_path}")
                else:
                    output_path = Path(args.output)

                if args.batch_size > 0:
                    output_dir = output_path.parent
                    prefix = output_path.stem
                    exporter.export_batch_to_csv(
                        votes, output_dir, args.batch_size, prefix)
                    exporter.export_batch_to_parquet(
                        votes, output_dir, args.batch_size, prefix)
                else:
                    exporter.export_to_csv(votes, output_path)
                    parquet_path = output_path.with_suffix('.parquet')
                    exporter.export_to_parquet(votes, parquet_path)
                print(f"[SUCCESS] Fetched and exported {len(votes)} votes")

            else:
                # 2) Otherwise, bulk by dao_id (space)
                proposals_csv_path = f"data/{args.dao_id}/governance_data/offchain_proposals.csv"
                output_dir = f"data/{args.dao_id}/governance_data"

            # if test mode pick first proposal with transfers
            if args.test:
                from pathlib import Path
                transfers = Path(__file__).resolve(
                ).parents[2] / 'data' / args.dao_id / 'governance_transfer_data' / 'offchain_matched_transfers.csv'
                target = None
                if transfers.exists():
                    with open(transfers, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            pid = row.get('proposal_id')
                            if pid:
                                target = pid
                                break
                if target:
                    print(f"[TEST] Using proposal with transfers: {target}")
                    votes = fetcher.fetch_all_votes(
                        target, start_after_timestamp=args.start_from_timestamp)
                    # save outputs
                    space_name = fetcher.get_proposal_space(target)
                    output_path = exporter.get_output_path(space_name)
                    if args.batch_size > 0:
                        exporter.export_batch_to_csv(
                            votes, output_path.parent, args.batch_size, output_path.stem)
                        exporter.export_batch_to_parquet(
                            votes, output_path.parent, args.batch_size, output_path.stem)
                    else:
                        exporter.export_to_csv(votes, output_path)
                        exporter.export_to_parquet(
                            votes, output_path.with_suffix('.parquet'))
                    print(
                        f"[SUCCESS] Fetched and exported {len(votes)} votes (test mode)")
                else:
                    print(
                        "[WARNING] No transfers found; running minimal bulk of first N proposals")
                    results = fetcher.fetch_votes_by_space(
                        args.dao_id, proposals_csv_path, output_dir, resume=args.resume, delay=args.delay, batch_size=args.batch_size, start_from=1, max_choices=args.max_choices, start_after_timestamp=args.start_from_timestamp)
            else:
                results = fetcher.fetch_votes_by_space(
                    args.dao_id, proposals_csv_path, output_dir, resume=args.resume, delay=args.delay, batch_size=args.batch_size, start_from=1, max_choices=args.max_choices, start_after_timestamp=args.start_from_timestamp)

    except KeyboardInterrupt:
        print("\n[CANCELLED] Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        return 1

    return 0


# example usage as a library
def example_usage():
    """illustrative example for library usage"""
    # setup logging
    setup_logging(verbose=True)

    # example proposal id (replace with actual proposal)
    proposal_id = "0x123abc..."

    # initialize fetcher optionally with api key
    fetcher = SnapshotVoteFetcher(api_key=None)  # or api_key="your_key_here"
    exporter = VoteCSVExporter()

    try:
        # fetch all votes
        print(f"STARTING: Fetching votes for proposal: {proposal_id}")
        votes = fetcher.fetch_all_votes(proposal_id)

        # validate votes
        validation = fetcher.validate_votes(votes)
        print(f"Validation results: {validation}")

        # export to csv
        output_file = "votes.csv"
        exporter.export_to_csv(votes, output_file)
        # also export to parquet
        parquet_file = "votes.parquet"
        exporter.export_to_parquet(votes, parquet_file)
        print(
            f"[SUCCESS] Exported {len(votes)} votes to {output_file} and {parquet_file}")

    except Exception as e:
        print(f"[ERROR] Error: {e}")


if __name__ == "__main__":
    main()
