#!/usr/bin/env python3
"""fetch snapshot proposals for a space and export to csv or parquet"""

try:
    from data_collection_config import ENDPOINT_SNAPSHOT, PAGE_LIMIT, RETRY_WAIT
except Exception:
    # fallback defaults when local config is unavailable
    ENDPOINT_SNAPSHOT = "https://hub.snapshot.org/graphql"
    PAGE_LIMIT = 100
    RETRY_WAIT = 5
import logging
import httpx
import backoff
from email.utils import parsedate_to_datetime
import csv
import json
import time
import pandas as pd
from typing import Dict, Any, List, Generator, Optional, Set, Tuple
import sys
import os
from pathlib import Path

# ensure env vars are loaded from project root
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    def load_dotenv(*args, **kwargs):  # type: ignore
        return False

# detect project root using requirements.txt
_THIS_DIR = Path(__file__).resolve().parent
_CANDIDATES = [_THIS_DIR, *_THIS_DIR.parents]
_DETECTED_ROOT: Optional[Path] = None
for _cand in _CANDIDATES:
    if (_cand / "requirements.txt").exists():
        _DETECTED_ROOT = _cand
        break
if _DETECTED_ROOT is None:
    _DETECTED_ROOT = _THIS_DIR.parent  # fallback to repo root guess (../)

# load .env or env from detected project root
for _env_name in (".env", "env"):
    _env_path = _DETECTED_ROOT / _env_name
    if _env_path.exists():
        load_dotenv(dotenv_path=str(_env_path), override=False)
        break

# intentionally avoid sys.path manipulation and rely on detected root


# quality filtering and api configuration
MAX_RETRIES = 3
REQUEST_TIMEOUT = 120
TEST_MODE_MAX_PAGES = 10
REQS_PER_MIN = 58  # leave a small buffer under 60 rpm

# progress tracking intervals
PROGRESS_LOG_INTERVAL = 5   # log progress every n pages
PERFORMANCE_LOG_INTERVAL = 25  # log performance every n pages
REALTIME_PROGRESS_INTERVAL = 1  # log simple progress every n pages


class ProgressTracker:
    """track progress and performance while collecting proposals"""

    def __init__(self, operation_name: str = "Proposals Collection", space_id: str = ""):
        self.operation_name = operation_name
        self.space_id = space_id
        self.start_time = time.time()
        self.page_count = 0
        self.total_proposals = 0
        self.query_times = []
        self.rate_limit_count = 0
        self.error_count = 0
        self.last_progress_log = 0
        self.last_performance_log = 0

    def log_page_completion(self, proposals_count: int, query_time: float = None):
        """log completion of a page with performance metrics"""
        self.page_count += 1
        self.total_proposals += proposals_count

        if query_time is not None:
            self.query_times.append(query_time)
            if len(self.query_times) > 100:
                self.query_times = self.query_times[-100:]

        # real time progress indicator
        if self.page_count % REALTIME_PROGRESS_INTERVAL == 0:
            self._log_realtime_progress()

        # detailed progress at regular intervals
        if self.page_count % PROGRESS_LOG_INTERVAL == 0:
            self._log_progress()

        # log performance metrics at regular intervals
        if self.page_count % PERFORMANCE_LOG_INTERVAL == 0:
            self._log_performance()

    def log_rate_limit(self):
        """log a rate limiting event"""
        self.rate_limit_count += 1
        print(
            f"[rate_limit] rate limited {self.rate_limit_count} times so far")

    def log_error(self):
        """log an error event"""
        self.error_count += 1
        print(f"[error] error count: {self.error_count}")

    def _log_realtime_progress(self):
        """log a simple real time progress indicator"""
        elapsed_time = time.time() - self.start_time
        proposals_per_sec = self.total_proposals / \
            elapsed_time if elapsed_time > 0 else 0

        # simple one line progress indicator
        print(
            f"[progress] page {self.page_count} | {self.total_proposals:,} proposals | {proposals_per_sec:.1f}/sec | {self._format_duration(elapsed_time)}")

    def _log_progress(self):
        """log current progress statistics"""
        elapsed_time = time.time() - self.start_time
        avg_time_per_page = elapsed_time / self.page_count if self.page_count > 0 else 0

        print(
            f"[progress] {self.operation_name} ({self.space_id}) - page {self.page_count}:")
        print(f"  - total proposals collected: {self.total_proposals:,}")
        print(
            f"  - average proposals per page: {self.total_proposals / self.page_count:.1f}")
        print(f"  - elapsed time: {self._format_duration(elapsed_time)}")
        print(
            f"  - average time per page: {self._format_duration(avg_time_per_page)}")

        if self.rate_limit_count > 0:
            print(f"  - rate limit events: {self.rate_limit_count}")
        if self.error_count > 0:
            print(f"  - error events: {self.error_count}")

    def _log_performance(self):
        """log detailed performance metrics"""
        if not self.query_times:
            return

        avg_query_time = sum(self.query_times) / len(self.query_times)
        min_query_time = min(self.query_times)
        max_query_time = max(self.query_times)

        print(
            f"[performance] query performance (last {len(self.query_times)} queries):")
        print(f"  - average query time: {avg_query_time:.2f}s")
        print(f"  - min query time: {min_query_time:.2f}s")
        print(f"  - max query time: {max_query_time:.2f}s")
        print(f"  - total rate limit events: {self.rate_limit_count}")
        print(f"  - total error events: {self.error_count}")

    def final_summary(self):
        """print the final summary of the run"""
        total_time = time.time() - self.start_time

        print(f"\n[final summary] {self.operation_name} ({self.space_id})")
        print("=" * 60)
        print(f"total pages processed: {self.page_count}")
        print(f"total proposals collected: {self.total_proposals:,}")
        print(f"total time: {self._format_duration(total_time)}")
        print(
            f"average time per page: {self._format_duration(total_time / self.page_count) if self.page_count > 0 else 'n/a'}")
        print(
            f"average proposals per page: {self.total_proposals / self.page_count:.1f}" if self.page_count > 0 else "n/a")

        if self.query_times:
            avg_query_time = sum(self.query_times) / len(self.query_times)
            print(f"average query time: {avg_query_time:.2f}s")

        if self.rate_limit_count > 0:
            print(f"rate limit events: {self.rate_limit_count}")
        if self.error_count > 0:
            print(f"error events: {self.error_count}")
        print("=" * 60)

    def _format_duration(self, seconds: float) -> str:
        """format duration into a simple human readable string"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"


# optional snapshot api key
SNAPSHOT_API_KEY = os.getenv("SNAPSHOT_API_KEY")

# clamp page_limit to snapshot ceiling and validate
try:
    PAGE_LIMIT = min(int(PAGE_LIMIT), 100)
    if PAGE_LIMIT <= 0:
        PAGE_LIMIT = 100
except Exception:
    PAGE_LIMIT = 100


logger = logging.getLogger(__name__)


class SnapshotHttpClient:
    """http client with http/2, backoff, rate limiting, and retry-after handling"""

    def __init__(self, endpoint: str, api_key: Optional[str] = None):
        self.endpoint = endpoint
        self.client = httpx.Client(http2=True, timeout=REQUEST_TIMEOUT)
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["X-Api-Key"] = api_key
        self.reqs_per_min = REQS_PER_MIN
        self.last_request_time = 0.0

    def _respect_rate_limit(self):
        elapsed = time.time() - self.last_request_time
        min_interval = 60.0 / float(self.reqs_per_min)
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

    def _parse_retry_after_seconds(self, header_value: Optional[str]) -> float:
        if not header_value:
            return float(RETRY_WAIT)
        try:
            return float(header_value)
        except Exception:
            pass
        try:
            dt = parsedate_to_datetime(header_value)
            now_ts = time.time()
            return max(0.0, dt.timestamp() - now_ts)
        except Exception:
            return float(RETRY_WAIT)

    @backoff.on_exception(
        backoff.expo,
        (httpx.HTTPStatusError, httpx.RequestError, RuntimeError),
        max_tries=5,
        jitter=backoff.full_jitter,
        giveup=lambda e: hasattr(e, 'response') and getattr(
            e.response, 'status_code', 0) not in (429, 500, 502, 503, 504)
    )
    def post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._respect_rate_limit()
        response = self.client.post(
            self.endpoint, headers=self.headers, json=payload)
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            wait = self._parse_retry_after_seconds(retry_after)
            logger.warning(
                f"rate limited (429). waiting {wait:.1f}s before retry...")
            time.sleep(wait)
            response.raise_for_status()
        response.raise_for_status()
        self.last_request_time = time.time()
        data = response.json()
        if 'errors' in data:
            logger.error(f"graphql errors: {data['errors']}")
            raise RuntimeError(f"graphql errors: {data['errors']}")
        return data.get('data', {})


# lean and enhanced graphql queries
PROPOSALS_QUERY_MIN_BASE = """
query ($space:String!, $first:Int!, $after:Int!{state_var}){{
  proposals(
    first: $first,
    where: {{ space: $space{state_where}, created_gt: $after }},
    orderBy: "created",
    orderDirection: asc
  ){{
    id ipfs space {{ id }} author created network title start end snapshot type state votes scores_total scores_state scores_updated plugins
  }}
}}
"""

PROPOSALS_QUERY_ENHANCED_BASE = """
query ($space:String!, $first:Int!, $after:Int!{state_var}){{
  proposals(
    first: $first,
    where: {{ space: $space{state_where}, created_gt: $after }},
    orderBy: "created",
    orderDirection: asc
  ){{
    id ipfs space {{ id }} author created network title body discussion choices privacy app
    start end snapshot type quorum state flagged votes scores scores_by_strategy scores_total scores_state scores_updated symbol
    strategies {{ name network params }} validation {{ name params }} plugins
  }}
}}
"""


def build_query(enhanced: bool, include_state: bool) -> str:
    base = PROPOSALS_QUERY_ENHANCED_BASE if enhanced else PROPOSALS_QUERY_MIN_BASE
    state_var = ", $state:String!" if include_state else ""
    state_where = ", state: $state" if include_state else ""
    return base.format(state_var=state_var, state_where=state_where)


def execute_graphql_query(query: str, variables: Dict[str, Any], endpoint: str = ENDPOINT_SNAPSHOT, progress_tracker: ProgressTracker = None) -> tuple[Dict[str, Any], float]:
    """run a snapshot graphql query with retry aware client"""
    start_time = time.time()
    client = SnapshotHttpClient(endpoint=endpoint, api_key=SNAPSHOT_API_KEY)
    try:
        data = client.post({"query": query, "variables": variables})
    except Exception as e:
        if progress_tracker:
            progress_tracker.log_error()
        raise e
    query_time = time.time() - start_time
    return data, query_time


def flatten_proposal_data(proposal: Dict[str, Any]) -> Dict[str, Any]:
    """flatten a proposal into the optimized schema"""
    # top level fields maintained for filtering
    flattened = {
        'id': proposal.get('id', ''),
        'ipfs': proposal.get('ipfs', ''),
        'space': proposal.get('space', {}).get('id', '') if proposal.get('space') else '',
        'author': proposal.get('author', ''),
        'network': proposal.get('network', ''),
        'created': proposal.get('created', 0),
        'start': proposal.get('start', 0),
        'end': proposal.get('end', 0),
        'snapshot': str(proposal.get('snapshot', '')),
        'type': proposal.get('type', ''),
        'quorum': proposal.get('quorum', 0),
        'state': proposal.get('state', ''),
        'flagged': proposal.get('flagged', False),
        'votes': proposal.get('votes', 0),
        'scores_total': proposal.get('scores_total', 0),
        'scores_state': proposal.get('scores_state', ''),
        'scores_updated': proposal.get('scores_updated', 0)
    }

    # metadata fields stored in json for compactness
    # Always includes plugins for complete data collection
    metadata = {
        'title': proposal.get('title', ''),
        'body': proposal.get('body', ''),
        'discussion': proposal.get('discussion', ''),
        'choices': proposal.get('choices', []),
        'privacy': proposal.get('privacy', ''),
        'symbol': proposal.get('symbol', ''),
        'app': proposal.get('app', ''),
        'scores': proposal.get('scores', []),
        'scores_by_strategy': proposal.get('scores_by_strategy', []),
        'strategies': proposal.get('strategies', []),
        'validation': proposal.get('validation', {}),
        'plugins': proposal.get('plugins', {})
    }

    # convert metadata to json string
    flattened['metadata'] = json.dumps(metadata, ensure_ascii=False)

    return flattened


def fetch_proposals_test_mode(space_id: str, enhanced: bool, state: Optional[str]) -> List[Dict[str, Any]]:
    """fetch a limited set of proposals for testing"""
    all_proposals = []
    max_pages = TEST_MODE_MAX_PAGES
    progress_tracker = ProgressTracker(
        "Test Mode Proposals Collection", space_id)

    print(
        f"[progress] starting test mode for space '{space_id}' - fetching first {max_pages} pages...")

    query = build_query(enhanced=enhanced, include_state=bool(state))

    cursor = 0
    seen_ids: Set[str] = set()
    for page in range(max_pages):
        print(
            f"[progress] fetching test page {page + 1}/{max_pages} (after: {cursor})...")

        variables = {
            "space": space_id,
            "first": PAGE_LIMIT,
            "after": int(cursor),
        }
        if state:
            variables["state"] = state

        try:
            print(
                f"[query] executing graphql query for test page {page + 1}/{max_pages}...")
            data, query_time = execute_graphql_query(
                query, variables, ENDPOINT_SNAPSHOT, progress_tracker)
            proposals = data.get("proposals", [])
            print(f"[success] query completed in {query_time:.2f}s")

            if not proposals:
                print(
                    f"[progress] no proposals found on page {page + 1}, stopping")
                progress_tracker.log_page_completion(0, query_time)
                break

            # de-duplicate within the run in case of equal timestamps
            new_batch = [p for p in proposals if p.get("id") not in seen_ids]
            for p in new_batch:
                pid = p.get("id")
                if pid:
                    seen_ids.add(pid)

            all_proposals.extend(new_batch)
            progress_tracker.log_page_completion(len(new_batch), query_time)
            print(
                f"[progress] found {len(new_batch)} new proposals on page {page + 1} (total: {len(all_proposals)})")

            last_created = proposals[-1].get("created", cursor) or cursor
            try:
                cursor = max(0, int(last_created) - 1)
            except Exception:
                cursor = int(cursor)

            if len(proposals) < PAGE_LIMIT:
                print("[progress] reached end of data (partial page received)")
                break

        except Exception as e:
            print(f"[error] failed to fetch page {page + 1}: {e}")
            progress_tracker.log_error()
            break

    progress_tracker.final_summary()
    print(
        f"[success] test mode: fetched {len(all_proposals)} proposals for space '{space_id}'")
    return all_proposals


def fetch_all_proposals(space_id: str, enhanced: bool, state: Optional[str], stop_when_seen_ids_on_first_page: Optional[Set[str]] = None, initial_cursor: Optional[int] = None) -> List[Dict[str, Any]]:
    """fetch all proposals for a space with pagination"""
    all_proposals: List[Dict[str, Any]] = []
    page_count = 0
    cursor: int = int(initial_cursor) if initial_cursor is not None else 0
    seen_ids: Set[str] = set()
    progress_tracker = ProgressTracker("Full Proposals Collection", space_id)

    print(
        f"[progress] starting to fetch all proposals for space '{space_id}'...")

    query = build_query(enhanced=enhanced, include_state=bool(state))

    while True:
        page_count += 1
        print(
            f"[progress] fetching proposals page {page_count} (after: {cursor})...")

        variables = {
            "space": space_id,
            "first": PAGE_LIMIT,
            "after": int(cursor),
        }
        if state:
            variables["state"] = state

        try:
            print(f"[query] executing graphql query for page {page_count}...")
            data, query_time = execute_graphql_query(
                query, variables, ENDPOINT_SNAPSHOT, progress_tracker)
            proposals = data.get("proposals", [])
            print(f"[success] query completed in {query_time:.2f}s")

            if not proposals:
                print("[progress] no more proposals found, pagination complete")
                progress_tracker.log_page_completion(0, query_time)
                break

            # de-duplicate and drop already seen ids when incremental
            page_new: List[Dict[str, Any]] = []
            for p in proposals:
                pid = p.get("id")
                if not pid:
                    continue
                if stop_when_seen_ids_on_first_page and page_count == 1 and pid in stop_when_seen_ids_on_first_page:
                    # skip items already present in existing csv when running incremental
                    continue
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)
                page_new.append(p)

            # stop immediately if incremental finds no new items on first page
            if stop_when_seen_ids_on_first_page and page_count == 1 and not page_new:
                progress_tracker.log_page_completion(0, query_time)
                print(
                    "[progress] incremental: first page contained only already-seen proposals; stopping")
                break

            all_proposals.extend(page_new)
            progress_tracker.log_page_completion(len(page_new), query_time)
            print(
                f"[progress] found {len(page_new)} new proposals on page {page_count} (total: {len(all_proposals)})")

            # check for partial final page
            if len(proposals) < PAGE_LIMIT:
                print("[progress] reached end of data (partial page received)")
                break

            # advance cursor using last created timestamp with small rollback
            last_created = proposals[-1].get("created")
            try:
                cursor = max(0, int(last_created) -
                             1 if last_created is not None else int(cursor))
            except Exception:
                cursor = int(cursor)

        except Exception as e:
            print(f"[error] failed to fetch page {page_count}: {e}")
            progress_tracker.log_error()
            break

    progress_tracker.final_summary()
    print(
        f"[success] fetched {len(all_proposals)} total proposals for space '{space_id}'")
    return all_proposals


def _flatten_proposals_generator(proposals: List[Dict[str, Any]]) -> Generator[Dict[str, Any], None, None]:
    """yield flattened proposals one at a time"""
    for proposal in proposals:
        yield flatten_proposal_data(proposal)


def write_proposals_to_csv(proposals: List[Dict[str, Any]], output_path: str) -> None:
    """write proposals to csv using the optimized schema"""
    if not proposals:
        print("[warning] no proposals data to write")
        return

    # ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # fixed optimized schema order
    fieldnames = [
        'id', 'ipfs', 'space', 'author', 'network', 'created', 'start', 'end', 'snapshot',
        'type', 'quorum', 'state', 'flagged', 'votes', 'scores_total', 'scores_state', 'scores_updated',
        'metadata'
    ]

    print(f"[progress] writing {len(proposals)} proposals to {output_path}...")

    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # use generator for memory efficient processing
        for flattened_proposal in _flatten_proposals_generator(proposals):
            writer.writerow(flattened_proposal)

    print(
        f"[success] successfully wrote {len(proposals)} proposals to {output_path}")


def write_proposals_to_parquet(proposals: List[Dict[str, Any]], output_path: str) -> None:
    """write proposals to parquet in chunks"""
    if not proposals:
        print("[warning] no proposals data to write")
        return

    # ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # stream to parquet in chunks using pyarrow
    import pyarrow as pa
    import pyarrow.parquet as pq

    print(f"[progress] writing {len(proposals)} proposals to {output_path}...")

    # fixed column order schema
    columns = [
        'id', 'ipfs', 'space', 'author', 'network', 'created', 'start', 'end', 'snapshot',
        'type', 'quorum', 'state', 'flagged', 'votes', 'scores_total', 'scores_state', 'scores_updated',
        'metadata'
    ]

    # prepare chunks
    chunk_size = 50000
    writer = None
    buffer: List[Dict[str, Any]] = []

    def flush_buffer(buf: List[Dict[str, Any]], wr):
        if not buf:
            return wr
        norm = [{col: row.get(col) for col in columns} for row in buf]
        table = pa.Table.from_pylist(norm)
        if wr is None:
            wr = pq.ParquetWriter(output_path, table.schema)
        wr.write_table(table)
        return wr

    for proposal in proposals:
        buffer.append(flatten_proposal_data(proposal))
        if len(buffer) >= chunk_size:
            writer = flush_buffer(buffer, writer)
            buffer = []

    writer = flush_buffer(buffer, writer)
    if writer is not None:
        writer.close()
    else:
        # fallback for very small datasets
        df = pd.DataFrame([flatten_proposal_data(p)
                          for p in proposals], columns=columns)
        df.to_parquet(output_path, index=False)

    print(
        f"[success] successfully wrote {len(proposals)} proposals to {output_path}")


def main() -> None:
    """cli entry point for offchain proposal export"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch Snapshot proposals for a DAO (unified flags)")
    parser.add_argument(
        "dao_id", help="DAO/space identifier (e.g., 'uniswapgovernance.eth')")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: select first proposal with transfers and export only it")
    parser.add_argument("--enhanced", action="store_true",
                        help="Include heavy fields (body, strategies, plugins, etc.)")
    parser.add_argument("--state", choices=["pending", "open", "active", "closed"],
                        help="Filter proposals by state")
    parser.add_argument("--incremental", action="store_true",
                        help="Incremental mode: stop when an already-seen id appears on first page; merges with existing CSV/Parquet")
    parser.add_argument(
        "--output-slug",
        default=None,
        help="Optional output directory name (e.g., 'aave'). Defaults to using the DAO ID (space id).",
    )
    parser.add_argument("--csv-only", action="store_true",
                        help="Only write CSV output")
    parser.add_argument("--parquet-only", action="store_true",
                        help="Only write Parquet output")
    parser.add_argument("--start-after-created", type=int, default=0,
                        help="Seed the created cursor with a unix timestamp (created_gt)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging (DEBUG)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Reduce logging output (ERROR)")

    args = parser.parse_args()

    # configure logging: default INFO; --verbose -> DEBUG; --quiet -> ERROR
    if args.verbose:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.ERROR
    else:
        log_level = logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # calculate output paths under data/<dao>/governance_data
    project_root = os.path.abspath(os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir))
    out_folder = (args.output_slug or args.dao_id or "").strip().lower()
    out_folder = out_folder.replace('/', '_').replace('\\', '_')
    output_dir = os.path.join(project_root, "data",
                              out_folder, "governance_data")
    output_path_csv = os.path.join(output_dir, "offchain_proposals.csv")
    output_path_parquet = os.path.join(
        output_dir, "offchain_proposals.parquet")

    print(f"[info] dao id: {args.dao_id}")
    print(f"[info] output file (csv): {output_path_csv}")
    print(f"[info] output file (parquet): {output_path_parquet}")
    if args.test:
        print(
            "[info] running in test mode - finding first proposal with transfers if available")
    print("[info] collecting proposals")

    try:
        # select execution mode
        if args.test:
            # attempt to find a proposal id with transfers
            def _find_first_proposal_with_transfers(dao_id: str) -> Optional[str]:
                try:
                    from pathlib import Path
                    project_root = Path(__file__).resolve().parents[2]
                    transfers_path = project_root / "data" / dao_id / \
                        "governance_transfer_data" / "offchain_matched_transfers.csv"
                    if not transfers_path.exists():
                        return None
                    with open(transfers_path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            pid = row.get('proposal_id')
                            if pid:
                                return pid
                except Exception:
                    return None
                return None

            target_proposal_id = _find_first_proposal_with_transfers(
                args.dao_id)
            if target_proposal_id:
                print(
                    f"[test] found proposal with transfers: {target_proposal_id}")
                all_props = fetch_all_proposals(
                    args.dao_id, enhanced=args.enhanced, state=args.state)
                proposals = [p for p in all_props if p.get(
                    'id') == target_proposal_id]
                if not proposals:
                    print(
                        "[warning] target proposal not found in api response; falling back to limited test fetch")
                    proposals = fetch_proposals_test_mode(
                        args.dao_id, enhanced=args.enhanced, state=args.state)
            else:
                print(
                    "[warning] no offchain_matched_transfers.csv found or no proposal_id present; using limited test fetch")
                proposals = fetch_proposals_test_mode(
                    args.dao_id, enhanced=args.enhanced, state=args.state)
        else:
            # incremental mode loads existing ids and cursor
            stop_when_seen_ids_on_first_page = None
            initial_cursor = args.start_after_created if args.start_after_created else None
            if args.incremental and os.path.exists(output_path_csv):
                try:
                    existing_df = pd.read_csv(output_path_csv, usecols=[
                                              "id", "created"])  # small read
                    stop_when_seen_ids_on_first_page = set(
                        existing_df["id"].dropna().astype(str).tolist())
                    if not existing_df.empty and not args.start_after_created:
                        initial_cursor = int(existing_df["created"].max())
                        print(
                            f"[info] incremental mode: latest existing created ts = {initial_cursor}")
                except Exception as e:
                    print(
                        f"[warn] failed to read existing csv for incremental mode: {e}")
                    stop_when_seen_ids_on_first_page = None
                    if not args.start_after_created:
                        initial_cursor = None

            proposals = fetch_all_proposals(
                args.dao_id,
                enhanced=args.enhanced,
                state=("active" if args.state == "open" else args.state),
                stop_when_seen_ids_on_first_page=stop_when_seen_ids_on_first_page,
                initial_cursor=initial_cursor,
            )

        # integrity sanity check: sum(scores) should match total when final
        def _ok(p: Dict[str, Any]) -> bool:
            s = p.get("scores") or []
            st = p.get("scores_state")
            total = p.get("scores_total") or 0
            try:
                return (st != "final") or (abs(sum(s) - total) < 1e-6)
            except Exception:
                return True

        if args.enhanced:
            bad = [p.get("id") for p in proposals if not _ok(p)]
            if bad:
                print(
                    f"[warn] {len(bad)} proposals failed scores sanity check (e.g. {bad[:3]})")

        # export to csv or parquet based on flags
        if not args.parquet_only:
            write_proposals_to_csv(proposals, output_path_csv)
        if not args.csv_only:
            write_proposals_to_parquet(proposals, output_path_parquet)

        print("\n" + "="*60)
        print("[final summary]")
        print("="*60)
        print(f"dao id: {args.dao_id}")
        print(f"total proposals exported: {len(proposals):,}")
        if not args.parquet_only:
            print(f"output file (csv): {output_path_csv}")
        if not args.csv_only:
            print(f"output file (parquet): {output_path_parquet}")
        print("mode: unified cli (offchain proposals)")
        print("schema: 17 essential fields + 1 metadata json column")
        print("="*60)

    except Exception as e:
        print(f"[error] script failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
