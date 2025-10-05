import argparse
import json
import os
import re
import sys
import time
import random
from typing import Dict, Iterable, List, Optional, Set, Tuple

import backoff
from dotenv import load_dotenv
import requests
import pyarrow as pa
import pyarrow.parquet as pq


SNAPSHOT_GRAPHQL_ENDPOINT = "https://hub.snapshot.org/graphql"
TALLY_GRAPHQL_ENDPOINT = "https://api.tally.xyz/query"

# simple client-side rate limiter for tally to avoid 429s
TALLY_REQS_PER_MIN = 20
_TALLY_LAST_REQ_TS: float = 0.0

# force-include snapshot spaces regardless of proposal thresholds
FORCED_SNAPSHOT_INCLUDE_IDS: Set[str] = {"comp-vote.eth", "gitcoindao.eth"}

# shared output directories for parquet files
REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
CATALOGUE_DIR = os.path.join(REPO_ROOT, "catalogue")

# set in main() based on test mode
SNAPSHOT_UNFILTERED_PARQUET = None
SNAPSHOT_FILTERED_PARQUET = None
TALLY_UNFILTERED_PARQUET = None
TALLY_FILTERED_PARQUET = None


def set_output_paths(test_mode: bool) -> None:
    """set output file paths based on test mode"""
    global SNAPSHOT_UNFILTERED_PARQUET, SNAPSHOT_FILTERED_PARQUET
    global TALLY_UNFILTERED_PARQUET, TALLY_FILTERED_PARQUET

    suffix = "_test" if test_mode else ""

    # offchain snapshot paths
    offchain_dir = os.path.join(CATALOGUE_DIR, "offchain")

    # onchain tally paths
    onchain_dir = os.path.join(CATALOGUE_DIR, "onchain")

    SNAPSHOT_UNFILTERED_PARQUET = os.path.join(
        offchain_dir, f"snapshot_daos_unfiltered{suffix}.parquet")
    SNAPSHOT_FILTERED_PARQUET = os.path.join(
        offchain_dir, f"snapshot_daos{suffix}.parquet")
    TALLY_UNFILTERED_PARQUET = os.path.join(
        onchain_dir, f"tally_daos_unfiltered{suffix}.parquet")
    TALLY_FILTERED_PARQUET = os.path.join(
        onchain_dir, f"tally_daos{suffix}.parquet")


def set_tally_rate_limit(requests_per_minute: int) -> None:
    global TALLY_REQS_PER_MIN
    # clamp rate limit to a safe range
    try:
        rpm = int(requests_per_minute)
    except Exception:
        rpm = 20
    TALLY_REQS_PER_MIN = max(1, min(rpm, 60))


def build_spaces_query() -> str:
    """return the snapshot spaces graphql query with pagination variables"""
    return (
        """
        query Spaces($first: Int!, $skip: Int!) {
          spaces(first: $first, skip: $skip) {
            id
            name
            about
            network
            symbol
            proposalsCount
            website
            twitter
            github
            coingecko
            avatar
            terms
            admins
            members
            domain
            strategies {
              name
              params
            }
            filters {
              minScore
              onlyMembers
            }
            voting {
              delay
              period
              quorum
              type
              privacy
              hideAbstain
            }
            plugins
          }
        }
        """
        .strip()
    )


@backoff.on_exception(
    backoff.expo,
    (requests.HTTPError, requests.RequestException),
    max_tries=8,
    jitter=backoff.full_jitter,
    giveup=lambda e: not (
        hasattr(e, "response")
        and getattr(e, "response") is not None
        and e.response.status_code in (429, 502, 503, 504)
    ),
)
def post_graphql(query: str, variables: Dict) -> Dict:
    """post a snapshot graphql query and return json with retry handling"""
    response = requests.post(
        SNAPSHOT_GRAPHQL_ENDPOINT,
        json={"query": query, "variables": variables},
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            print(
                f"[Snapshot] 429 Too Many Requests. Retry-After={retry_after}s")
        else:
            print("[Snapshot] 429 Too Many Requests. Backing off and retrying...")
    response.raise_for_status()
    data = response.json()
    if "errors" in data and data["errors"]:
        # surface graphql errors clearly
        raise RuntimeError(f"GraphQL errors: {data['errors']}")
    return data


@backoff.on_exception(
    backoff.expo,
    (requests.HTTPError, requests.RequestException),
    max_tries=8,
    jitter=backoff.full_jitter,
    giveup=lambda e: not (
        hasattr(e, "response")
        and getattr(e, "response") is not None
        and e.response.status_code in (429, 502, 503, 504)
    ),
)
def post_tally_graphql(query: str, variables: Dict, api_key: str) -> Dict:
    """post a tally graphql query with api key auth and return json"""
    # client-side pacing
    global _TALLY_LAST_REQ_TS
    now = time.time()
    min_interval = 60.0 / float(TALLY_REQS_PER_MIN)
    elapsed = now - _TALLY_LAST_REQ_TS
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed + random.uniform(0.0, 0.2))

    response = requests.post(
        TALLY_GRAPHQL_ENDPOINT,
        json={"query": query, "variables": variables},
        headers={"Content-Type": "application/json", "Api-Key": api_key},
        timeout=30,
    )
    _TALLY_LAST_REQ_TS = time.time()
    # surface tally 422 responses for easier debugging
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            print(f"[Tally] 429 Too Many Requests. Retry-After={retry_after}s")
        else:
            print("[Tally] 429 Too Many Requests. Backing off and retrying...")
    if response.status_code == 422:
        try:
            print("[Tally] 422 response:", response.text)
        except Exception:
            pass
    response.raise_for_status()
    data = response.json()
    if "errors" in data and data["errors"]:
        raise RuntimeError(f"Tally GraphQL errors: {data['errors']}")
    return data


def fetch_snapshot_spaces(test_mode: bool = False, page_size: int = 1000) -> List[Dict]:
    """fetch snapshot spaces with pagination and optional test mode limit"""
    query = build_spaces_query()
    all_spaces: List[Dict] = []

    # in test mode fetch only the first page and slice to 10 entries
    if test_mode:
        print("[Snapshot] Fetching spaces (test mode)...")
        data = post_graphql(query, {"first": min(page_size, 1000), "skip": 0})
        spaces = data.get("data", {}).get("spaces", [])
        count = len(spaces[:10])
        print(f"[Snapshot] Fetched {count} spaces")
        return spaces[:10]

    # full pagination
    print("[Snapshot] Fetching spaces (paginated)...")
    skip = 0
    page = 0
    start_time = time.time()
    while True:
        page += 1
        data = post_graphql(query, {"first": page_size, "skip": skip})
        spaces = data.get("data", {}).get("spaces", [])
        if not spaces:
            break
        all_spaces.extend(spaces)
        elapsed = time.time() - start_time
        print(
            f"[Snapshot] Page {page}: +{len(spaces)} (total {len(all_spaces)}) in {elapsed:.1f}s")
        # if fewer than page_size returned we've reached the end
        if len(spaces) < page_size:
            break
        skip += page_size

    return all_spaces


def build_tally_organizations_query() -> str:
    """return the tally organizations graphql query with pagination"""
    return (
        """
        query Organizations($input: OrganizationsInput!) {
          organizations(input: $input) {
            nodes {
              ... on Organization {
                id
                name
                slug
                chainIds
                tokenIds
                governorIds
                hasActiveProposals
                proposalsCount
                delegatesCount
                delegatesVotesCount
                tokenOwnersCount
                metadata { icon color description }
              }
            }
            pageInfo { lastCursor }
          }
        }
        """
        .strip()
    )


def fetch_tally_organizations(
    api_key: str,
    test_mode: bool = False,
    page_size: int = 50,
) -> List[Dict]:
    """fetch tally organizations with pagination and optional test limit"""
    if not api_key:
        raise RuntimeError(
            "Tally API key not provided. Set TALLY_API_KEY env var or pass --tally-api-key."
        )

    query = build_tally_organizations_query()

    # clamp page size to tally's conservative limit
    try:
        page_size = int(page_size)
    except Exception:
        page_size = 20
    page_size = max(1, min(page_size, 20))

    if test_mode:
        print("[Tally] Fetching organizations (test mode)...")
        variables = {
            "input": {
                "page": {"limit": page_size, "afterCursor": None}
            }
        }
        data = post_tally_graphql(query, variables, api_key)
        payload = data.get("data", {}).get("organizations", {}) or {}
        orgs = payload.get("nodes", []) or []
        count = len(orgs[:10])
        print(f"[Tally] Fetched {count}")
        return orgs[:10]

    all_orgs: List[Dict] = []
    after: Optional[str] = None
    page = 0
    start_time = time.time()
    print("[Tally] Fetching organizations (paginated)...")
    while True:
        page += 1
        variables = {
            "input": {
                "page": {"limit": page_size, "afterCursor": after}
            }
        }
        data = post_tally_graphql(query, variables, api_key)
        payload = data.get("data", {}).get("organizations", {}) or {}
        nodes = payload.get("nodes", []) or []
        page_info = payload.get("pageInfo") or {}
        # adapt rpm up when page is full else trim down
        if len(nodes) >= page_size:
            set_tally_rate_limit(min(TALLY_REQS_PER_MIN + 2, 30))
        else:
            set_tally_rate_limit(max(TALLY_REQS_PER_MIN - 1, 8))
        all_orgs.extend(nodes)
        after = page_info.get("lastCursor")
        elapsed = time.time() - start_time
        print(
            f"[Tally] Page {page}: +{len(nodes)} (total {len(all_orgs)}) in {elapsed:.1f}s")
        if not nodes or not after:
            break
    return all_orgs


def normalize_name(raw_name: str) -> str:
    """normalize dao name for consistent comparison"""
    if not raw_name:
        return ""
    # collapse whitespace
    name = re.sub(r"\s+", " ", raw_name.strip())
    # remove trailing dao token in a case insensitive way
    name = re.sub(r"\s*dao\s*$", "", name, flags=re.IGNORECASE)
    name = name.strip()
    # lowercase final output
    normalized = name.lower()
    if not normalized:
        # fallback to lowercased original if everything was stripped
        normalized = raw_name.strip().lower()
    return normalized


def write_dao_parquet(
    daos: List[Dict],
    output_path: str,
    min_proposals: int = 30,
    forced_include_ids: Optional[Set[str]] = None,
    is_filtered: bool = False,
) -> Tuple[int, int]:
    """write dao data to parquet with optional proposal filter"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    schema = pa.schema([
        ("name", pa.string()),
        ("normalized_name", pa.string()),
        ("id", pa.string()),
        ("proposalsCount", pa.int64()),
        ("membersCount", pa.int64()),
        ("created", pa.int64()),
    ])

    names = []
    normalized_names = []
    ids = []
    proposals_counts = []
    members_counts = []
    created_times = []

    total_seen = 0
    total_written = 0

    forced_include_ids = forced_include_ids or set()

    for dao in daos:
        total_seen += 1

    # skip filtering when not requested
        if is_filtered:
            proposals = dao.get("proposalsCount")
            try:
                proposals_int = int(proposals) if proposals is not None else 0
            except Exception:
                proposals_int = 0

            dao_id = dao.get("id")
            dao_name = dao.get("name")

            # check forced inclusions
            dao_id_str = str(dao_id) if dao_id is not None else None
            dao_name_str = str(dao_name) if dao_name is not None else None
            normalized_name_for_check = (
                normalize_name(
                    dao_name_str) if dao_name_str is not None else None
            )
            is_forced = (
                (dao_id_str in forced_include_ids)
                or (
                    normalized_name_for_check is not None
                    and normalized_name_for_check in {normalize_name(name) for name in forced_include_ids}
                )
            )

            if proposals_int < min_proposals and not is_forced:
                continue

        # extract fields
        dao_name = dao.get("name", "")
        dao_id = dao.get("id", "")
        proposals_count = dao.get("proposalsCount", 0)
        created = dao.get("created", 0)

        # handle members count differences between snapshot and tally
        members_val = dao.get("members")
        if isinstance(members_val, list):
            members_count = len(members_val)
        else:
            # for tally use delegates or token owner counts when available
            members_count = (
                dao.get("delegatesCount", 0) or
                dao.get("tokenOwnersCount", 0) or
                members_val or 0
            )
            try:
                members_count = int(members_count)
            except Exception:
                members_count = 0

        try:
            proposals_count = int(proposals_count)
        except Exception:
            proposals_count = 0

        try:
            created = int(created)
        except Exception:
            created = 0

        # skip rows missing essential fields
        if not dao_id or not dao_name:
            continue

        names.append(str(dao_name))
        normalized_names.append(normalize_name(str(dao_name)))
        ids.append(str(dao_id))
        proposals_counts.append(proposals_count)
        members_counts.append(members_count)
        created_times.append(created)
        total_written += 1

    # create and write table
    table = pa.Table.from_arrays(
        [
            pa.array(names, type=pa.string()),
            pa.array(normalized_names, type=pa.string()),
            pa.array(ids, type=pa.string()),
            pa.array(proposals_counts, type=pa.int64()),
            pa.array(members_counts, type=pa.int64()),
            pa.array(created_times, type=pa.int64()),
        ],
        names=[
            "name",
            "normalized_name",
            "id",
            "proposalsCount",
            "membersCount",
            "created",
        ],
    )

    pq.write_table(table, output_path, compression="snappy")
    return total_seen, total_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch DAO configs for offchain (Snapshot) and/or onchain (Tally).\n"
            "Creates both unfiltered and filtered Parquet files.\n"
            "Interactive menu: choose 1) offchain, 2) onchain, 3) both.\n"
            "Use --mode to skip prompt. Use --test to limit to first 10 entries for each."
        )
    )
    parser.add_argument("--test", action="store_true",
                        help="Limit each fetch to first 10 entries.")
    parser.add_argument(
        "--mode",
        choices=["offchain", "onchain", "both"],
        help="Run without prompt by selecting which configs to generate.",
    )
    parser.add_argument(
        "--min-proposals",
        type=int,
        default=30,
        help="Minimum proposal count for filtered datasets (default: 30).",
    )
    parser.add_argument(
        "--tally-api-key",
        default=os.environ.get("TALLY_API_KEY"),
        help="Tally API key (or set TALLY_API_KEY env var).",
    )
    return parser.parse_args()


def main() -> None:
    # load env from repo root before parsing args (so defaults pick it up)
    # go up one level from utils/ to find the project root
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for env_candidate in (os.path.join(repo_root, ".env"), os.path.join(repo_root, "env")):
        if os.path.exists(env_candidate):
            load_dotenv(dotenv_path=env_candidate, override=False)

    args = parse_args()

    # set output paths based on test mode
    set_output_paths(args.test)

    # choose mode interactive unless --mode provided
    mode = args.mode
    if not mode:
        print("Select data to fetch:")
        print("  1) Offchain (Snapshot)")
        print("  2) Onchain (Tally)")
        print("  3) Both")
        choice = input("Enter 1, 2, or 3: ").strip()
        mode = {"1": "offchain", "2": "onchain",
                "3": "both"}.get(choice, "offchain")

    results = {}

    if mode in ("offchain", "both"):
        print("[Snapshot] Fetching spaces...")
        spaces = fetch_snapshot_spaces(test_mode=args.test)

        # write unfiltered parquet
        unfiltered_seen, unfiltered_written = write_dao_parquet(
            spaces, SNAPSHOT_UNFILTERED_PARQUET, is_filtered=False)
        print(
            f"[Snapshot] Wrote {unfiltered_written} unfiltered spaces to: {SNAPSHOT_UNFILTERED_PARQUET}")

        # write filtered parquet
        filtered_seen, filtered_written = write_dao_parquet(
            spaces, SNAPSHOT_FILTERED_PARQUET,
            min_proposals=args.min_proposals,
            forced_include_ids=FORCED_SNAPSHOT_INCLUDE_IDS,
            is_filtered=True)
        print(
            f"[Snapshot] Wrote {filtered_written} filtered spaces to: {SNAPSHOT_FILTERED_PARQUET}")

        results["snapshot"] = {
            "fetched": len(spaces),
            "unfiltered": {"seen": unfiltered_seen, "written": unfiltered_written, "output": SNAPSHOT_UNFILTERED_PARQUET},
            "filtered": {"seen": filtered_seen, "written": filtered_written, "output": SNAPSHOT_FILTERED_PARQUET}
        }

    if mode in ("onchain", "both"):
        tally_api_key = args.tally_api_key or os.environ.get("TALLY_API_KEY")
        if not tally_api_key:
            raise SystemExit(
                "[Tally] Missing TALLY_API_KEY. Pass --tally-api-key or set the env var.")

        print("[Tally] Fetching organizations...")
        orgs = fetch_tally_organizations(
            api_key=tally_api_key, test_mode=args.test)

        # write unfiltered parquet
        unfiltered_seen, unfiltered_written = write_dao_parquet(
            orgs, TALLY_UNFILTERED_PARQUET, is_filtered=False)
        print(
            f"[Tally] Wrote {unfiltered_written} unfiltered organizations to: {TALLY_UNFILTERED_PARQUET}")

        # write filtered parquet
        filtered_seen, filtered_written = write_dao_parquet(
            orgs, TALLY_FILTERED_PARQUET,
            min_proposals=args.min_proposals,
            is_filtered=True)
        print(
            f"[Tally] Wrote {filtered_written} filtered organizations to: {TALLY_FILTERED_PARQUET}")

        results["tally"] = {
            "fetched": len(orgs),
            "unfiltered": {"seen": unfiltered_seen, "written": unfiltered_written, "output": TALLY_UNFILTERED_PARQUET},
            "filtered": {"seen": filtered_seen, "written": filtered_written, "output": TALLY_FILTERED_PARQUET}
        }

    # print summary
    print("\nSummary:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
