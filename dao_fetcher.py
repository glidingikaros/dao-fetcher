#!/usr/bin/env python3
"""
interactive cli to fetch dao governance proposals from snapshot or tally
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
import importlib.util
import time
import requests
import logging


THIS_DIR = Path(__file__).resolve().parent


def detect_project_root() -> Path:
    """detect project root by walking upward for a requirements marker"""
    start = THIS_DIR
    for candidate in [start, *start.parents]:
        has_requirements = (candidate / "requirements.txt").exists()
        if has_requirements:
            return candidate
    return start


PROJECT_ROOT = detect_project_root()

# ensure project root and src are importable
SRC_DIR = PROJECT_ROOT / "src"
for path in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)


def _preload_data_collection_config() -> None:
    """ensure data_collection_config loads before sys.path adjustments"""
    module_name = "data_collection_config"
    if module_name in sys.modules:
        return
    module_path = PROJECT_ROOT / f"{module_name}.py"
    if not module_path.exists():
        return
    spec = importlib.util.spec_from_file_location(
        module_name, str(module_path))
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)  # type: ignore[attr-defined]


_preload_data_collection_config()


def _import_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if not spec or not spec.loader:
        raise ImportError(
            f"Cannot load spec for {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


logger = logging.getLogger(__name__)


def _existing_output_files(dao_slug: str, is_offchain: bool, kind: str) -> List[Path]:
    """return existing governance_data paths for the given dao and kind"""
    assert kind in {"proposals", "votes"}
    output_dir = PROJECT_ROOT / "data" / dao_slug / "governance_data"
    candidates: List[str]
    if is_offchain:
        if kind == "proposals":
            candidates = ["offchain_proposals.csv",
                          "offchain_proposals.parquet"]
        else:
            candidates = ["offchain_votes.csv", "offchain_votes.parquet"]
    else:
        if kind == "proposals":
            # include optional json outputs from other exporters
            candidates = [
                "onchain_proposals.csv",
                "onchain_proposals.parquet",
                "onchain_proposals.json",
            ]
        else:
            candidates = ["onchain_votes.csv", "onchain_votes.parquet"]

    existing: List[Path] = []
    for name in candidates:
        p = output_dir / name
        if p.exists():
            existing.append(p)
    return existing


OVERWRITE_OVERRIDE: Optional[bool] = None


def _prompt_yes_no(prompt: str, default_no: bool = True) -> bool:
    """prompt the user and return true if they answered yes"""
    if OVERWRITE_OVERRIDE is not None:
        choice_txt = "yes" if OVERWRITE_OVERRIDE else "no"
        logger.info(
            f"Overwrite prompt overridden (--overwrite={choice_txt}): {prompt}")
        return OVERWRITE_OVERRIDE
    suffix = " [y/N]: " if default_no else " [Y/n]: "
    ans = input(prompt + suffix).strip().lower()
    if not ans:
        return not default_no
    return ans in {"y", "yes"}


def _confirm_overwrite(kind: str, files: List[Path]) -> bool:
    """ask for overwrite confirmation for existing outputs"""
    if not files:
        return True
    plural = "files" if len(files) > 1 else "file"
    rels = ", ".join(str(f.relative_to(PROJECT_ROOT)) for f in files)
    return _prompt_yes_no(
        f"{kind.capitalize()} {plural} already exist ({rels}). Overwrite?",
        default_no=True,
    )


def _export_metadata_for_dao(source: str, identifier: str, dao_slug: str, is_tally_slug: Optional[bool] = None) -> Optional[Path]:
    """export dao metadata to data/<dao_slug>/metadata.parquet"""
    try:
        mod = _import_module_from_path(
            "fetch_dao_metadata_mod", SRC_DIR / "fetch_dao_metadata.py"
        )
        export_metadata = getattr(mod, "export_metadata")
    except Exception as e:
        logger.error(f"Could not load metadata exporter: {e}")
        return None

    out_path = PROJECT_ROOT / "data" / dao_slug / "metadata.parquet"
    try:
        record = export_metadata(
            source, identifier, str(out_path), is_tally_slug)
        logger.info(
            f"Metadata written for source={source} identifier={identifier} -> {out_path}"
        )
        # silence unused variable warning
        if record is None:
            pass
        return out_path
    except Exception as e:
        logger.warning(f"Metadata export failed: {e}")
        return None


def normalize(text: Optional[str]) -> str:
    return (text or "").strip().lower()


def normalize_loose(text: Optional[str]) -> str:
    """lowercase and strip non alphanumeric characters for lenient matching"""
    import re as _re
    return _re.sub(r"[^a-z0-9]", "", (text or "").strip().lower())


def _slugify(text: Optional[str]) -> str:
    base = (text or "").strip().lower()
    base = re.sub(r"\s+", "-", base)
    base = re.sub(r"[^a-z0-9._-]", "-", base)
    return base.strip("-_")


def strip_surrounding_quotes(text: Optional[str]) -> str:
    """remove one layer of surrounding quotes when present"""
    t = (text or "").strip()
    if len(t) < 2:
        return t
    # map of opening to closing quote chars
    pairs = {
        '"': '"',
        "'": "'",
        "`": "`",
        "“": "”",
        "‘": "’",
    }
    opener = t[0]
    closer = t[-1]
    expected = pairs.get(opener)
    if expected and closer == expected:
        return t[1:-1].strip()
    return t


def normalize_output_slug(text: Optional[str]) -> str:
    """normalize dao slug for output directories"""
    base = normalize(text)
    # remove a trailing dao token after common separators
    base = re.sub(r"[-_\s]*dao$", "", base, flags=re.IGNORECASE)
    base = base.strip("-_ ")
    return base


def _looks_like_snapshot_handle(text: Optional[str]) -> bool:
    """detect whether text resembles a snapshot space handle"""
    t = (text or "").strip().lower()
    if not t:
        return False
    return "." in t


def _load_parquet_records(candidates: List[Path]) -> Optional[List[Dict[str, Any]]]:
    """load records from the first readable parquet file in candidates"""
    for pq_path in candidates:
        if not pq_path.exists():
            continue
        try:
            import pandas as pd

            df = pd.read_parquet(pq_path)
            logger.debug(f"Loaded catalog from {pq_path}")
            return df.to_dict(orient="records")
        except ImportError as exc:
            logger.warning(f"pandas is required to read {pq_path}: {exc}")
            return None
        except Exception as exc:
            logger.warning(f"Failed to load catalog from {pq_path}: {exc}")
    return None


def _check_catalog_available(source_type: str) -> bool:
    """check whether the filtered catalog exists for the given source type"""
    catalogue_root = PROJECT_ROOT / "catalogue"
    if source_type == "offchain":
        catalog_path = catalogue_root / "offchain" / "snapshot_daos.parquet"
    elif source_type == "onchain":
        catalog_path = catalogue_root / "onchain" / "tally_daos.parquet"
    else:
        return False

    return catalog_path.exists()


def _prompt_run_list_fetch(source_type: str) -> bool:
    """prompt for running the list fetch synchronously"""
    if source_type == "offchain":
        source_name = "Snapshot (offchain)"
        cmd_mode = "offchain"
    elif source_type == "onchain":
        source_name = "Tally (onchain)"
        cmd_mode = "onchain"
    else:
        source_name = f"{source_type} (unknown)"
        cmd_mode = source_type

    cmd_display = f"python src/fetch_dao_list.py --mode {cmd_mode}"
    prompt = f"No {source_name} catalog found. Run '{cmd_display}' to fetch the latest DAO list?"
    suffix = " [Y/n]: "
    try:
        ans = input(prompt + suffix).strip().lower()
        accepted = (not ans) or ans in {"y", "yes"}
    except (EOFError, KeyboardInterrupt):
        accepted = True

    if not accepted:
        return False

    # run fetch synchronously
    try:
        import subprocess
        cmd = [sys.executable, str(
            SRC_DIR / "fetch_dao_list.py"), "--mode", cmd_mode]
        logger.info(f"Running list fetch: {' '.join(cmd)}")
        env = os.environ.copy()
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:  # type: ignore[name-defined]
        logger.error(f"List fetch failed with exit code {e.returncode}")
    except Exception as e:
        logger.error(f"List fetch failed: {e}")

    return True


def _load_snapshot_catalog(prompt_if_missing: bool = False) -> Optional[List[Dict[str, Any]]]:
    """load snapshot spaces from the filtered parquet catalog"""
    catalog_path = PROJECT_ROOT / "catalogue" / "offchain" / "snapshot_daos.parquet"
    candidates = [catalog_path]
    catalog = _load_parquet_records(candidates)
    if catalog is None and prompt_if_missing:
        if _prompt_run_list_fetch("offchain"):
            logger.info("Please run the list fetch command and try again.")
            sys.exit(0)
    return catalog


def _load_tally_catalog(prompt_if_missing: bool = False) -> Optional[List[Dict[str, Any]]]:
    """load tally organizations from the filtered parquet catalog"""
    catalog_path = PROJECT_ROOT / "catalogue" / "onchain" / "tally_daos.parquet"
    candidates = [catalog_path]
    catalog = _load_parquet_records(candidates)
    if catalog is None and prompt_if_missing:
        if _prompt_run_list_fetch("onchain"):
            logger.info("Please run the list fetch command and try again.")
            sys.exit(0)
    return catalog


def _resolve_from_snapshot(dao_input: str, spaces: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """resolve snapshot identifiers and slug from the provided catalog"""
    target = normalize(dao_input)
    target_loose = normalize_loose(dao_input)

    # exact match by id (strict or loose)
    for s in spaces:
        sid = normalize(s.get("id"))
        if sid == target or normalize_loose(s.get("id")) == target_loose:
            name = s.get("name") or s.get("id")
            return s.get("id"), str(name), _slugify(name)

    # exact name match collection and ranking
    exact_name_candidates: List[Dict[str, Any]] = []
    for s in spaces:
        if normalize(s.get("name")) == target or normalize_loose(s.get("name")) == target_loose:
            exact_name_candidates.append(s)
    if exact_name_candidates:
        def _score(entry: Dict[str, Any]) -> tuple:
            sid = str(entry.get("id") or "")
            is_eth = sid.endswith(".eth")
            try:
                props = int(entry.get("proposalsCount") or 0)
            except Exception:
                props = 0
            try:
                created = int(entry.get("created") or 0)
            except Exception:
                created = 0
            created_sort = created if created and created > 0 else 2**31-1
            return (1 if is_eth else 0, props, -created_sort)

        best = sorted(exact_name_candidates, key=_score, reverse=True)[0]
        name = best.get("name") or best.get("id")
        return best.get("id"), str(name), _slugify(name)

    # unique prefix match by id (strict or loose)
    prefix_candidates: List[Dict[str, Any]] = []
    for s in spaces:
        sid_raw = s.get("id")
        sid_norm = normalize(sid_raw)
        sid_loose = normalize_loose(sid_raw)
        if sid_norm and (sid_norm.startswith(target) or target.startswith(sid_norm)):
            prefix_candidates.append(s)
            continue
        if sid_loose and (sid_loose.startswith(target_loose) or target_loose.startswith(sid_loose)):
            prefix_candidates.append(s)
    # require a unique match before returning
    if len({(c.get("id") or "").strip().lower() for c in prefix_candidates if c.get("id")}) == 1 and prefix_candidates:
        best = prefix_candidates[0]
        name = best.get("name") or best.get("id")
        return best.get("id"), str(name), _slugify(name)

    # rank prefix or fuzzy matches and accept only unique candidates
    candidates: List[Dict[str, Any]] = []
    for s in spaces:
        nm_raw = s.get("name")
        sid_raw = s.get("id")
        nm = normalize(nm_raw)
        sid = normalize(sid_raw)
        nm_loose = normalize_loose(nm_raw)
        sid_loose = normalize_loose(sid_raw)
        if (
            (nm and nm.startswith(target))
            or (sid and sid.startswith(target))
            or (nm and target.startswith(nm))
            or (sid and target.startswith(sid))
            or (nm_loose and nm_loose.startswith(target_loose))
            or (sid_loose and sid_loose.startswith(target_loose))
            or (nm_loose and target_loose.startswith(nm_loose))
            or (sid_loose and target_loose.startswith(sid_loose))
        ):
            candidates.append(s)

    if candidates:
        # require a unique candidate before returning a match
        uniq: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        for c in candidates:
            sidv = (c.get("id") or "").strip().lower()
            if sidv and sidv not in seen_ids:
                seen_ids.add(sidv)
                uniq.append(c)
        if len(uniq) == 1:
            best = uniq[0]
            name = best.get("name") or best.get("id")
            return best.get("id"), str(name), _slugify(name)

    return None, None, None


def _resolve_from_tally(dao_input: str, orgs: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """resolve tally identifier and slug using catalog entries"""
    target = normalize(dao_input)
    target_loose = normalize_loose(dao_input)

    # exact id or slug
    for o in orgs:
        if (
            normalize(o.get("id")) == target
            or normalize(o.get("slug")) == target
            or normalize_loose(o.get("id")) == target_loose
            or normalize_loose(o.get("slug")) == target_loose
        ):
            name = o.get("name") or o.get("slug") or o.get("id")
            ident = o.get("id") or o.get("slug")
            return str(ident) if ident is not None else None, str(name), _slugify(name)

    # exact name match
    for o in orgs:
        if normalize(o.get("name")) == target or normalize_loose(o.get("name")) == target_loose:
            name = o.get("name") or o.get("slug") or o.get("id")
            ident = o.get("id") or o.get("slug")
            return str(ident) if ident is not None else None, str(name), _slugify(name)

    # prefix id or slug match when unique
    prefix_candidates: List[Dict[str, Any]] = []
    for o in orgs:
        oid_raw = o.get("id")
        oslug_raw = o.get("slug")
        oid = normalize(oid_raw)
        oslug = normalize(oslug_raw)
        oid_loose = normalize_loose(oid_raw)
        oslug_loose = normalize_loose(oslug_raw)
        if (
            (oid and (oid.startswith(target) or target.startswith(oid)))
            or (oslug and (oslug.startswith(target) or target.startswith(oslug)))
            or (oid_loose and (oid_loose.startswith(target_loose) or target_loose.startswith(oid_loose)))
            or (oslug_loose and (oslug_loose.startswith(target_loose) or target_loose.startswith(oslug_loose)))
        ):
            prefix_candidates.append(o)
    if len({(c.get("id") or c.get("slug") or "").strip().lower() for c in prefix_candidates if (c.get("id") or c.get("slug"))}) == 1 and prefix_candidates:
        best = prefix_candidates[0]
        name = best.get("name") or best.get("slug") or best.get("id")
        ident = best.get("id") or best.get("slug")
        return str(ident) if ident is not None else None, str(name), _slugify(name)

    # prefix name match when unique
    name_candidates: List[Dict[str, Any]] = []
    for o in orgs:
        oname_raw = o.get("name")
        oname = normalize(oname_raw)
        oname_loose = normalize_loose(oname_raw)
        if (
            (oname and (oname.startswith(target) or target.startswith(oname)))
            or (oname_loose and (oname_loose.startswith(target_loose) or target_loose.startswith(oname_loose)))
        ):
            name_candidates.append(o)
    if len({(c.get("id") or c.get("slug") or "").strip().lower() for c in name_candidates if (c.get("id") or c.get("slug"))}) == 1 and name_candidates:
        best = name_candidates[0]
        name = best.get("name") or best.get("slug") or best.get("id")
        ident = best.get("id") or best.get("slug")
        return str(ident) if ident is not None else None, str(name), _slugify(name)

    return None, None, None


def fetch_offchain(snapshot_space_id: str, dao_slug: str) -> None:
    logger.info(f"Using Snapshot space: {snapshot_space_id}")
    proposals = None
    try:
        offchain_mod = _import_module_from_path(
            "fetch_offchain_proposals_mod", SRC_DIR / "fetch_offchain_proposals.py"
        )
        snapshot_fetch_all_proposals = getattr(
            offchain_mod, "fetch_all_proposals")
        snapshot_write_csv = getattr(offchain_mod, "write_proposals_to_csv")
        snapshot_write_parquet = getattr(
            offchain_mod, "write_proposals_to_parquet")
        proposals = snapshot_fetch_all_proposals(
            snapshot_space_id, enhanced=True, state=None)
        # output to <root>/data/<dao_slug>/governance_data/
        output_dir = PROJECT_ROOT / "data" / dao_slug / "governance_data"
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "offchain_proposals.csv"
        parquet_path = output_dir / "offchain_proposals.parquet"
        snapshot_write_csv(proposals, str(csv_path))
        snapshot_write_parquet(proposals, str(parquet_path))
        logger.info(
            f"Offchain proposals: {len(proposals)} written to {csv_path} and {parquet_path}")
        return
    except Exception as e:
        logger.warning(
            f"Module import/execution failed; will attempt subprocess then direct fetch: {e}")

    # fallback: run the script via subprocess and then copy outputs
    import subprocess
    try:
        cmd = [sys.executable, str(
            SRC_DIR / "fetch_offchain_proposals.py"), snapshot_space_id, "--enhanced"]
        if logger.isEnabledFor(logging.DEBUG):
            cmd.append("--verbose")
        env = os.environ.copy()
        # ensure project root is on pythonpath so data_collection_config resolves
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{PROJECT_ROOT}:{existing_pp}" if existing_pp else str(
            PROJECT_ROOT)
        subprocess.run(cmd, check=True, env=env)
        # copy outputs from top-level output/<space_id>/ to <root>/data/<dao_slug>/governance_data/
        src_output_dir = PROJECT_ROOT / "output" / snapshot_space_id
        root_output_dir = PROJECT_ROOT / "data" / dao_slug / "governance_data"
        root_output_dir.mkdir(parents=True, exist_ok=True)
        # move or copy csv and parquet when present
        for fname in ("offchain_proposals.csv", "offchain_proposals.parquet"):
            src_path = src_output_dir / fname
            if src_path.exists():
                import shutil
                shutil.copy2(src_path, root_output_dir / fname)
        logger.info(f"Offchain outputs copied to {root_output_dir}")
    except subprocess.CalledProcessError as se:
        logger.warning(f"Subprocess execution failed: {se}")

    # final fallback: direct snapshot graphql fetch and write outputs
    try:
        from data_collection_config import ENDPOINT_SNAPSHOT, PAGE_LIMIT, RETRY_WAIT
    except Exception:
        # defaults when config import fails
        ENDPOINT_SNAPSHOT = "https://hub.snapshot.org/graphql"
        PAGE_LIMIT = 100
        RETRY_WAIT = 5

    PROPOSALS_QUERY = """
    query ProposalsWithPlugins($space:String!, $first:Int!, $skip:Int!) {
      proposals(
        first:$first, skip:$skip,
        where:{ space:$space },
        orderBy:"created", orderDirection:desc
      ) {
        id ipfs space { id } author created app network title body discussion choices type privacy quorum start end snapshot symbol scores scores_by_strategy scores_total scores_state scores_updated votes strategies { name network params } validation { name params } state flagged plugins
      }
    }
    """

    def _request_with_retry(payload: Dict[str, Any]) -> Dict[str, Any]:
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                r = requests.post(
                    ENDPOINT_SNAPSHOT,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=120,
                )
                if r.status_code == 429:
                    wait_s = RETRY_WAIT * (2 ** attempt)
                    logger.warning(
                        f"Snapshot 429 rate limited, backing off for {wait_s}s (attempt {attempt+1}/{max_attempts})")
                    time.sleep(wait_s)
                    continue
                r.raise_for_status()
                data = r.json()
                if "errors" in data:
                    raise RuntimeError(str(data["errors"]))
                return data.get("data", {})
            except Exception as e:
                if attempt == max_attempts - 1:
                    logger.error(
                        f"Snapshot request failed after {max_attempts} attempts: {e}")
                    raise
                wait_s = RETRY_WAIT * (2 ** attempt)
                logger.warning(
                    f"Snapshot request error, retrying in {wait_s}s (attempt {attempt+1}/{max_attempts}): {e}")
                time.sleep(wait_s)
        return {}

    all_props: List[Dict[str, Any]] = []
    offset = 0
    while True:
        variables = {"space": snapshot_space_id,
                     "first": PAGE_LIMIT, "skip": offset}
        data = _request_with_retry(
            {"query": PROPOSALS_QUERY, "variables": variables})
        props = data.get("proposals", []) or []
        all_props.extend(props)
        if len(props) < PAGE_LIMIT:
            break
        offset += PAGE_LIMIT

    # flatten minimal fields with metadata similar to module behavior
    def _flatten(p: Dict[str, Any]) -> Dict[str, Any]:
        flat = {
            "id": p.get("id", ""),
            "ipfs": p.get("ipfs", ""),
            "space": (p.get("space", {}) or {}).get("id", ""),
            "author": p.get("author", ""),
            "network": p.get("network", ""),
            "created": p.get("created", 0),
            "start": p.get("start", 0),
            "end": p.get("end", 0),
            "snapshot": p.get("snapshot", 0),
            "type": p.get("type", ""),
            "quorum": p.get("quorum", 0),
            "state": p.get("state", ""),
            "flagged": p.get("flagged", False),
            "votes": p.get("votes", 0),
            "scores_total": p.get("scores_total", 0),
            "scores_state": p.get("scores_state", ""),
            "scores_updated": p.get("scores_updated", 0),
        }
        meta = {
            "title": p.get("title", ""),
            "body": p.get("body", ""),
            "discussion": p.get("discussion", ""),
            "choices": p.get("choices", []),
            "privacy": p.get("privacy", ""),
            "symbol": p.get("symbol", ""),
            "app": p.get("app", ""),
            "scores": p.get("scores", []),
            "scores_by_strategy": p.get("scores_by_strategy", []),
            "strategies": p.get("strategies", []),
            "validation": p.get("validation", {}),
            "plugins": p.get("plugins", {}),
        }
        import json as _json
        flat["metadata"] = _json.dumps(meta, ensure_ascii=False)
        return flat

    output_dir = PROJECT_ROOT / "data" / dao_slug / "governance_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "offchain_proposals.csv"
    parquet_path = output_dir / "offchain_proposals.parquet"

    if all_props:
        flattened = [_flatten(p) for p in all_props]
        import pandas as pd
        df = pd.DataFrame(flattened)
        df.to_csv(csv_path, index=False)
        df.to_parquet(parquet_path, index=False)
    logger.info(
        f"Offchain proposals: {len(all_props)} written to {csv_path} and {parquet_path}")


def fetch_onchain(tally_org_id: str, dao_label: str, dao_slug: str) -> None:
    logger.info(f"Using Tally organization id: {tally_org_id}")

    api_key = os.environ.get("TALLY_API_KEY")
    if not api_key:
        logger.error(
            "TALLY_API_KEY environment variable is not set. Get an API key from Tally settings and export it, e.g.: export TALLY_API_KEY=\"<your_key>\"")
        sys.exit(1)

    # try module implementation first
    try:
        onchain_mod = _import_module_from_path(
            "fetch_onchain_proposals_mod", SRC_DIR / "fetch_onchain_proposals.py"
        )
        TallyClient = getattr(onchain_mod, "TallyClient")
        export_org = getattr(onchain_mod, "export_proposals_for_organization")
    except Exception as e:
        logger.warning(
            f"Could not load onchain module; falling back to direct requests: {e}")
        TallyClient = None

    if TallyClient is None:
        # minimal graphql fallback that writes csv and parquet
        import pandas as pd
        headers = {"Api-Key": api_key, "Content-Type": "application/json"}
        url = "https://api.tally.xyz/query"
        query = """
        query GetProposals($input: ProposalsInput!) {
          proposals(input: $input) {
            nodes { ... on Proposal { id onchainId status metadata { title } } }
            pageInfo { lastCursor }
          }
        }
        """
        all_nodes: List[Dict[str, Any]] = []
        after: Optional[str] = None
        limit = 100
        while True:
            page = {"limit": limit}
            if after:
                page["afterCursor"] = after
            variables = {"input": {"filters": {
                "organizationId": str(tally_org_id)}, "page": page}}
            resp = requests.post(url, headers=headers, json={
                                 "query": query, "variables": variables}, timeout=60)
            resp.raise_for_status()
            data = resp.json().get("data", {})
            props = data.get("proposals", {})
            nodes = props.get("nodes", []) or []
            page_info = props.get("pageInfo", {})
            all_nodes.extend(nodes)
            if len(nodes) < limit:
                break
            after = page_info.get("lastCursor")

        # write csv and parquet outputs
        output_dir = PROJECT_ROOT / "data" / dao_slug / "governance_data"
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "onchain_proposals.csv"
        parquet_path = output_dir / "onchain_proposals.parquet"

        rows = [{
            "id": n.get("id"),
            "onchainId": n.get("onchainId"),
            "status": n.get("status"),
            "title": (n.get("metadata") or {}).get("title")
        } for n in all_nodes]
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        try:
            df.to_parquet(parquet_path, index=False)
        except Exception as e:
            logger.warning(f"Failed writing Parquet: {e}")
        logger.info(
            f"Onchain proposals for {dao_label}: {len(all_nodes)} total -> {csv_path} and {parquet_path}")
        return

    # normal path via module client and exporter
    client = TallyClient(api_key=api_key)
    output_dir = PROJECT_ROOT / "data" / dao_slug / "governance_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # detect whether tally_org_id looks numeric or slug-like
    is_slug = False
    try:
        int(str(tally_org_id))
        is_slug = False
    except Exception:
        is_slug = True

    try:
        nodes = export_org(
            client=client,
            organization=str(tally_org_id),
            output_dir=str(output_dir),
            is_slug=is_slug,
            page_limit=100,
        )
        logger.info(
            f"Onchain proposals for {dao_label}: {len(nodes)} total -> {output_dir}/onchain_proposals.csv and .parquet")
    except Exception as e:
        logger.error(f"Failed to export onchain proposals: {e}")
        sys.exit(1)


def _fetch_offchain_votes(snapshot_space_id: str, dao_slug: str) -> None:
    """fetch snapshot votes for all proposals into data/<dao_slug>"""
    try:
        votes_mod = _import_module_from_path(
            "fetch_offchain_votes_mod", SRC_DIR / "fetch_offchain_votes.py"
        )
        SnapshotVoteFetcher = getattr(votes_mod, "SnapshotVoteFetcher")
    except Exception as e:
        logger.warning(f"Could not load offchain votes module: {e}")
        return

    # output paths
    output_dir = PROJECT_ROOT / "data" / dao_slug / "governance_data"
    proposals_csv_path = output_dir / "offchain_proposals.csv"
    if not proposals_csv_path.exists():
        logger.warning(
            f"Offchain proposals CSV not found at {proposals_csv_path}; skipping votes fetch")
        return

    # optional snapshot api key
    snapshot_api_key = os.environ.get("SNAPSHOT_API_KEY")
    fetcher = SnapshotVoteFetcher(api_key=snapshot_api_key)
    try:
        # ensure output dir exists and write consolidated outputs there
        output_dir.mkdir(parents=True, exist_ok=True)
        fetcher.fetch_votes_by_space(
            space_id=snapshot_space_id,
            proposals_csv_path=str(proposals_csv_path),
            output_dir=str(output_dir),
            resume=False,
            delay=1.0,
            batch_size=0,
            start_from=1,
        )
        logger.info(
            f"Offchain votes written to {output_dir}/offchain_votes.*")
    except Exception as e:
        logger.error(f"Offchain votes fetch failed: {e}")


def _write_onchain_proposals_csv_from_json(dao_slug: str) -> Optional[Path]:
    """convert onchain_proposals.json to a minimal csv with id and title columns"""
    output_dir = PROJECT_ROOT / "data" / dao_slug / "governance_data"
    json_path = output_dir / "onchain_proposals.json"
    csv_path = output_dir / "onchain_proposals.csv"
    if not json_path.exists():
        logger.warning(f"Onchain proposals JSON not found at {json_path}")
        return None

    try:
        import json as _json
        with open(json_path, "r", encoding="utf-8") as f:
            nodes = _json.load(f) or []
        # normalize to list of dicts
        if not isinstance(nodes, list):
            logger.warning(
                f"Unexpected JSON structure in {json_path}; skipping CSV conversion")
            return None
        # write minimal csv
        import csv as _csv
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(
                f, fieldnames=["id", "onchainId", "metadata_title"])
            writer.writeheader()
            for n in nodes:
                meta = (n.get("metadata") or {})
                writer.writerow({
                    "id": n.get("id", ""),
                    "onchainId": n.get("onchainId", ""),
                    "metadata_title": meta.get("title", ""),
                })
        return csv_path
    except Exception as e:
        logger.error(f"Failed to convert {json_path} to CSV: {e}")
        return None


def _fetch_onchain_votes(dao_slug: str) -> None:
    """fetch tally votes for all dao proposals into data/<dao_slug>"""
    api_key = os.environ.get("TALLY_API_KEY")
    if not api_key:
        logger.warning("TALLY_API_KEY not set; skipping onchain votes fetch")
        return

    # ensure proposals csv exists, convert from json if needed
    proposals_csv_path = PROJECT_ROOT / "data" / dao_slug / \
        "governance_data" / "onchain_proposals.csv"
    if not proposals_csv_path.exists():
        converted = _write_onchain_proposals_csv_from_json(dao_slug)
        if not converted:
            logger.warning(
                "Could not prepare proposals CSV; skipping onchain votes fetch")
            return
        proposals_csv_path = converted

    try:
        votes_mod = _import_module_from_path(
            "fetch_onchain_votes_mod", SRC_DIR / "fetch_onchain_votes.py"
        )
        TallyVotesClient = getattr(votes_mod, "TallyVotesClient")
    except Exception as e:
        logger.warning(f"Could not load onchain votes module: {e}")
        return

    # use root output as base directory
    output_base_dir = PROJECT_ROOT / "data"
    client = TallyVotesClient(api_key=api_key)
    try:
        client.bulk_fetch_dao_votes(
            dao_name=dao_slug,
            proposals_csv_path=str(proposals_csv_path),
            output_base_dir=str(output_base_dir),
            resume=False,
            delay=1.0,
            include_pending_votes=False,
            token_decimals=18,
        )
        logger.info(
            f"Onchain votes written to {output_base_dir/dao_slug/'governance_data'}/onchain_votes.csv")
    except Exception as e:
        logger.error(f"Onchain votes fetch failed: {e}")


# -------------------------- TEST MODE HELPERS --------------------------

def _detect_mid_window_timestamp(dao_slug: str) -> Tuple[Optional[int], bool]:
    """estimate a mid window unix timestamp from available transfer data ranges"""
    # prefer new canonical location data/<dao_slug>/transfer_data
    # fall back to data/transfer_data/<dao_slug> or transfer_data/<dao_slug>
    base_new = PROJECT_ROOT / "data" / dao_slug / "transfer_data"
    base_legacy_under_data = PROJECT_ROOT / "data" / "transfer_data" / dao_slug
    base_legacy_top = PROJECT_ROOT / "transfer_data" / dao_slug
    base = (
        base_new
        if base_new.exists()
        else (base_legacy_under_data if base_legacy_under_data.exists() else base_legacy_top)
    )
    if not base.exists():
        return None, False
    ranges: List[Tuple[int, int]] = []
    for f in sorted(base.glob("*.parquet")):
        try:
            name = f.stem  # e.g. 15000000-15999999
            start_str, end_str = name.split("-")
            start_b = int(start_str)
            end_b = int(end_str)
            ranges.append((start_b, end_b))
        except Exception:
            continue
    if not ranges:
        return None, False
    min_block = min(r[0] for r in ranges)
    max_block = max(r[1] for r in ranges)
    mid_block = (min_block + max_block) // 2

    # map block numbers to timestamps via utils/blocks_ts.csv.gz when available
    blocks_map_path = PROJECT_ROOT / "utils" / "blocks_ts.csv.gz"
    if not blocks_map_path.exists():
        return None, True
    try:
        import pandas as pd
        df = pd.read_csv(blocks_map_path, compression="gzip")
        # expect columns named block and timestamp in seconds
        if "block" in df.columns and "timestamp" in df.columns:
            # choose the closest block entry
            idx = (df["block"] - mid_block).abs().idxmin()
            ts = int(df.loc[idx, "timestamp"]) if pd.notnull(
                df.loc[idx, "timestamp"]) else None
            return (ts if ts and ts > 0 else None), True
    except Exception:
        pass
    return None, True


def _select_offchain_proposal_near_ts(dao_slug: str, target_ts: Optional[int]) -> Optional[str]:
    """select a snapshot proposal near the target timestamp or fallback to the median"""
    csv_path = PROJECT_ROOT / "data" / dao_slug / \
        "governance_data" / "offchain_proposals.csv"
    if not csv_path.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        if df.empty:
            return None
        if target_ts and "start" in df.columns:
            df["_dist"] = (df["start"] - target_ts).abs()
            row = df.sort_values("_dist").iloc[0]
            return str(row.get("id")) if row.get("id") else None
        # fallback: pick the middle proposal when no timestamp available
        row = df.iloc[len(df) // 2]
        return str(row.get("id")) if row.get("id") else None
    except Exception:
        return None


def _offchain_test_fetch_single(snapshot_space_id: str, dao_slug: str, proposal_id: str) -> None:
    """fetch votes for one snapshot proposal and write csv plus parquet"""
    try:
        votes_mod = _import_module_from_path(
            "fetch_offchain_votes_mod", SRC_DIR / "fetch_offchain_votes.py"
        )
        SnapshotVoteFetcher = getattr(votes_mod, "SnapshotVoteFetcher")
        VoteCSVExporter = getattr(votes_mod, "VoteCSVExporter")
    except Exception as e:
        logger.warning(f"Could not load offchain votes module for test: {e}")
        return

    fetcher = SnapshotVoteFetcher(api_key=os.environ.get("SNAPSHOT_API_KEY"))
    votes = fetcher.fetch_all_votes(proposal_id)
    exporter = VoteCSVExporter()
    out_dir = PROJECT_ROOT / "data" / dao_slug / "governance_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_out = out_dir / "offchain_votes.csv"
    parquet_out = out_dir / "offchain_votes.parquet"
    exporter.export_to_csv(votes, csv_out)
    exporter.export_to_parquet(votes, parquet_out)
    logger.info(
        f"Test-mode offchain votes exported -> {csv_out} and {parquet_out}")


def _select_onchain_proposal_near_ts(dao_slug: str, target_ts: Optional[int]) -> Optional[int]:
    """select a tally proposal near the target timestamp with csv fallback"""
    json_path = PROJECT_ROOT / "data" / dao_slug / \
        "governance_data" / "onchain_proposals.json"
    if json_path.exists():
        try:
            import json as _json
            with open(json_path, "r", encoding="utf-8") as f:
                nodes = _json.load(f) or []
            if isinstance(nodes, list) and nodes:
                if target_ts:
                    def node_ts(n: Dict[str, Any]) -> Optional[int]:
                        block = n.get("block") or {}
                        ts = block.get("timestamp")
                        if ts is None:
                            start = n.get("start") or {}
                            ts = start.get("timestamp")
                        return int(ts) if isinstance(ts, (int, float)) else None
                    best = None
                    best_dist = None
                    for n in nodes:
                        nts = node_ts(n)
                        if nts is None:
                            continue
                        d = abs(nts - target_ts)
                        if best is None or d < best_dist:
                            best = n
                            best_dist = d
                    if best and best.get("id"):
                        try:
                            return int(best.get("id"))
                        except Exception:
                            pass
                mid = nodes[len(nodes) // 2]
                pid = mid.get("id") or mid.get("onchainId")
                return int(pid) if pid and str(pid).isdigit() else None
        except Exception:
            pass

    # fallback: csv
    try:
        import pandas as pd
        csv_path = PROJECT_ROOT / "data" / dao_slug / \
            "governance_data" / "onchain_proposals.csv"
        if not csv_path.exists():
            return None
        df = pd.read_csv(csv_path)
        if df.empty:
            return None
        row = df.iloc[len(df) // 2]
        candidate = row.get("id") if "id" in df.columns else None
        if candidate is None or (isinstance(candidate, str) and not candidate.isdigit()):
            candidate = row.get(
                "onchainId") if "onchainId" in df.columns else None
        try:
            return int(candidate) if candidate is not None and str(candidate).isdigit() else None
        except Exception:
            return None
    except Exception:
        return None


def _onchain_test_fetch_single(dao_slug: str, tally_proposal_id: int) -> None:
    """fetch votes for one tally proposal and write csv plus parquet"""
    api_key = os.environ.get("TALLY_API_KEY")
    if not api_key:
        logger.warning(
            "TALLY_API_KEY not set; skipping onchain votes (test mode)")
        return
    try:
        votes_mod = _import_module_from_path(
            "fetch_onchain_votes_mod", SRC_DIR / "fetch_onchain_votes.py"
        )
        TallyVotesClient = getattr(votes_mod, "TallyVotesClient")
    except Exception as e:
        logger.warning(f"Could not load onchain votes module for test: {e}")
        return
    client = TallyVotesClient(api_key=api_key)
    df = client.fetch_votes_for_proposal(tally_proposal_id)
    out_dir = PROJECT_ROOT / "data" / dao_slug / "governance_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_out = out_dir / "onchain_votes.csv"
    parquet_out = out_dir / "onchain_votes.parquet"
    df.to_csv(csv_out, index=False)
    try:
        df.to_parquet(parquet_out, index=False)
    except Exception as e:
        logger.warning(f"Failed to write parquet {parquet_out}: {e}")
    logger.info(
        f"Test-mode onchain votes exported -> {csv_out} and {parquet_out}")


def main() -> None:
    # load environment from project root (supports ".env" or "env")
    dotenv_candidates = [PROJECT_ROOT / ".env", PROJECT_ROOT / "env"]
    for candidate in dotenv_candidates:
        if candidate.exists():
            load_dotenv(dotenv_path=candidate, override=False)

    # cli args
    import argparse as _argparse
    parser = _argparse.ArgumentParser(
        description="Fetch DAO governance data from Snapshot (offchain) or Tally (onchain).",
        formatter_class=_argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Offchain proposals+votes (interactive overwrite):\n"
            "    python dao_fetcher.py --source snapshot --dao aave\n\n"
            "  Onchain proposals only (non-interactive, overwrite=yes):\n"
            "    python dao_fetcher.py --source tally --dao uniswap --overwrite yes --only-proposals\n\n"
            "  Use known IDs and skip votes:\n"
            "    python dao_fetcher.py --source snapshot --dao aave --space-id aave.eth --no-votes\n"
        ),
    )
    parser.add_argument("--source", default=None,
                        help="Data source: 1/offchain/snapshot or 2/onchain/tally")
    parser.add_argument(
        "--dao",
        dest="dao",
        default=None,
        help=(
            "DAO identifier. For offchain, pass the Snapshot handle (e.g., 'harvestfi.eth'). "
            "For onchain, pass a name/slug (e.g., 'uniswap')."
        ),
    )
    parser.add_argument("--test", action="store_true",
                        help="Test mode: fetch votes for a single nearby proposal")
    parser.add_argument("--overwrite", choices=["yes", "no"], default=None,
                        help="Non-interactive overwrite choice for existing output files")
    parser.add_argument("--no-votes", action="store_true",
                        help="Skip votes step")
    parser.add_argument("--only-proposals", action="store_true",
                        help="Fetch proposals only; skip votes")
    parser.add_argument("--only-votes", action="store_true",
                        help="Fetch votes only; requires proposals outputs to exist")
    parser.add_argument("--space-id", dest="space_id",
                        default=None, help="Override Snapshot space id")
    parser.add_argument("--org-id", dest="org_id", default=None,
                        help="Override Tally organization id or slug")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose logging (DEBUG)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Reduce logging output (ERROR)")

    args = parser.parse_args()

    # logging setup: default INFO; --verbose -> DEBUG; --quiet -> ERROR
    if args.verbose:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.ERROR
    else:
        log_level = logging.INFO
    logging.basicConfig(level=log_level, format='[%(levelname)s] %(message)s')
    logger.info("DAO Governance Fetcher")

    # choose source type supporting numeric or textual options
    def _normalize_source_choice(val: str) -> Optional[str]:
        v = (val or "").strip().lower()
        if v in {"1", "offchain", "snapshot"}:
            return "offchain"
        if v in {"2", "onchain", "tally"}:
            return "onchain"
        return None

    # apply overwrite override if provided
    global OVERWRITE_OVERRIDE
    if getattr(args, "overwrite", None) in ("yes", "no"):
        OVERWRITE_OVERRIDE = True if args.overwrite == "yes" else False

    if args.source:
        source_choice_norm = _normalize_source_choice(args.source)
        if not source_choice_norm:
            logger.error(
                "Invalid --source. Use 1/offchain/snapshot or 2/onchain/tally")
            sys.exit(1)
        source_choice = source_choice_norm
    else:
        source_choice = None
        while source_choice is None:
            raw = input(
                " Select data source\n  1) Off-chain (Snapshot)\n  2) On-chain  (Tally)\n[default: 1] >")
            if not (raw or "").strip():
                raw = "1"
            source_choice = _normalize_source_choice(raw)

    is_offchain = source_choice == "offchain"
    logger.info(
        f"Source selected: {'offchain/snapshot' if is_offchain else 'onchain/tally'}")

    # check if catalog exists before prompting for dao
    source_type = "offchain" if is_offchain else "onchain"
    if not _check_catalog_available(source_type):
        if _prompt_run_list_fetch(source_type):
            logger.info("Please run the list fetch command and try again.")
            sys.exit(0)

    # validate mutually exclusive switches
    if args.only_proposals and args.only_votes:
        logger.error(
            "--only-proposals and --only-votes are mutually exclusive")
        sys.exit(2)

    # dao input prompt
    if args.dao:
        dao_input = strip_surrounding_quotes(args.dao)
    else:
        dao_input = strip_surrounding_quotes(
            input(
                "Enter DAO handle: "
            )
        )
        if not dao_input:
            logger.error("DAO name is required")
            sys.exit(1)

    # resolve ids based on source selection
    snapshot_id: Optional[str] = None
    tally_id: Optional[str] = None
    dao_name: str = dao_input
    dao_slug: str = normalize(dao_input)

    if is_offchain:
        # prefer explicit handle via --space-id or direct dao handle
        if getattr(args, "space_id", None):
            snapshot_id = args.space_id
        elif _looks_like_snapshot_handle(dao_input):
            snapshot_id = dao_input
        else:
            # allow backward compatibility using catalog lookup
            spaces = _load_snapshot_catalog(prompt_if_missing=False)
            if spaces:
                sid, label, slug = _resolve_from_snapshot(dao_input, spaces)
                if sid and not snapshot_id:
                    snapshot_id = sid
                    dao_name = label or dao_name
                    dao_slug = slug or dao_slug
    else:
        # optional direct override via --org-id
        if getattr(args, "org_id", None):
            tally_id = args.org_id
        orgs = _load_tally_catalog(prompt_if_missing=False)
        if orgs:
            tid, label, slug = _resolve_from_tally(dao_input, orgs)
            if tid and not tally_id:
                tally_id = tid
                dao_name = label or dao_name
                dao_slug = slug or dao_slug

    if is_offchain and not snapshot_id:
        logger.error(
            f"Could not resolve Snapshot space id for '{dao_input}'. Use an explicit handle like 'aave.eth'.")
        sys.exit(1)
    if not is_offchain and not tally_id:
        logger.error(
            f"Could not resolve Tally organization id for '{dao_input}'. Use an explicit slug/id.")
        sys.exit(1)

    # normalize slug for output directories
    if is_offchain and snapshot_id and _looks_like_snapshot_handle(dao_input):
        # prefer catalog name for slug otherwise fallback to handle prefix
        spaces = _load_snapshot_catalog()
        chosen_slug = None
        if spaces:
            for s in spaces:
                if normalize(s.get("id")) == normalize(snapshot_id):
                    label = s.get("name") or s.get("id")
                    chosen_slug = _slugify(label)
                    break
        if not chosen_slug:
            chosen_slug = snapshot_id.split(".")[0].lower()
        dao_slug = chosen_slug
        dao_name = dao_name or snapshot_id
    dao_slug = (dao_slug or normalize(dao_name)).lower()
    logger.info(f"Resolved DAO: name='{dao_name}', slug='{dao_slug}'")
    logger.info(f"IDs -> snapshot='{snapshot_id}', tally='{tally_id}'")

    # detect transfer data presence and mid-window timestamp for test mode
    target_ts: Optional[int] = None
    has_transfer_data = False
    # use normalized output slug to strip trailing dao tokens for paths
    output_slug = normalize_output_slug(dao_slug)
    target_ts, has_transfer_data = _detect_mid_window_timestamp(output_slug)
    if not has_transfer_data:
        logger.warning(
            "No transfer_data found for this DAO; will output proposals and votes only and skip subsequent analysis")

    # always attempt to export basic dao metadata first
    try:
        if is_offchain and snapshot_id:
            _export_metadata_for_dao("snapshot", snapshot_id, output_slug)
        elif (not is_offchain) and tally_id:
            _export_metadata_for_dao("tally", tally_id, output_slug)
    except Exception as e:
        logger.warning(f"DAO metadata export skipped due to error: {e}")

    if is_offchain:
        if not snapshot_id:
            logger.error(f"No Snapshot ID available for '{dao_name}'")
            sys.exit(1)
        # proposals (offchain)
        if args.only_votes:
            logger.info("Skipping offchain proposals fetch (--only-votes)")
        else:
            existing_prop = _existing_output_files(
                output_slug, is_offchain=True, kind="proposals")
            do_props = _confirm_overwrite("offchain proposals", existing_prop)
            if do_props:
                fetch_offchain(snapshot_id, output_slug)
            else:
                logger.info(
                    "Skipping offchain proposals fetch (user chose not to overwrite)")

        # votes (offchain)
        if args.only_proposals or args.no_votes:
            logger.info(
                "Skipping offchain votes (--only-proposals/--no-votes)")
        else:
            existing_votes = _existing_output_files(
                output_slug, is_offchain=True, kind="votes")
            do_votes = _confirm_overwrite("offchain votes", existing_votes)
            if do_votes:
                if args.test:
                    pid = _select_offchain_proposal_near_ts(
                        output_slug, target_ts)
                    if pid:
                        logger.info(
                            f"[TEST] Selected offchain proposal {pid} for test run")
                        _offchain_test_fetch_single(
                            snapshot_id, output_slug, pid)
                    else:
                        logger.warning(
                            "Could not select a test proposal; skipping votes")
                else:
                    _fetch_offchain_votes(snapshot_id, output_slug)
            else:
                logger.info(
                    "Skipping offchain votes fetch (user chose not to overwrite)")
    else:
        if not tally_id:
            logger.error(
                f"No Tally organization ID available for '{dao_name}'")
            sys.exit(1)
        # proposals (onchain)
        if args.only_votes:
            logger.info("Skipping onchain proposals fetch (--only-votes)")
        else:
            existing_prop = _existing_output_files(
                output_slug, is_offchain=False, kind="proposals")
            do_props = _confirm_overwrite("onchain proposals", existing_prop)
            if do_props:
                fetch_onchain(tally_id, dao_name, output_slug)
            else:
                logger.info(
                    "Skipping onchain proposals fetch (user chose not to overwrite)")

        # votes (onchain)
        if args.only_proposals or args.no_votes:
            logger.info("Skipping onchain votes (--only-proposals/--no-votes)")
        else:
            existing_votes = _existing_output_files(
                output_slug, is_offchain=False, kind="votes")
            do_votes = _confirm_overwrite("onchain votes", existing_votes)
            if do_votes:
                if args.test:
                    pid = _select_onchain_proposal_near_ts(
                        output_slug, target_ts)
                    if pid is not None:
                        logger.info(
                            f"[TEST] Selected onchain proposal {pid} for test run")
                        _onchain_test_fetch_single(output_slug, pid)
                    else:
                        logger.warning(
                            "Could not select a test onchain proposal; skipping votes")
                else:
                    _fetch_onchain_votes(output_slug)
            else:
                logger.info(
                    "Skipping onchain votes fetch (user chose not to overwrite)")


if __name__ == "__main__":
    main()
