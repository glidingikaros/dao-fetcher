### DAO Governance Data Fetcher

Fetch proposals and votes from DAOs using offchain Snapshot.org and onchain Tally.xyz sources. Outputs CSV/Parquet with automatic retries, throttling, and resume support.

## Highlights
- **Sources**: Snapshot (offchain) and Tally (onchain)
- **Artifacts**: Proposals, votes, DAO metadata
- **Formats**: CSV and Parquet
- **CLI + Python API**: Interactive and non-interactive
- **Smart DAO resolution**: Via slugs in catalogs

## Requirements
- Python 3.10+

## Install
```bash
pip install -r requirements.txt
```

## Configure API keys
Create a `.env` in the repo root:
```bash
TALLY_API_KEY=your_tally_api_key   # required for onchain
SNAPSHOT_API_KEY=your_snapshot_key # optional, improves limits
```
Get keys: Tally (`https://www.tally.xyz/settings`), Snapshot (`https://snapshot.org`).

## Quick start (CLI)
```bash
# Interactive
python dao_fetcher.py

# Snapshot (offchain)
python dao_fetcher.py --source snapshot --dao aave

# Tally (onchain)
python dao_fetcher.py --source tally --dao uniswap
```

Common flags (see `-h` for all): `--only-proposals`, `--only-votes`, `--no-votes`, `--test`, `--overwrite yes`, `--space-id`, `--org-id`, `--verbose`, `--quiet`.

## Bootstrap DAO catalog (first run)
```bash
python src/fetch_dao_list.py --mode both
```
This generates filtered/unfiltered Parquet catalogs under `catalogue/` for Snapshot and Tally.

## Smoke test
```bash
# Offchain proposals in test-like mode (limited pages via the module)
python src/fetch_offchain_proposals.py aave.eth --enhanced -v

# Then offchain votes consolidation
python src/fetch_offchain_votes.py aave.eth -v

# Onchain proposals (requires TALLY_API_KEY)
python src/fetch_onchain_proposals.py --org-slug uniswap --output data/uniswap/governance_data -v
```

## Programmatic usage
```python
# Offchain proposals (Snapshot)
from src.fetch_offchain_proposals import fetch_all_proposals, write_proposals_to_csv
proposals = fetch_all_proposals("aave.eth", enhanced=True)
write_proposals_to_csv(proposals, "output.csv")

# Onchain proposals (Tally)
from src.fetch_onchain_proposals import TallyClient, export_proposals_for_organization
client = TallyClient(api_key="YOUR_TALLY_KEY")
export_proposals_for_organization(client, "uniswap", "./data/uniswap/governance_data", is_slug=True)

# Votes
from src.fetch_offchain_votes import SnapshotVoteFetcher
SnapshotVoteFetcher().fetch_votes_by_space("aave.eth", "proposals.csv", "./output", resume=False)
```

## Output layout
```
data/
├── <dao_slug>/
│   ├── metadata.parquet
│   └── governance_data/
│       ├── offchain_proposals.csv/.parquet
│       ├── offchain_votes.csv/.parquet
│       ├── onchain_proposals.csv/.parquet
│       └── onchain_votes.csv/.parquet
```

Notes:
- DAO slugs are resolved automatically; you can override with `--space-id` or `--org-id`.
- Resume is supported: existing files are detected and skipped/continued.

## Tips & limits
- Use `--test` to fetch a single proposal's votes
- Snapshot limits improve with `SNAPSHOT_API_KEY`
- Large DAOs can take hours. Run off-peak for fewer 429s

## Troubleshooting (quick)
- Cannot resolve IDs: check spelling in relevant catalogue file
- Missing `TALLY_API_KEY`: add to `.env` or export in shell
- 429s: automatic backoff; add API keys; retry later

## Project structure
```
dao_fetcher/
├── dao_fetcher.py
├── src/
│   ├── fetch_dao_list.py
│   ├── fetch_dao_metadata.py
│   ├── fetch_offchain_proposals.py
│   ├── fetch_offchain_votes.py
│   ├── fetch_onchain_proposals.py
│   └── fetch_onchain_votes.py
├── catalogue/
└── data/
```
