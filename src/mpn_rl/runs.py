import json
from pathlib import Path

import pandas as pd


def find_run_files(
    filename: str, experiments_dir: Path | None, root: Path = Path()
) -> list[Path]:
    """Return `filename` (config.json/metrics.jsonl) for every run.

    With experiments_dir given, scan that flat directory directly (a single
    sweep's experiments/, or an arbitrary tree). Otherwise scan the default
    layout under root: ad-hoc runs in experiments/, sweep runs in
    results/<name>/experiments/.
    """
    if experiments_dir is not None:
        return list(experiments_dir.glob(f"*/{filename}"))
    return [
        *(root / "experiments").glob(f"*/{filename}"),
        *(root / "results").glob(f"*/experiments/*/{filename}"),
    ]


def load_runs(root: Path = Path()) -> pd.DataFrame:
    """One row per run: the flat config fields plus a `path` column."""
    records = []
    for p in find_run_files("config.json", None, root=root):
        config = json.loads(p.read_text())
        config["path"] = str(p.parent)
        records.append(config)
    return pd.DataFrame(records)
