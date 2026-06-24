import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from mpn_rl.runs import find_run_files, load_runs


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")


def _write_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config))


def test_find_run_files_scans_experiments_and_results() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _touch(root / "experiments" / "adhoc" / "config.json")
        _touch(root / "results" / "swp" / "experiments" / "run1" / "config.json")
        files = find_run_files("config.json", None, root=root)
        names = sorted(p.parent.name for p in files)
    assert names == ["adhoc", "run1"]


def test_find_run_files_scans_given_experiments_dir() -> None:
    with tempfile.TemporaryDirectory() as directory:
        experiments_dir = Path(directory)
        _touch(experiments_dir / "run1" / "config.json")
        _touch(experiments_dir / "run2" / "config.json")
        names = sorted(
            p.parent.name for p in find_run_files("config.json", experiments_dir)
        )
    assert names == ["run1", "run2"]


def test_load_runs_adds_path_and_flat_config_columns() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        run_dir = root / "results" / "swp" / "experiments" / "run1"
        _write_config(
            run_dir / "config.json", {"model_type": "mpn", "activation": "tanh"}
        )
        df = load_runs(root=root)
    assert df.loc[0, "model_type"] == "mpn"
    assert df.loc[0, "activation"] == "tanh"
    assert df.loc[0, "path"] == str(run_dir)


def test_load_runs_unions_heterogeneous_configs() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _write_config(
            root / "experiments" / "mpn-run" / "config.json",
            {"model_type": "mpn", "mpn_bias": True},
        )
        _write_config(
            root / "experiments" / "lstm-run" / "config.json",
            {"model_type": "lstm", "forget_bias": 1.0},
        )
        df = load_runs(root=root).set_index("model_type")
    assert df.loc["mpn", "mpn_bias"]
    assert df.loc["lstm", "forget_bias"] == 1.0
    assert pd.isna(df.loc["mpn", "forget_bias"])
