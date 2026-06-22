import tempfile
from pathlib import Path

from mpn_rl.runs import find_run_files


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")


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
