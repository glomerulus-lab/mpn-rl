from pathlib import Path


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
