import itertools
import json
import shlex
import shutil
from pathlib import Path
from typing import Any

import yaml

from mpn_rl.commands.train import TrainConfig
from mpn_rl.git import git_revision

_SNAPSHOT_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.swp", "*.egg-info")


def _expand(value: Any) -> list[Any]:
    if isinstance(value, dict):
        keys = list(value)
        return [
            dict(zip(keys, combo))
            for combo in itertools.product(*(_expand(v) for v in value.values()))
        ]
    if isinstance(value, list):
        return [variant for element in value for variant in _expand(element)]
    return [value]


def experiments_for_sweep(
    sweep: dict[str, Any], sweep_name: str, results_dir: Path
) -> list[dict[str, Any]]:
    experiments_dir = results_dir / sweep_name / "experiments"
    return [
        {**experiment, "experiments_dir": experiments_dir, "sweep_name": sweep_name}
        for experiment in _expand(sweep)
    ]


def snapshot_code(sweep_dir: Path) -> Path:
    """Freeze the code a sweep runs into an immutable, self-contained snapshot."""
    code_dir = sweep_dir / "code"
    shutil.copytree("src", code_dir / "src", ignore=_SNAPSHOT_IGNORE)
    shutil.copy("main_a2c.py", code_dir / "main_a2c.py")

    revision = code_dir / "src" / "mpn_rl" / "REVISION"
    revision.write_text(json.dumps(git_revision()))

    run_script = code_dir / "run_experiment.sh"
    src = (code_dir / "src").resolve()
    main = (code_dir / "main_a2c.py").resolve()
    venv = (Path.cwd() / ".venv").resolve()
    run_script.write_text(
        "#!/bin/bash\n"
        "set -e\n"
        f"source {shlex.quote(str(venv))}/bin/activate\n"
        f"exec env PYTHONPATH={shlex.quote(str(src))} \\\n"
        f'  python {shlex.quote(str(main))} train-neurogym "$@"\n'
    )
    run_script.chmod(0o755)

    # Make the snapshot read-only so an accidental edit-in-place can't silently
    # reintroduce the multi-version bug. Files only — dirs stay writable so the
    # sweep directory can still be removed with `rm -rf`.
    for f in code_dir.rglob("*"):
        if f.is_file():
            f.chmod(f.stat().st_mode & ~0o222)
    return run_script


def create_sweep(
    config_file: Path, sweep_name: str | None, results_dir: Path
) -> tuple[Path, int]:
    config = yaml.safe_load(config_file.read_text())
    name = sweep_name if sweep_name is not None else config_file.stem
    sweep_dir = results_dir / name
    if sweep_dir.exists():
        raise FileExistsError(f"Sweep directory already exists: {sweep_dir}")

    experiments = [
        TrainConfig(**{**experiment, "experiment_name": f"{name}-{i:04d}"})
        for i, experiment in enumerate(experiments_for_sweep(config, name, results_dir))
    ]
    devices = {experiment.device for experiment in experiments}
    if len(devices) > 1:
        raise ValueError("Sweeps cannot mix devices")
    request_gpus = 1 if devices == {"gpu"} else 0
    sweep_dir.mkdir(parents=True)
    config_paths = []
    for i, experiment in enumerate(experiments):
        path = sweep_dir / f"{name}-{i:04d}.yaml"
        path.write_text(
            yaml.safe_dump(experiment.model_dump(mode="json"), sort_keys=False)
        )
        config_paths.append(path)

    args = "\n".join(f"--config {path}" for path in config_paths)
    (sweep_dir / "args.txt").write_text(args)

    (sweep_dir / "logs").mkdir()
    run_script = snapshot_code(sweep_dir)
    sweep_job = (
        "universe = vanilla\n"
        f"executable = {run_script.resolve()}\n"
        f"initialdir = {Path.cwd()}\n"
        "request_cpus   = 2\n"
        f"request_gpus   = {request_gpus}\n"
        "request_memory = 8GB\n"
        f"output = {sweep_dir}/logs/{name}-$INT(Process,%04d).out\n"
        f"error  = {sweep_dir}/logs/{name}-$INT(Process,%04d).err\n"
        f"log    = {sweep_dir}/logs/{name}-$INT(Process,%04d).log\n"
        "+CSCI_GrpDesktop = true\n"
        "max_materialize = 50\n"
        "arguments = $(args)\n"
        f"Queue args from {sweep_dir}/args.txt\n"
    )
    (sweep_dir / "sweep.job").write_text(sweep_job)
    return sweep_dir, len(experiments)
