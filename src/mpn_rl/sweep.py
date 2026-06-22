import itertools
from pathlib import Path
from typing import Any

import yaml

from mpn_rl.commands.train import TrainConfig


def experiments_for_sweep(
    sweep: dict[str, Any], sweep_name: str, results_dir: Path
) -> list[dict[str, Any]]:
    swept = {k: v for k, v in sweep.items() if isinstance(v, list)}
    fixed = {k: v for k, v in sweep.items() if not isinstance(v, list)}
    experiments_dir = results_dir / sweep_name / "experiments"
    keys = list(swept)
    return [
        {
            **fixed,
            **dict(zip(keys, combo)),
            "experiments_dir": experiments_dir,
            "sweep_name": sweep_name,
        }
        for combo in itertools.product(*(swept[k] for k in keys))
    ]


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
    sweep_job = (
        "universe = vanilla\n"
        "executable = condor/scripts/run_experiment.sh\n"
        f"initialdir = {Path.cwd()}\n"
        "request_cpus   = 2\n"
        "request_gpus   = 1\n"
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
