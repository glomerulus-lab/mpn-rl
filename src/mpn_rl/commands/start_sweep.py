import subprocess
from pathlib import Path
from typing import Annotated

import tyro
from pydantic import BaseModel, Field

from mpn_rl import git
from mpn_rl.sweep import create_sweep


class StartSweepCommand(BaseModel):
    config_file: tyro.conf.Positional[Path]
    name: Annotated[
        str | None, tyro.conf.arg(help="Sweep name to use instead of config name")
    ] = None
    results_dir: Path = Path("results")
    allow_dirty: Annotated[
        bool,
        tyro.conf.arg(help="Submit even if the working tree has uncommitted changes"),
    ] = False
    request_memory_gb: Annotated[
        int, tyro.conf.arg(help="Per-job memory request in GB"), Field(ge=1)
    ] = 2
    max_materialize: Annotated[
        int,
        tyro.conf.arg(help="Max jobs materialized/running at once"),
        Field(ge=1),
    ] = 50


def start_sweep(command: StartSweepCommand) -> None:
    if not command.allow_dirty and git.is_dirty():
        print(
            "Working tree has uncommitted changes; commit them or pass --allow-dirty:"
        )
        print(git.status())
        return
    sweep_dir, count = create_sweep(
        command.config_file,
        command.name,
        command.results_dir,
        request_memory_gb=command.request_memory_gb,
        max_materialize=command.max_materialize,
    )
    answer = input(
        f"Submit {count} experiments from {sweep_dir} "
        f"(up to {command.max_materialize} running at once, "
        f"{command.request_memory_gb}GB each)? [y/N] "
    )
    if answer.strip().lower() != "y":
        print(f"Not submitted. Sweep is at {sweep_dir}")
        return
    subprocess.run(["condor_submit", str(sweep_dir / "sweep.job")], check=True)
    print(f"Submitted sweep at {sweep_dir}")
