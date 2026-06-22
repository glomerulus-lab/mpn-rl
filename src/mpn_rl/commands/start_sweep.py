import subprocess
from pathlib import Path
from typing import Annotated

import tyro
from pydantic import BaseModel

from mpn_rl.sweep import create_sweep


class StartSweepCommand(BaseModel):
    config_file: tyro.conf.Positional[Path]
    name: Annotated[
        str | None, tyro.conf.arg(help="Sweep name to use instead of config name")
    ] = None
    results_dir: Path = Path("results")


def start_sweep(command: StartSweepCommand) -> None:
    sweep_dir, count = create_sweep(
        command.config_file, command.name, command.results_dir
    )
    answer = input(f"Submit {count} experiments from {sweep_dir}? [y/N] ")
    if answer.strip().lower() != "y":
        print(f"Not submitted. Sweep is at {sweep_dir}")
        return
    subprocess.run(["condor_submit", str(sweep_dir / "sweep.job")], check=True)
    print(f"Submitted sweep at {sweep_dir}")
