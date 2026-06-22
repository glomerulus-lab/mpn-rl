import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mpn_rl.sweep import create_sweep, experiments_for_sweep


def test_experiments_for_sweep_expands_grid() -> None:
    sweep = {"hidden_dim": 64, "model_type": ["mpn", "lstm"], "num_layers": [1, 2]}
    experiments = Path("results/sweep_a/experiments")
    assert experiments_for_sweep(sweep, "sweep_a", Path("results")) == [
        {
            "sweep_name": "sweep_a",
            "hidden_dim": 64,
            "model_type": "mpn",
            "num_layers": 1,
            "experiments_dir": experiments,
        },
        {
            "sweep_name": "sweep_a",
            "hidden_dim": 64,
            "model_type": "mpn",
            "num_layers": 2,
            "experiments_dir": experiments,
        },
        {
            "sweep_name": "sweep_a",
            "hidden_dim": 64,
            "model_type": "lstm",
            "num_layers": 1,
            "experiments_dir": experiments,
        },
        {
            "sweep_name": "sweep_a",
            "hidden_dim": 64,
            "model_type": "lstm",
            "num_layers": 2,
            "experiments_dir": experiments,
        },
    ]


def test_experiments_for_sweep_handles_lists_and_single_values() -> None:
    sweep = {"hidden_dim": 64, "model_type": ["mpn", "lstm"]}
    experiments = Path("results/sweep_b/experiments")
    assert experiments_for_sweep(sweep, "sweep_b", Path("results")) == [
        {
            "sweep_name": "sweep_b",
            "hidden_dim": 64,
            "model_type": "mpn",
            "experiments_dir": experiments,
        },
        {
            "sweep_name": "sweep_b",
            "hidden_dim": 64,
            "model_type": "lstm",
            "experiments_dir": experiments,
        },
    ]


def test_experiments_for_sweep_yields_single_config_without_lists() -> None:
    experiments = Path("results/sweep_c/experiments")
    assert experiments_for_sweep({"hidden_dim": 64}, "sweep_c", Path("results")) == [
        {"sweep_name": "sweep_c", "hidden_dim": 64, "experiments_dir": experiments}
    ]


def test_create_sweep_errors_when_dir_exists() -> None:
    with tempfile.TemporaryDirectory() as directory:
        results_dir = Path(directory)
        config_file = results_dir / "mysweep.yaml"
        config_file.write_text("model_type: [mpn, lstm]\n")
        (results_dir / "mysweep").mkdir()
        with pytest.raises(FileExistsError):
            create_sweep(config_file, None, results_dir)


def test_create_sweep_writes_experiment_configs() -> None:
    with tempfile.TemporaryDirectory() as directory:
        results_dir = Path(directory)
        config_file = results_dir / "mysweep.yaml"
        config_file.write_text("model_type: [mpn, lstm]\n")
        sweep_dir, _ = create_sweep(config_file, None, results_dir)
        names = sorted(p.name for p in sweep_dir.glob("*.yaml"))
    assert names == ["mysweep-0000.yaml", "mysweep-0001.yaml"]


def test_create_sweep_writes_args_file() -> None:
    with tempfile.TemporaryDirectory() as directory:
        results_dir = Path(directory)
        config_file = results_dir / "mysweep.yaml"
        config_file.write_text("model_type: [mpn, lstm]\n")
        sweep_dir, _ = create_sweep(config_file, None, results_dir)
        args = (sweep_dir / "args.txt").read_text()
        expected = (
            f"--config {sweep_dir / 'mysweep-0000.yaml'}\n"
            f"--config {sweep_dir / 'mysweep-0001.yaml'}"
        )
    assert args == expected


def test_create_sweep_writes_condor_job() -> None:
    with tempfile.TemporaryDirectory() as directory:
        results_dir = Path(directory)
        config_file = results_dir / "mysweep.yaml"
        config_file.write_text("model_type: [mpn, lstm]\n")
        sweep_dir, _ = create_sweep(config_file, None, results_dir)
        job = (sweep_dir / "sweep.job").read_text()
    assert "executable = condor/scripts/run_experiment.sh" in job
    assert "initialdir =" in job
    assert f"output = {sweep_dir}/logs/{sweep_dir.name}-$INT(Process,%04d).out" in job
    assert f"Queue args from {sweep_dir}/args.txt" in job


def test_create_sweep_returns_experiment_count() -> None:
    with tempfile.TemporaryDirectory() as directory:
        results_dir = Path(directory)
        config_file = results_dir / "mysweep.yaml"
        config_file.write_text("model_type: [mpn, lstm]\nnum_layers: [1, 2]\n")
        _, count = create_sweep(config_file, None, results_dir)
    assert count == 4


def test_create_sweep_names_experiments_by_index() -> None:
    with tempfile.TemporaryDirectory() as directory:
        results_dir = Path(directory)
        config_file = results_dir / "mysweep.yaml"
        config_file.write_text("model_type: [mpn, lstm]\n")
        sweep_dir, _ = create_sweep(config_file, None, results_dir)
        config = yaml.safe_load((sweep_dir / "mysweep-0001.yaml").read_text())
    assert config["experiment_name"] == "mysweep-0001"


def test_create_sweep_rejects_unknown_keys() -> None:
    with tempfile.TemporaryDirectory() as directory:
        results_dir = Path(directory)
        config_file = results_dir / "mysweep.yaml"
        config_file.write_text("bogus_key: 1\n")
        with pytest.raises(ValidationError):
            create_sweep(config_file, None, results_dir)
