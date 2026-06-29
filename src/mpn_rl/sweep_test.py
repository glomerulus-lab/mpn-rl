import json
import stat
import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mpn_rl.sweep import create_sweep, experiments_for_sweep, snapshot_code


def test_experiments_for_sweep_expands_grid() -> None:
    sweep = {
        "hidden_dim": 64,
        "model": [{"model_type": "mpn"}, {"model_type": "lstm"}],
        "num_layers": [1, 2],
    }
    experiments = Path("results/sweep_a/experiments")
    assert experiments_for_sweep(sweep, "sweep_a", Path("results")) == [
        {
            "sweep_name": "sweep_a",
            "hidden_dim": 64,
            "model": {"model_type": "mpn"},
            "num_layers": 1,
            "experiments_dir": experiments,
        },
        {
            "sweep_name": "sweep_a",
            "hidden_dim": 64,
            "model": {"model_type": "mpn"},
            "num_layers": 2,
            "experiments_dir": experiments,
        },
        {
            "sweep_name": "sweep_a",
            "hidden_dim": 64,
            "model": {"model_type": "lstm"},
            "num_layers": 1,
            "experiments_dir": experiments,
        },
        {
            "sweep_name": "sweep_a",
            "hidden_dim": 64,
            "model": {"model_type": "lstm"},
            "num_layers": 2,
            "experiments_dir": experiments,
        },
    ]


def test_experiments_for_sweep_handles_lists_and_single_values() -> None:
    sweep = {"hidden_dim": 64, "model": [{"model_type": "mpn"}, {"model_type": "lstm"}]}
    experiments = Path("results/sweep_b/experiments")
    assert experiments_for_sweep(sweep, "sweep_b", Path("results")) == [
        {
            "sweep_name": "sweep_b",
            "hidden_dim": 64,
            "model": {"model_type": "mpn"},
            "experiments_dir": experiments,
        },
        {
            "sweep_name": "sweep_b",
            "hidden_dim": 64,
            "model": {"model_type": "lstm"},
            "experiments_dir": experiments,
        },
    ]


def test_experiments_for_sweep_yields_single_config_without_lists() -> None:
    experiments = Path("results/sweep_c/experiments")
    assert experiments_for_sweep({"hidden_dim": 64}, "sweep_c", Path("results")) == [
        {"sweep_name": "sweep_c", "hidden_dim": 64, "experiments_dir": experiments}
    ]


def test_experiments_for_sweep_expands_list_elements() -> None:
    sweep = {
        "model": [
            {"model_type": "mpn", "eta_init": [0.01, 0.05]},
            {"model_type": "lstm"},
        ]
    }
    experiments = Path("results/sweep_d/experiments")
    assert experiments_for_sweep(sweep, "sweep_d", Path("results")) == [
        {
            "sweep_name": "sweep_d",
            "model": {"model_type": "mpn", "eta_init": 0.01},
            "experiments_dir": experiments,
        },
        {
            "sweep_name": "sweep_d",
            "model": {"model_type": "mpn", "eta_init": 0.05},
            "experiments_dir": experiments,
        },
        {
            "sweep_name": "sweep_d",
            "model": {"model_type": "lstm"},
            "experiments_dir": experiments,
        },
    ]


def test_experiments_for_sweep_expands_dict_values() -> None:
    sweep = {"model": {"model_type": "mpn", "eta_init": [0.01, 0.05]}}
    experiments = Path("results/sweep_e/experiments")
    assert experiments_for_sweep(sweep, "sweep_e", Path("results")) == [
        {
            "sweep_name": "sweep_e",
            "model": {"model_type": "mpn", "eta_init": 0.01},
            "experiments_dir": experiments,
        },
        {
            "sweep_name": "sweep_e",
            "model": {"model_type": "mpn", "eta_init": 0.05},
            "experiments_dir": experiments,
        },
    ]


def test_create_sweep_errors_when_dir_exists() -> None:
    with tempfile.TemporaryDirectory() as directory:
        results_dir = Path(directory)
        config_file = results_dir / "mysweep.yaml"
        config_file.write_text("model: [{model_type: mpn}, {model_type: lstm}]\n")
        (results_dir / "mysweep").mkdir()
        with pytest.raises(FileExistsError):
            create_sweep(config_file, None, results_dir)


def test_create_sweep_writes_experiment_configs() -> None:
    with tempfile.TemporaryDirectory() as directory:
        results_dir = Path(directory)
        config_file = results_dir / "mysweep.yaml"
        config_file.write_text("model: [{model_type: mpn}, {model_type: lstm}]\n")
        sweep_dir, _ = create_sweep(config_file, None, results_dir)
        names = sorted(p.name for p in (sweep_dir / "configs").glob("*.yaml"))
    assert names == ["mysweep-0000.yaml", "mysweep-0001.yaml"]


def test_create_sweep_writes_args_file() -> None:
    with tempfile.TemporaryDirectory() as directory:
        results_dir = Path(directory)
        config_file = results_dir / "mysweep.yaml"
        config_file.write_text("model: [{model_type: mpn}, {model_type: lstm}]\n")
        sweep_dir, _ = create_sweep(config_file, None, results_dir)
        args = (sweep_dir / "args.txt").read_text()
        expected = (
            f"--config {sweep_dir / 'configs' / 'mysweep-0000.yaml'}\n"
            f"--config {sweep_dir / 'configs' / 'mysweep-0001.yaml'}"
        )
    assert args == expected


def test_create_sweep_writes_condor_job() -> None:
    with tempfile.TemporaryDirectory() as directory:
        results_dir = Path(directory)
        config_file = results_dir / "mysweep.yaml"
        config_file.write_text("model: [{model_type: mpn}, {model_type: lstm}]\n")
        sweep_dir, _ = create_sweep(config_file, None, results_dir)
        job = (sweep_dir / "sweep.job").read_text()
        run_script = (sweep_dir / "code" / "run_experiment.sh").resolve()
    assert f"executable = {run_script}" in job
    assert "initialdir =" in job
    assert f"output = {sweep_dir}/logs/{sweep_dir.name}-$INT(Process,%04d).out" in job
    assert f"Queue args from {sweep_dir}/args.txt" in job


def test_create_sweep_requests_gpu_for_gpu_device() -> None:
    with tempfile.TemporaryDirectory() as directory:
        results_dir = Path(directory)
        config_file = results_dir / "mysweep.yaml"
        config_file.write_text("model: [{model_type: mpn}]\ndevice: gpu\n")
        sweep_dir, _ = create_sweep(config_file, None, results_dir)
        job = (sweep_dir / "sweep.job").read_text()
    assert "request_gpus   = 1" in job


def test_create_sweep_requests_no_gpu_for_cpu_device() -> None:
    with tempfile.TemporaryDirectory() as directory:
        results_dir = Path(directory)
        config_file = results_dir / "mysweep.yaml"
        config_file.write_text("model: [{model_type: mpn}]\ndevice: cpu\n")
        sweep_dir, _ = create_sweep(config_file, None, results_dir)
        job = (sweep_dir / "sweep.job").read_text()
    assert "request_gpus   = 0" in job


def test_create_sweep_rejects_mixed_devices() -> None:
    with tempfile.TemporaryDirectory() as directory:
        results_dir = Path(directory)
        config_file = results_dir / "mysweep.yaml"
        config_file.write_text("model: [{model_type: mpn}]\ndevice: [cpu, gpu]\n")
        with pytest.raises(ValueError, match="cannot mix devices"):
            create_sweep(config_file, None, results_dir)


def test_create_sweep_returns_experiment_count() -> None:
    with tempfile.TemporaryDirectory() as directory:
        results_dir = Path(directory)
        config_file = results_dir / "mysweep.yaml"
        config_file.write_text(
            "model: [{model_type: mpn}, {model_type: lstm}]\nnum_layers: [1, 2]\n"
        )
        _, count = create_sweep(config_file, None, results_dir)
    assert count == 4


def test_create_sweep_names_experiments_by_index() -> None:
    with tempfile.TemporaryDirectory() as directory:
        results_dir = Path(directory)
        config_file = results_dir / "mysweep.yaml"
        config_file.write_text("model: [{model_type: mpn}, {model_type: lstm}]\n")
        sweep_dir, _ = create_sweep(config_file, None, results_dir)
        config = yaml.safe_load(
            (sweep_dir / "configs" / "mysweep-0001.yaml").read_text()
        )
    assert config["experiment_name"] == "mysweep-0001"


def test_create_sweep_rejects_unknown_keys() -> None:
    with tempfile.TemporaryDirectory() as directory:
        results_dir = Path(directory)
        config_file = results_dir / "mysweep.yaml"
        config_file.write_text("bogus_key: 1\n")
        with pytest.raises(ValidationError):
            create_sweep(config_file, None, results_dir)


def test_snapshot_code_copies_package_and_entrypoint() -> None:
    with tempfile.TemporaryDirectory() as directory:
        sweep_dir = Path(directory) / "mysweep"
        sweep_dir.mkdir()
        snapshot_code(sweep_dir)
        code = sweep_dir / "code"
        assert (code / "main_a2c.py").exists()
        assert (code / "src" / "mpn_rl" / "__init__.py").exists()


def test_snapshot_code_freezes_revision() -> None:
    with tempfile.TemporaryDirectory() as directory:
        sweep_dir = Path(directory) / "mysweep"
        sweep_dir.mkdir()
        snapshot_code(sweep_dir)
        revision = json.loads(
            (sweep_dir / "code" / "src" / "mpn_rl" / "REVISION").read_text()
        )
    assert set(revision) == {"commit", "dirty"}
    assert revision["commit"]


def test_snapshot_code_is_read_only() -> None:
    with tempfile.TemporaryDirectory() as directory:
        sweep_dir = Path(directory) / "mysweep"
        sweep_dir.mkdir()
        snapshot_code(sweep_dir)
        train = sweep_dir / "code" / "src" / "mpn_rl" / "commands" / "train.py"
        mode = train.stat().st_mode
    assert stat.S_IMODE(mode) == 0o444


def test_snapshot_code_excludes_pycache() -> None:
    with tempfile.TemporaryDirectory() as directory:
        sweep_dir = Path(directory) / "mysweep"
        sweep_dir.mkdir()
        snapshot_code(sweep_dir)
        assert not list((sweep_dir / "code" / "src").rglob("__pycache__"))
