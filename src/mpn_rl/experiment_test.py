import copy
import json
import tempfile
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from mpn_rl.experiment import (
    SCHEMA_VERSION,
    ExperimentManager,
    find_experiment_files,
    load_experiments,
)


def test_init_creates_checkpoint_and_plot_dirs() -> None:
    with tempfile.TemporaryDirectory() as directory:
        manager = ExperimentManager(directory, "exp")
        assert manager.checkpoint_dir.is_dir() and manager.plot_dir.is_dir()


def test_init_uses_given_name_verbatim() -> None:
    with tempfile.TemporaryDirectory() as directory:
        assert ExperimentManager(directory, "exp").experiment_name == "exp"


def test_init_uses_id_as_name_when_none() -> None:
    with tempfile.TemporaryDirectory() as directory:
        manager = ExperimentManager(directory, None)
    assert manager.experiment_name == manager.experiment_id


def test_init_generates_uuid7_id() -> None:
    with tempfile.TemporaryDirectory() as directory:
        experiment_id = ExperimentManager(directory, "exp").experiment_id
    assert uuid.UUID(experiment_id).version == 7


def test_save_config_roundtrips() -> None:
    config = {"model_type": "mpn", "hidden_dim": 128, "eta_init": 0.01}
    with tempfile.TemporaryDirectory() as directory:
        manager = ExperimentManager(directory, "exp")
        manager.save_config(config)
        loaded = manager.load_config()
    assert loaded == {
        **config,
        "schema_version": SCHEMA_VERSION,
        "created_at": loaded["created_at"],
        "git": loaded["git"],
    }
    assert set(loaded["git"]) == {"commit", "dirty"}


def test_save_model_roundtrips() -> None:
    with tempfile.TemporaryDirectory() as directory:
        manager = ExperimentManager(directory, "exp")
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        manager.save_model(
            model,
            optimizer=optimizer,
            checkpoint_name="best_model.pt",
            metadata={"episode": 7},
        )

        loaded_model = torch.nn.Linear(2, 2)
        loaded_optimizer = torch.optim.SGD(
            loaded_model.parameters(), lr=0.1, momentum=0.9
        )
        metadata = manager.load_model(
            loaded_model, checkpoint_name="best_model.pt", optimizer=loaded_optimizer
        )

    assert metadata == {"episode": 7}
    assert torch.equal(loaded_model.weight, model.weight)
    torch.testing.assert_close(loaded_optimizer.state_dict(), optimizer.state_dict())


def test_load_model_skips_optimizer_when_not_saved() -> None:
    with tempfile.TemporaryDirectory() as directory:
        manager = ExperimentManager(directory, "exp")
        manager.save_model(torch.nn.Linear(2, 2), checkpoint_name="best_model.pt")

        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        before = copy.deepcopy(optimizer.state_dict())
        manager.load_model(model, checkpoint_name="best_model.pt", optimizer=optimizer)
    torch.testing.assert_close(optimizer.state_dict(), before)


def test_append_training_history_writes_row() -> None:
    with tempfile.TemporaryDirectory() as directory:
        manager = ExperimentManager(directory, "exp")
        manager.append_training_history(
            frames=100,
            reward=1.5,
            length=10,
            loss=0.2,
            oracle_reward=2.0,
            pct_oracle=0.75,
            episode=3,
        )
        rows = manager.metrics_path.read_text().splitlines()
    assert json.loads(rows[0]) == {
        "experiment_name": manager.experiment_name,
        "frame": 100,
        "episode": 3,
        "reward": 1.5,
        "length": 10,
        "loss": 0.2,
        "oracle_reward": 2.0,
        "pct_oracle": 0.75,
    }


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")


def _write_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config))


def test_find_experiment_files_scans_experiments_and_results() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _touch(root / "experiments" / "adhoc" / "config.json")
        _touch(root / "results" / "swp" / "experiments" / "run1" / "config.json")
        files = find_experiment_files("config.json", None, root=root)
        names = sorted(p.parent.name for p in files)
    assert names == ["adhoc", "run1"]


def test_find_experiment_files_scans_given_experiments_dir() -> None:
    with tempfile.TemporaryDirectory() as directory:
        experiments_dir = Path(directory)
        _touch(experiments_dir / "run1" / "config.json")
        _touch(experiments_dir / "run2" / "config.json")
        names = sorted(
            p.parent.name for p in find_experiment_files("config.json", experiments_dir)
        )
    assert names == ["run1", "run2"]


def test_load_experiments_adds_path_and_flat_config_columns() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        run_dir = root / "results" / "swp" / "experiments" / "run1"
        _write_config(
            run_dir / "config.json", {"model_type": "mpn", "activation": "tanh"}
        )
        df = load_experiments(root=root)
    assert df.loc[0, "model_type"] == "mpn"
    assert df.loc[0, "activation"] == "tanh"
    assert df.loc[0, "path"] == str(run_dir)


def test_load_experiments_unions_heterogeneous_configs() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _write_config(
            root / "experiments" / "mpn-run" / "config.json",
            {"model_type": "mpn", "mpn_bias": True},
        )
        _write_config(
            root / "experiments" / "lstm-run" / "config.json",
            {"model_type": "lstm"},
        )
        df = load_experiments(root=root).set_index("model_type")
    assert df.loc["mpn", "mpn_bias"]
    assert pd.isna(df.loc["lstm", "mpn_bias"])
