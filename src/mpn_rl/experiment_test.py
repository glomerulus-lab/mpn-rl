import copy
import json
import re
import tempfile

import torch

from mpn_rl.experiment import SCHEMA_VERSION, ExperimentManager


def test_init_creates_checkpoint_and_plot_dirs() -> None:
    with tempfile.TemporaryDirectory() as directory:
        manager = ExperimentManager(directory, "exp")
        assert manager.checkpoint_dir.is_dir() and manager.plot_dir.is_dir()


def test_init_generates_name_when_none() -> None:
    with tempfile.TemporaryDirectory() as directory:
        name = ExperimentManager(directory, None).experiment_name
    assert re.fullmatch(r"[a-z]+-[a-z]+", name) is not None


def test_init_uses_given_name_verbatim() -> None:
    with tempfile.TemporaryDirectory() as directory:
        assert ExperimentManager(directory, "exp").experiment_name == "exp"


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
    }


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
            epsilon=0.0,
            oracle_reward=2.0,
            pct_oracle=0.75,
            episode=3,
        )
        rows = manager.metrics_path.read_text().splitlines()
    assert json.loads(rows[0]) == {
        "experiment_name": "exp",
        "frame": 100,
        "episode": 3,
        "reward": 1.5,
        "length": 10,
        "loss": 0.2,
        "epsilon": 0.0,
        "oracle_reward": 2.0,
        "pct_oracle": 0.75,
    }
