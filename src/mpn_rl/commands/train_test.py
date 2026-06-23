import tempfile
from pathlib import Path

from mpn_rl.commands.train import resolve_train_config


def test_resolve_train_config_uses_defaults_without_config():
    assert resolve_train_config([]).model.model_type == "lstm"


def test_resolve_train_config_applies_config_file():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.yaml"
        path.write_text("model:\n  model_type: mpn\n  eta_init: 0.5")
        cfg = resolve_train_config(["--config", str(path)])
    assert cfg.model.eta_init == 0.5


def test_resolve_train_config_cli_overrides_config_file():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.yaml"
        path.write_text("model:\n  model_type: mpn\n  eta_init: 0.5")
        cfg = resolve_train_config(["--config", str(path), "mpn", "--eta-init", "0.7"])
    assert cfg.model.eta_init == 0.7
