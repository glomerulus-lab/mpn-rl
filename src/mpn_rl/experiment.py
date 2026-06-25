"""
Experiment management for MPN-RL

Handles:
- Saving/loading model weights and optimizer states
- Experiment directory structure
- Configuration management
- Training history tracking
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
import torch.nn as nn
import uuid6

from mpn_rl.git import git_revision

SCHEMA_VERSION = 1


class ExperimentManager:
    """
    Manages experiment directory structure and file I/O.

    Directory structure (root set by experiments_dir, default experiments/):
        {experiments_dir}/{experiment_name}/
        ├── config.json
        ├── training_history.json
        ├── checkpoints/
        │   ├── best_model.pt
        │   ├── checkpoint_100.pt
        │   └── final_model.pt
        └── plots/
            └── training_curves.png
    """

    def __init__(
        self,
        experiments_dir: str | Path = "experiments",
        experiment_name: str | None = None,
    ):
        """
        Args:
            experiments_dir: Root directory for all experiments (default: experiments/)
            experiment_name: Experiment directory name (defaults to the generated id)
        """
        self.experiment_id = str(uuid6.uuid7())
        if experiment_name is None:
            experiment_name = self.experiment_id

        self.experiment_name = experiment_name
        self.exp_dir = Path(experiments_dir) / experiment_name

        # Create directory structure
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.plot_dir = self.exp_dir / "plots"

        for dir_path in [self.checkpoint_dir, self.plot_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.config_path = self.exp_dir / "config.json"
        self.metrics_path = self.exp_dir / "metrics.jsonl"

    def save_config(self, config: Dict[str, Any]) -> None:
        """Save experiment configuration."""
        created_at = datetime.now().isoformat()
        config = {
            **config,
            "schema_version": SCHEMA_VERSION,
            "created_at": created_at,
            "git": git_revision(),
        }
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        print(f"Saved config to {self.config_path}")

    def load_config(self) -> Dict[str, Any]:
        """Load experiment configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        with open(self.config_path, "r") as f:
            config: Dict[str, Any] = json.load(f)
        return config

    def save_model(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        checkpoint_name: str = "model.pt",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save model checkpoint.

        Args:
            model: Model to save (nn.Module)
            optimizer: Optimizer to save (optional, for resuming training)
            checkpoint_name: Name of checkpoint file
            metadata: Additional metadata (episode, reward, etc.)
        """
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "metadata": metadata or {},
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    def load_model(
        self,
        model: nn.Module,
        checkpoint_name: str = "best_model.pt",
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.

        Args:
            model: Model to load weights into
            checkpoint_name: Name of checkpoint file
            optimizer: Optimizer to load state into (if resuming training)
            device: Device to load model onto

        Returns:
            metadata: Dictionary with episode, reward, etc.
        """
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"Loaded checkpoint from {checkpoint_path}")
        metadata: Dict[str, Any] = checkpoint.get("metadata", {})
        return metadata

    def append_training_history(
        self,
        frames: int,
        reward: float,
        length: int,
        loss: float,
        epsilon: float,
        oracle_reward: Optional[float] = None,
        pct_oracle: Optional[float] = None,
        episode: Optional[int] = None,
    ) -> None:
        """Append a single eval step to metrics.jsonl."""
        with open(self.metrics_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "experiment_name": self.experiment_name,
                        "frame": int(frames),
                        "episode": int(episode) if episode is not None else None,
                        "reward": float(reward),
                        "length": int(length),
                        "loss": float(loss),
                        "epsilon": float(epsilon),
                        "oracle_reward": (
                            float(oracle_reward) if oracle_reward is not None else None
                        ),
                        "pct_oracle": (
                            float(pct_oracle) if pct_oracle is not None else None
                        ),
                    }
                )
                + "\n"
            )

    def __repr__(self) -> str:
        return f"ExperimentManager('{self.experiment_name}', dir='{self.exp_dir}')"


def find_experiment_files(
    filename: str, experiments_dir: Path | None, root: Path = Path()
) -> list[Path]:
    """Return `filename` (config.json/metrics.jsonl) for every experiment.

    With experiments_dir given, scan that flat directory directly (a single
    sweep's experiments/, or an arbitrary tree). Otherwise scan the default
    layout under root: ad-hoc experiments in experiments/, sweep experiments in
    results/<name>/experiments/.
    """
    if experiments_dir is not None:
        return list(experiments_dir.glob(f"*/{filename}"))
    return [
        *(root / "experiments").glob(f"*/{filename}"),
        *(root / "results").glob(f"*/experiments/*/{filename}"),
    ]


def load_experiments(root: Path = Path()) -> pd.DataFrame:
    """One row per experiment: the flat config fields plus a `path` column."""
    records = []
    for p in find_experiment_files("config.json", None, root=root):
        config = json.loads(p.read_text())
        config["path"] = str(p.parent)
        records.append(config)
    return pd.DataFrame(records)
