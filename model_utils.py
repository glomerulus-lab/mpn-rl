"""
Model save/load utilities and experiment management for MPN-DQN

Handles:
- Saving/loading model weights and optimizer states
- Experiment directory structure
- Configuration management
- Training history tracking
- Random experiment name generation
- Replay buffer and TD loss computation
"""

import json
import os
import random
import sqlite3
import time
from collections import deque, namedtuple
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

# Experience tuple for replay buffer
Experience = namedtuple(
    "Experience", ["obs", "action", "reward", "next_obs", "done", "state", "next_state"]
)

# Trial tuple for trial-based replay buffer
Trial = namedtuple("Trial", ["obs_list", "action_list", "reward_list", "done_list"])


class ReplayBuffer:
    """Simple replay buffer for DQN."""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        """Add an experience to the buffer."""
        self.buffer.append(Experience(*args))

    def sample(self, batch_size):
        """Sample a batch of experiences."""
        experiences = random.sample(self.buffer, batch_size)

        # Stack into tensors
        obs = torch.stack([e.obs for e in experiences])
        actions = torch.tensor([e.action for e in experiences], dtype=torch.long)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32)
        next_obs = torch.stack([e.next_obs for e in experiences])
        dones = torch.tensor([e.done for e in experiences], dtype=torch.float32)
        states = torch.stack([e.state for e in experiences])
        next_states = torch.stack([e.next_state for e in experiences])

        return obs, actions, rewards, next_obs, dones, states, next_states

    def __len__(self):
        return len(self.buffer)


class TrialReplayBuffer:
    """
    Replay buffer for storing complete trial sequences.

    Used for training recurrent networks like MPNs where temporal coherence
    is important. Instead of storing individual transitions, this buffer stores
    complete trial sequences and samples entire trials for training.

    This solves the "stale state" problem where stored MPN states were computed
    by old network parameters. By replaying trials from scratch, we regenerate
    states using the current network.
    """

    def __init__(self, capacity=1000):
        """
        Args:
            capacity: Maximum number of trials to store (not timesteps)
        """
        self.buffer = deque(maxlen=capacity)

    def push_trial(self, obs_list, action_list, reward_list, done_list):
        """
        Add a complete trial to the buffer.

        Args:
            obs_list: List of observations for the trial (each is a tensor)
            action_list: List of actions (integers)
            reward_list: List of rewards (floats)
            done_list: List of done flags (booleans)
        """
        # Convert lists to tensors for efficient storage
        obs_tensor = torch.stack(obs_list)  # [T, obs_dim]
        action_tensor = torch.tensor(action_list, dtype=torch.long)  # [T]
        reward_tensor = torch.tensor(reward_list, dtype=torch.float32)  # [T]
        done_tensor = torch.tensor(done_list, dtype=torch.float32)  # [T]

        trial = Trial(
            obs_list=obs_tensor,
            action_list=action_tensor,
            reward_list=reward_tensor,
            done_list=done_tensor,
        )
        self.buffer.append(trial)

    def sample(self, batch_size):
        """
        Sample a batch of complete trials.

        Args:
            batch_size: Number of trials to sample

        Returns:
            List of Trial namedtuples
        """
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


class SequenceReplayBuffer:
    """
    DRQN-style replay buffer that stores transitions and samples fixed-length sequences.

    Key features (from DRQN paper):
    1. Stores individual transitions with episode tracking
    2. Samples fixed-length sequences (default L=10)
    3. Zero-pads at episode boundaries to prevent training on invalid sequences

    This matches the approach from "Deep Recurrent Q-Learning with Recurrent Neural Networks"
    (Chen, Ying, Laird), which showed this is critical for stable recurrent Q-learning.

    Example:
        Episode 1: [s0, s1, s2, s3] (episode_id=0)
        Episode 2: [s4, s5, s6, s7, s8] (episode_id=1)

        If we sample ending at s6 with L=4, we get:
        [zero, zero, s4, s5] - first 2 are zero-padded because they're from episode 0
    """

    def __init__(self, capacity=10000, sequence_length=10):
        """
        Args:
            capacity: Maximum number of transitions to store
            sequence_length: Fixed length L for sampled sequences (DRQN uses L=16)
        """
        self.buffer = deque(maxlen=capacity)
        self.sequence_length = sequence_length
        self._zero_transition_template = None

    def push(self, obs, action, reward, next_obs, done, episode_id):
        """
        Store a single transition with episode tracking.

        Args:
            obs: Observation tensor [obs_dim]
            action: Action (int)
            reward: Reward (float)
            next_obs: Next observation tensor [obs_dim]
            done: Done flag (bool)
            episode_id: Episode ID for tracking boundaries (int)
        """
        # Create zero transition template if not exists
        if self._zero_transition_template is None:
            self._zero_transition_template = {
                "obs": torch.zeros_like(obs),
                "action": 0,
                "reward": 0.0,
                "next_obs": torch.zeros_like(obs),
                "done": False,
                "episode_id": -1,
            }

        self.buffer.append(
            {
                "obs": obs,
                "action": action,
                "reward": reward,
                "next_obs": next_obs,
                "done": done,
                "episode_id": episode_id,
            }
        )

    def sample(self, batch_size):
        """
        Sample batch_size fixed-length sequences with zero-padding at episode boundaries.

        Following DRQN paper (Page 3):
        "We sample et ~ U(D), take the previous L states, {st-(L+1), ..., st},
        and then zero out states from previous games."

        Args:
            batch_size: Number of sequences to sample

        Returns:
            List of sequences, each a list of L transitions (dicts)
        """
        if len(self.buffer) < self.sequence_length:
            return []

        sequences = []

        for _ in range(batch_size):
            # Sample random end position (must have at least sequence_length elements before it)
            end_idx = random.randint(self.sequence_length - 1, len(self.buffer) - 1)

            # Get the episode ID at the end position
            current_episode = self.buffer[end_idx]["episode_id"]

            # Build sequence by looking back L steps
            sequence = []
            for i in range(self.sequence_length):
                idx = end_idx - (self.sequence_length - 1) + i
                transition = self.buffer[idx]

                # Zero out if from different episode (CRITICAL for DRQN)
                if transition["episode_id"] != current_episode:
                    sequence.append(self._get_zero_transition())
                else:
                    sequence.append(transition)

            sequences.append(sequence)

        return sequences

    def _get_zero_transition(self):
        """Return a zero-padded transition for episode boundaries."""
        return {
            "obs": self._zero_transition_template["obs"].clone(),
            "action": 0,
            "reward": 0.0,
            "next_obs": self._zero_transition_template["next_obs"].clone(),
            "done": False,
            "episode_id": -1,
        }

    def __len__(self):
        return len(self.buffer)


def compute_td_loss(dqn, target_dqn, batch, gamma=0.99):
    """
    Compute TD loss for DQN.

    Uses Double DQN update: Q_target = r + γ * Q_target(s', argmax_a Q_online(s', a))

    Args:
        dqn: Online DQN network
        target_dqn: Target DQN network
        batch: Tuple of (obs, actions, rewards, next_obs, dones, states, next_states)
        gamma: Discount factor

    Returns:
        loss: Smooth L1 loss between current Q-values and target Q-values
    """
    obs, actions, rewards, next_obs, dones, states, next_states = batch

    # Current Q-values: Q(s, a)
    current_q, _ = dqn(obs, states)
    current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Next Q-values from target network (Double DQN)
    with torch.no_grad():
        # Get best actions from online network
        next_q_online, _ = dqn(next_obs, next_states)
        next_actions = next_q_online.argmax(dim=1)

        # Evaluate those actions with target network
        next_q_target, _ = target_dqn(next_obs, next_states)
        next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

        # TD target: r + γ * Q_target(s', a*)
        target_q = rewards + gamma * next_q * (1 - dones)

    # Compute loss
    loss = F.smooth_l1_loss(current_q, target_q)

    return loss


def compute_td_loss_sequences(dqn, target_dqn, sequences, gamma=0.99, device="cpu"):
    """
    Compute TD loss over fixed-length sequences (parallel batch processing).

    Optimized version that processes all sequences in parallel for GPU efficiency.
    - Each sequence has fixed length L (e.g., 10-20 timesteps)
    - Full BPTT through the sequence (no chunking needed)
    - Zero-padded transitions are skipped in loss computation
    - All sequences processed in parallel with batch_size=len(sequences)

    Args:
        dqn: Online DQN network (MPNDQN)
        target_dqn: Target DQN network
        sequences: List of sequences from SequenceReplayBuffer.sample()
                  Each sequence is a list of L transitions (dicts)
        gamma: Discount factor
        device: Device to run on ('cpu' or 'cuda')

    Returns:
        loss: Average smooth L1 loss across all sequences and timesteps
    """
    if not sequences:
        return torch.tensor(0.0, device=device)

    batch_size = len(sequences)
    seq_len = len(sequences[0])

    # Stack all sequences into batched tensors [batch_size, seq_len, ...]
    obs_batch = torch.stack(
        [
            torch.stack([sequences[b][t]["obs"] for t in range(seq_len)])
            for b in range(batch_size)
        ]
    ).to(
        device
    )  # [batch_size, seq_len, obs_dim]

    next_obs_batch = torch.stack(
        [
            torch.stack([sequences[b][t]["next_obs"] for t in range(seq_len)])
            for b in range(batch_size)
        ]
    ).to(
        device
    )  # [batch_size, seq_len, obs_dim]

    actions_batch = torch.tensor(
        [
            [sequences[b][t]["action"] for t in range(seq_len)]
            for b in range(batch_size)
        ],
        dtype=torch.long,
        device=device,
    )  # [batch_size, seq_len]

    rewards_batch = torch.tensor(
        [
            [sequences[b][t]["reward"] for t in range(seq_len)]
            for b in range(batch_size)
        ],
        dtype=torch.float32,
        device=device,
    )  # [batch_size, seq_len]

    dones_batch = torch.tensor(
        [[sequences[b][t]["done"] for t in range(seq_len)] for b in range(batch_size)],
        dtype=torch.float32,
        device=device,
    )  # [batch_size, seq_len]

    episode_ids_batch = torch.tensor(
        [
            [sequences[b][t]["episode_id"] for t in range(seq_len)]
            for b in range(batch_size)
        ],
        dtype=torch.long,
        device=device,
    )  # [batch_size, seq_len]

    # Initialize states for all sequences in batch
    state = dqn.init_state(batch_size=batch_size, device=device)

    # Forward pass through all sequences in parallel
    q_values_list = []
    for t in range(seq_len):
        q_values, state = dqn(obs_batch[:, t, :], state)
        q_values_list.append(q_values)  # Each is [batch_size, action_dim]

    # Compute target Q-values (Double DQN)
    with torch.no_grad():
        target_state = target_dqn.init_state(batch_size=batch_size, device=device)
        target_q_list = []

        for t in range(seq_len):
            target_q, target_state = target_dqn(next_obs_batch[:, t, :], target_state)
            target_q_list.append(target_q)  # Each is [batch_size, action_dim]

    # Compute loss for all timesteps
    total_loss = 0.0
    total_timesteps = 0

    for t in range(seq_len):
        # Mask for valid (non-zero-padded) transitions
        valid_mask = episode_ids_batch[:, t] != -1  # [batch_size]

        if not valid_mask.any():
            continue

        # Current Q-values for actions taken
        current_q = (
            q_values_list[t].gather(1, actions_batch[:, t].unsqueeze(1)).squeeze(1)
        )  # [batch_size]

        # TD targets
        if t < seq_len - 1:
            # Not last timestep: use next Q-values
            with torch.no_grad():
                # Double DQN: select action with online, evaluate with target
                next_actions = q_values_list[t + 1].argmax(
                    dim=1, keepdim=True
                )  # [batch_size, 1]
                next_q = (
                    target_q_list[t + 1].gather(1, next_actions).squeeze(1)
                )  # [batch_size]

                # TD target: r + γ * Q(s', a') * (1 - done)
                target_q = rewards_batch[:, t] + gamma * next_q * (
                    1 - dones_batch[:, t]
                )
        else:
            # Last timestep: just reward
            target_q = rewards_batch[:, t]

        # Apply mask and compute loss only for valid transitions
        if valid_mask.any():
            loss = F.smooth_l1_loss(
                current_q[valid_mask], target_q[valid_mask], reduction="sum"
            )
            total_loss += loss
            total_timesteps += valid_mask.sum().item()

    # Average loss
    avg_loss = (
        total_loss / total_timesteps
        if total_timesteps > 0
        else torch.tensor(0.0, device=device)
    )

    return avg_loss


def compute_td_loss_trial(
    dqn, target_dqn, trial_batch, gamma=0.99, device="cpu", bptt_chunk_size=None
):
    """
    Compute TD loss for DQN over complete trial sequences with optional Truncated BPTT.

    Key features:
    - Replays each trial from scratch with fresh MPN state
    - Uses Truncated BPTT to manage memory for long sequences
    - Properly handles MPN's Hebbian plasticity during forward pass
    - Uses Double DQN for stable target values

    MPN Architecture Notes:
    - W (weight matrix) is learned via backpropagation
    - M (modulation matrix) is updated via Hebbian rule during forward pass
    - Truncated BPTT breaks gradient flow through M updates across chunks
    - This allows efficient training on long sequences (100-800 timesteps)

    Args:
        dqn: Online DQN network (MPNDQN)
        target_dqn: Target DQN network
        trial_batch: List of Trial namedtuples (from TrialReplayBuffer.sample())
        gamma: Discount factor
        device: Device to run on ('cpu' or 'cuda')
        bptt_chunk_size: Chunk size for Truncated BPTT. If None, uses full BPTT
                        through entire trial. Recommended: 20-50 for long sequences.

    Returns:
        loss: Average smooth L1 loss across all trials and timesteps

    Implementation Details:
    - Each trial is processed sequentially (batch_size=1 through time)
    - If bptt_chunk_size is set, gradients are truncated between chunks
    - State (M matrix) continues across chunks but gradients are detached
    - This allows MPN to maintain memory while keeping gradients manageable
    """
    total_loss = 0.0
    total_timesteps = 0

    for trial in trial_batch:
        # Get trial data
        obs_seq = trial.obs_list.to(device)  # [T, obs_dim]
        actions = trial.action_list.to(device)  # [T]
        rewards = trial.reward_list.to(device)  # [T]
        dones = trial.done_list.to(device)  # [T]

        trial_length = obs_seq.shape[0]

        # Initialize MPN state (fresh state from current network!)
        state = dqn.init_state(batch_size=1, device=device)

        # Determine chunking strategy
        if bptt_chunk_size is None or bptt_chunk_size >= trial_length:
            # Full BPTT through entire trial
            chunks = [(0, trial_length)]
        else:
            # Truncated BPTT: break into chunks
            chunks = []
            for chunk_start in range(0, trial_length, bptt_chunk_size):
                chunk_end = min(chunk_start + bptt_chunk_size, trial_length)
                chunks.append((chunk_start, chunk_end))

        # Process each chunk
        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
            chunk_length = chunk_end - chunk_start

            # Forward pass through chunk
            q_values_list = []
            states_list = [state]  # Store initial state for this chunk

            current_state = state
            for t in range(chunk_start, chunk_end):
                obs_t = obs_seq[t].unsqueeze(0)  # [1, obs_dim]
                q_values, next_state = dqn(obs_t, current_state)

                q_values_list.append(q_values)
                states_list.append(next_state)

                current_state = next_state

            # Compute TD targets for this chunk
            with torch.no_grad():
                next_q_values_target = []

                for i, t in enumerate(range(chunk_start, chunk_end)):
                    if t < trial_length - 1:
                        # Need to check if next timestep is in current chunk or next chunk
                        if t + 1 < chunk_end:
                            # Next timestep in current chunk - use already computed Q-values
                            next_q_online = q_values_list[i + 1]
                            next_state_for_target = states_list[i]
                        else:
                            # Next timestep in next chunk - need to compute
                            next_obs_t = obs_seq[t + 1].unsqueeze(0)
                            next_q_online, _ = dqn(next_obs_t, current_state)
                            next_state_for_target = current_state

                        best_next_action = next_q_online.argmax(dim=1, keepdim=True)

                        # Evaluate with target network
                        next_obs_t = obs_seq[t + 1].unsqueeze(0)
                        q_target, _ = target_dqn(next_obs_t, next_state_for_target)
                        next_q = q_target.gather(1, best_next_action).squeeze()
                    else:
                        # Terminal state
                        next_q = torch.tensor(0.0, device=device)

                    next_q_values_target.append(next_q)

            # Compute loss for this chunk
            chunk_loss = 0.0
            for i, t in enumerate(range(chunk_start, chunk_end)):
                current_q = (
                    q_values_list[i]
                    .gather(1, actions[t].unsqueeze(0).unsqueeze(0))
                    .squeeze()
                )
                target_q = rewards[t] + gamma * next_q_values_target[i] * (1 - dones[t])

                loss = F.smooth_l1_loss(current_q, target_q)
                chunk_loss += loss
                total_timesteps += 1

            # Accumulate chunk loss
            total_loss += chunk_loss

            # Truncate gradients between chunks (key for Truncated BPTT)
            # The state continues but gradients don't flow backward past this point
            if chunk_idx < len(chunks) - 1:  # Not the last chunk
                state = current_state.detach()
            else:
                state = current_state

    # Average loss over all timesteps in batch
    avg_loss = (
        total_loss / total_timesteps
        if total_timesteps > 0
        else torch.tensor(0.0, device=device)
    )

    return avg_loss


# Word lists for random experiment names
ADJECTIVES = [
    "swift",
    "brave",
    "bright",
    "calm",
    "clever",
    "bold",
    "eager",
    "fair",
    "gentle",
    "happy",
    "keen",
    "lively",
    "merry",
    "noble",
    "polite",
    "proud",
    "quiet",
    "rapid",
    "sincere",
    "tender",
    "vivid",
    "wise",
    "zealous",
    "agile",
    "cosmic",
    "digital",
    "electric",
    "frozen",
    "golden",
    "lunar",
    "mystic",
    "neural",
    "quantum",
    "radiant",
    "silver",
    "stellar",
    "turbo",
    "ultra",
]

NOUNS = [
    "tiger",
    "eagle",
    "falcon",
    "dragon",
    "phoenix",
    "wolf",
    "bear",
    "lion",
    "hawk",
    "raven",
    "shark",
    "panther",
    "cobra",
    "viper",
    "mantis",
    "spider",
    "scorpion",
    "leopard",
    "cheetah",
    "jaguar",
    "orca",
    "dolphin",
    "owl",
    "condor",
    "lynx",
    "puma",
    "fox",
    "badger",
    "otter",
    "weasel",
    "mink",
    "neuron",
    "synapse",
    "cortex",
    "network",
    "circuit",
    "matrix",
    "tensor",
]


def generate_experiment_name() -> str:
    """Generate a random experiment name like 'brave-tiger' or 'swift-eagle'."""
    adj = random.choice(ADJECTIVES)
    noun = random.choice(NOUNS)
    return f"{adj}-{noun}"


SCHEMA_VERSION = 1
EXPERIMENTS_DB = Path("experiments/experiments.sqlite")


def _get_db() -> sqlite3.Connection:
    """Open (or create) the SQLite experiments DB with WAL mode.

    Note: the DB is an index only — JSON files in each experiment directory
    are the source of truth.  DB writes are best-effort: if the filesystem
    doesn't support SQLite locking (e.g. NFS), writes fail silently and the
    JSON files remain intact.  Run `python query_experiments.py backfill` on
    the head node to rebuild the index from JSON at any time.
    """
    EXPERIMENTS_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(EXPERIMENTS_DB), timeout=60)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=60000")  # 60 s auto-retry on lock
    con.execute("PRAGMA synchronous=NORMAL")  # safe + faster than FULL
    con.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_name TEXT PRIMARY KEY,
            schema_version  INTEGER NOT NULL,
            created_at      TEXT NOT NULL,
            completed       INTEGER NOT NULL DEFAULT 0,
            config          TEXT NOT NULL
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS training_history (
            experiment_name TEXT NOT NULL,
            schema_version  INTEGER NOT NULL,
            frame           INTEGER NOT NULL,
            episode         INTEGER,
            reward          REAL,
            length          INTEGER,
            loss            REAL,
            epsilon         REAL,
            oracle_reward   REAL,
            pct_oracle      REAL,
            PRIMARY KEY (experiment_name, frame)
        )
    """)
    con.commit()
    return con


def _try_db_write(fn):
    """Call fn() which performs a DB write; silently ignore any DB errors.

    Training must not crash due to DB issues — JSON files are the real record.
    """
    try:
        fn()
    except Exception:
        pass


class ExperimentManager:
    """
    Manages experiment directory structure and file I/O.

    Directory structure:
        experiments/{experiment_name}/
        ├── config.json
        ├── training_history.json
        ├── checkpoints/
        │   ├── best_model.pt
        │   ├── checkpoint_100.pt
        │   └── final_model.pt
        ├── videos/
        │   └── episode_*.gif
        └── plots/
            └── training_curves.png
    """

    def __init__(
        self, experiment_name: Optional[str] = None, base_dir: str = "experiments"
    ):
        """
        Args:
            experiment_name: Name of experiment (generates random if None)
            base_dir: Base directory for all experiments
        """
        if experiment_name is None:
            experiment_name = generate_experiment_name()

        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.exp_dir = self.base_dir / experiment_name

        # Create directory structure
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.video_dir = self.exp_dir / "videos"
        self.plot_dir = self.exp_dir / "plots"

        for dir_path in [self.checkpoint_dir, self.video_dir, self.plot_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.config_path = self.exp_dir / "config.json"
        self.metrics_path = self.exp_dir / "metrics.jsonl"

    def save_config(self, config: Dict[str, Any]):
        """Save experiment configuration."""
        created_at = datetime.now().isoformat()
        config = {**config, "schema_version": SCHEMA_VERSION, "created_at": created_at}
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

        def _write():
            con = _get_db()
            con.execute(
                """
                INSERT OR REPLACE INTO experiments (experiment_name, schema_version, created_at, config)
                VALUES (?, ?, ?, ?)
            """,
                (self.experiment_name, SCHEMA_VERSION, created_at, json.dumps(config)),
            )
            con.commit()
            con.close()

        _try_db_write(_write)
        print(f"Saved config to {self.config_path}")

    def load_config(self) -> Dict[str, Any]:
        """Load experiment configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        with open(self.config_path, "r") as f:
            return json.load(f)

    def save_model(
        self,
        model,
        optimizer: Optional[torch.optim.Optimizer] = None,
        checkpoint_name: str = "model.pt",
        metadata: Optional[Dict] = None,
    ):
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
        model,
        checkpoint_name: str = "best_model.pt",
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
    ) -> Dict:
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
        return checkpoint.get("metadata", {})

    def append_training_history(
        self,
        frames: int,
        reward: float,
        length: int,
        loss: float,
        epsilon: float,
        oracle_reward: float = None,
        pct_oracle: float = None,
        episode: int = None,
    ):
        """Append a single eval step to metrics.jsonl and the DB."""
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

        def _write():
            con = _get_db()
            con.execute(
                """
                INSERT OR REPLACE INTO training_history
                    (experiment_name, schema_version, frame, episode, reward, length, loss, epsilon,
                     oracle_reward, pct_oracle)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    self.experiment_name,
                    SCHEMA_VERSION,
                    frames,
                    episode,
                    reward,
                    length,
                    loss,
                    epsilon,
                    oracle_reward,
                    pct_oracle,
                ),
            )
            con.commit()
            con.close()

        _try_db_write(_write)

    def mark_completed(self):
        """Mark this experiment as completed in the DB."""

        def _write():
            con = _get_db()
            con.execute(
                "UPDATE experiments SET completed = 1 WHERE experiment_name = ?",
                (self.experiment_name,),
            )
            con.commit()
            con.close()

        _try_db_write(_write)

    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best model checkpoint."""
        best_path = self.checkpoint_dir / "best_model.pt"
        return str(best_path) if best_path.exists() else None

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to most recent checkpoint."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            return None
        # Sort by episode number
        checkpoints.sort(key=lambda p: int(p.stem.split("_")[1]))
        return str(checkpoints[-1])

    def cleanup_checkpoints(self, max_checkpoints: int = 4):
        """
        Keep only the most recent periodic checkpoints, deleting older ones.
        best_model.pt and final_model.pt are never deleted.

        Args:
            max_checkpoints: Maximum number of checkpoint_*.pt files to keep
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if len(checkpoints) <= max_checkpoints:
            return
        checkpoints.sort(key=lambda p: int(p.stem.split("_")[1]))
        for old_ckpt in checkpoints[:-max_checkpoints]:
            old_ckpt.unlink()
            print(f"Removed old checkpoint: {old_ckpt.name}")

    def __repr__(self):
        return f"ExperimentManager('{self.experiment_name}', dir='{self.exp_dir}')"


def save_checkpoint(
    experiment_manager: ExperimentManager,
    model,
    optimizer,
    episode: int,
    avg_reward: float,
    is_best: bool = False,
    is_final: bool = False,
):
    """
    Convenience function to save a checkpoint.

    Args:
        experiment_manager: ExperimentManager instance
        model: Model to save
        optimizer: Optimizer to save
        episode: Current episode number
        avg_reward: Average reward (for tracking best)
        is_best: Whether this is the best model so far
        is_final: Whether this is the final checkpoint
    """
    metadata = {
        "episode": episode,
        "avg_reward": avg_reward,
        "timestamp": datetime.now().isoformat(),
    }

    # Save periodic checkpoint
    checkpoint_name = f"checkpoint_{episode}.pt"
    experiment_manager.save_model(model, optimizer, checkpoint_name, metadata)

    # Save as best if applicable
    if is_best:
        experiment_manager.save_model(model, optimizer, "best_model.pt", metadata)
        print(f"New best model! Avg reward: {avg_reward:.2f}")

    # Save as final if applicable
    if is_final:
        experiment_manager.save_model(model, optimizer, "final_model.pt", metadata)


def load_checkpoint_for_eval(
    experiment_manager: ExperimentManager,
    model,
    checkpoint_name: str = "best_model.pt",
    device: str = "cpu",
) -> Dict:
    """
    Load checkpoint for evaluation (no optimizer).

    Args:
        experiment_manager: ExperimentManager instance
        model: Model to load weights into
        checkpoint_name: Which checkpoint to load
        device: Device to load onto

    Returns:
        metadata: Checkpoint metadata
    """
    return experiment_manager.load_model(
        model, checkpoint_name, optimizer=None, device=device
    )


def load_checkpoint_for_resume(
    experiment_manager: ExperimentManager, model, optimizer, device: str = "cpu"
) -> Dict:
    """
    Load latest checkpoint to resume training.

    Args:
        experiment_manager: ExperimentManager instance
        model: Model to load weights into
        optimizer: Optimizer to load state into
        device: Device to load onto

    Returns:
        metadata: Checkpoint metadata with episode number
    """
    # Try to get latest checkpoint first
    latest_checkpoint = experiment_manager.get_latest_checkpoint()

    if latest_checkpoint:
        checkpoint_name = Path(latest_checkpoint).name
    else:
        # Fall back to best or final model
        if (experiment_manager.checkpoint_dir / "best_model.pt").exists():
            checkpoint_name = "best_model.pt"
        elif (experiment_manager.checkpoint_dir / "final_model.pt").exists():
            checkpoint_name = "final_model.pt"
        else:
            raise FileNotFoundError("No checkpoints found to resume from")

    metadata = experiment_manager.load_model(model, checkpoint_name, optimizer, device)
    print(f"Resuming from episode {metadata.get('episode', 0)}")
    return metadata


if __name__ == "__main__":
    print("Testing ExperimentManager...")

    # Test random name generation
    print("\nRandom experiment names:")
    for _ in range(5):
        print(f"  - {generate_experiment_name()}")

    # Test experiment manager
    print("\nCreating experiment...")
    exp = ExperimentManager("test-experiment")
    print(f"Created: {exp}")
    print(f"Experiment dir: {exp.exp_dir}")

    # Test config save/load
    print("\nTesting config save/load...")
    config = {
        "env_name": "CartPole-v1",
        "hidden_dim": 64,
        "eta": 0.05,
        "lambda_decay": 0.9,
        "num_episodes": 500,
    }
    exp.save_config(config)
    loaded_config = exp.load_config()
    print(f"Loaded config: {loaded_config}")

    print("\nExperimentManager test completed!")
    print(f"Check the '{exp.exp_dir}' directory to see generated files")
