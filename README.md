# Multi-Plasticity Networks for Reinforcement Learning

Trains MPN (Multi-Plasticity Network), LSTM, and RNN agents on cognitive neuroscience tasks using A2C. MPNs combine long-term weight learning via backpropagation with short-term Hebbian plasticity as a recurrent memory layer.

## Architecture

The actor-critic network uses a recurrent core selected by `--model-type`:

```
Observation → core (MPN / LSTM / RNN)
                → Linear(hidden_dim, 64) + ReLU
                → actor:  Linear(64, action_dim) → Softmax
                → critic: Linear(64, 1)
```

**MPN update rules:**

Hebbian plasticity:
```
M_t = λ·M_{t-1} + η·h_t·x_t^T
```

Multiplicative modulation:
```
h = activation(b + W·(M + 1)·x)
```

- `M`: synaptic modulation matrix (recurrent state)
- `W`: long-term weights (learned via backprop)
- `η` (eta): Hebbian learning rate
- `λ` (lambda): decay factor for M

## Installation

```bash
git clone <repository-url>
cd mpn-rl
./setup.sh
source .venv/bin/activate
```

## Quick Start

### Train on NeuroGym environments

```bash
# LSTM baseline on GoNogo
python main_a2c.py train-neurogym --env-name GoNogo-v0 --model-type lstm

# MPN agent on DMS
python main_a2c.py train-neurogym --env-name DelayMatchSample-v0 --model-type mpn

# Multi-layer MPN with custom hyperparameters
python main_a2c.py train-neurogym \
    --env-name GoNogo-v0 \
    --model-type mpn \
    --hidden-dim 128 \
    --num-layers 2 \
    --learning-rate 1e-4 \
    --total-frames 500000
```

### Train on custom environments

```bash
# IntervalTiming1D, IntervalDisc1D, DelayMatchSample
python main_a2c.py train-custom --env-name IntervalTiming1D --model-type mpn
python main_a2c.py train-custom --env-name DelayMatchSample --num-layers 2
```

### Evaluate and render

```bash
python main_a2c.py eval --experiment-name <name> --num-eval-episodes 20
python main_a2c.py render --experiment-name <name> --output render.png
```

### Query results

```bash
# Summary of all experiments
python query_experiments.py list

# Best runs for a given environment
python query_experiments.py best --env GoNogo-v0

# Compare model types
python query_experiments.py compare --model-type mpn
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-type` | `lstm` | `lstm`, `rnn`, `mpn`, `mpn-frozen` |
| `--hidden-dim` | `128` | Hidden layer size |
| `--num-layers` | `1` | Number of recurrent layers |
| `--total-frames` | `500000` | Training budget (frames) |
| `--learning-rate` | `1e-4` | Adam learning rate |
| `--gamma` | `0.98` | Discount factor |
| `--entropy-coef` | `0.01` | Entropy regularization |
| `--eta-init` | `0.01` | MPN Hebbian learning rate |
| `--lambda-init` | `0.99` | MPN decay factor |

## Key Files

- **`main_a2c.py`**: CLI entry point — train, eval, render
- **`src/mpn_rl/nn/mpn.py`**: MPN layer with Hebbian plasticity
- **`custom_envs.py`**: Custom RL environments (IntervalTiming1D, IntervalDisc1D, DelayMatchSample)
- **`src/mpn_rl/temporal_order_env.py`**: TemporalOrder environment family
- **`src/mpn_rl/model_utils.py`**: ExperimentManager, checkpointing, SQLite metrics
- **`src/mpn_rl/oracle_agents.py`**: Oracle baselines for all supported environments
- **`query_experiments.py`**: Live DuckDB queries over experiment output files
- **`plots/`**: Plotting scripts for rewards, sweeps, trial lengths

## Experiment Directory Structure

```
experiments/{experiment-name}/
├── config.json           # Hyperparameters
├── metrics.jsonl         # One JSON line per eval step (live-queryable)
├── checkpoints/
│   ├── best_model.pt
│   └── final_model.pt
└── render.png            # Episode render (if generated)
```

## Running Jobs on HTCondor

The `condor/` directory contains submit files (`.job`), shell scripts (`.sh`), and argument files (`_args.txt`) for running sweeps on the cluster.

### Submit a sweep

```bash
# Submit a job
condor_submit condor/jobs/train_neurogym.job

# Check status
condor_q

# Monitor output
tail -f condor/logs/train_0.out
```

### Create a new sweep

1. Add an args file to `condor/args/` (one set of arguments per line)
2. Add a shell script to `condor/scripts/` (activate `.venv`, call `main_a2c.py`)
3. Add a `.job` file to `condor/jobs/` pointing at the script and args file

### Resource guidelines

- CPU training: `request_cpus = 1`, `request_memory = 4GB`
- GPU training: `request_cpus = 2`, `request_gpus = 1`, `request_memory = 8GB`

### Managing jobs

```bash
condor_q              # job status
condor_q -hold        # check held jobs
condor_release <id>   # release a held job
condor_rm <username>  # remove all your jobs
```
