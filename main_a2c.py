"""
Main CLI for MPN A2C training.

Commands:
    train-neurogym - Train MPN/LSTM/RNN agent on NeuroGym environments using A2C
    eval           - Evaluate a trained agent
    render         - Render episode(s) to a static plot

Examples:
    python main_a2c.py start-sweep sweep.yaml
    python main_a2c.py train-neurogym --env-name GoNogo-v0
    python main_a2c.py train-neurogym --env-name GoNogo-v0 --env-config configs/gonogo.json
    python main_a2c.py eval --experiment-name my-agent --num-eval-episodes 10
    python main_a2c.py render --experiment-name my-agent --output render.png
"""

import sys
from typing import Annotated, Union

import tyro

from mpn_rl.commands.eval import EvalConfig, evaluate
from mpn_rl.commands.render import RenderConfig, render_to_plot
from mpn_rl.commands.start_sweep import StartSweepCommand, start_sweep
from mpn_rl.commands.train import TrainCommand, resolve_train_config, train_neurogym


def main():
    Command = Union[
        Annotated[StartSweepCommand, tyro.conf.subcommand("start-sweep")],
        Annotated[TrainCommand, tyro.conf.subcommand("train-neurogym")],
        Annotated[EvalConfig, tyro.conf.subcommand("eval")],
        Annotated[RenderConfig, tyro.conf.subcommand("render")],
    ]
    cfg = tyro.cli(
        Command,
        args=sys.argv[1:],
        description="MPN A2C training, evaluation, and rendering.",
    )
    if isinstance(cfg, StartSweepCommand):
        start_sweep(cfg)
    elif isinstance(cfg, TrainCommand):
        # sys.argv[1] is the "train-neurogym" subcommand token; resolve_train_config
        # parses bare TrainCommand, so pass only the flags after it.
        train_neurogym(resolve_train_config(sys.argv[2:]))
    elif isinstance(cfg, EvalConfig):
        evaluate(cfg)
    else:
        render_to_plot(cfg)


if __name__ == "__main__":
    main()
