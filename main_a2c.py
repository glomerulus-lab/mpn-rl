"""
Main CLI for MPN A2C training.

Commands:
    train-neurogym - Train MPN/LSTM/RNN agent on NeuroGym environments using A2C
    eval           - Evaluate a trained agent
    render         - Render episode(s) to a static plot

Examples:
    python main_a2c.py train-neurogym --env-name GoNogo-v0
    python main_a2c.py train-neurogym --env-name GoNogo-v0 --env-config configs/gonogo.json
    python main_a2c.py eval --experiment-name my-agent --num-eval-episodes 10
    python main_a2c.py render --experiment-name my-agent --output render.png
"""

from typing import Annotated, Union

import tyro

from mpn_rl.commands.eval import EvalConfig, evaluate
from mpn_rl.commands.render import RenderConfig, render_to_plot
from mpn_rl.commands.train import TrainConfig, train_neurogym

Command = Union[
    Annotated[TrainConfig, tyro.conf.subcommand("train-neurogym")],
    Annotated[EvalConfig, tyro.conf.subcommand("eval")],
    Annotated[RenderConfig, tyro.conf.subcommand("render")],
]


def main():
    cfg = tyro.cli(Command, description="MPN A2C training, evaluation, and rendering.")
    if isinstance(cfg, TrainConfig):
        train_neurogym(cfg)
    elif isinstance(cfg, EvalConfig):
        evaluate(cfg)
    else:
        render_to_plot(cfg)


if __name__ == "__main__":
    main()
