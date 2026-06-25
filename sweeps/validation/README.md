# Plot validation sweeps

Small, **seeded** (reproducible) sweeps whose only purpose is to generate data that
exercises every plot script and its branches — not real research runs. Training is
deliberately tiny; the figures will look like noise. We're checking that each plot
runs end-to-end and renders, not that anything learned well.

## Sweeps

| file | sweep_name | runs | covers |
|---|---|---|---|
| `coverage_seed_42.yaml` | `coverage_seed_42` | 144 | all plots; single-version paths |
| `coverage_seed_43.yaml` | `coverage_seed_43` | 144 | second version → bar error bars + curves mean±std band |

`coverage_seed_43` differs from `_42` only by `seed`, so the two are genuine
reproducible replicates (the variance the bar/curves plots show is seed variance).

## Run (on the cluster)

```bash
uv run python main_a2c.py start-sweep sweeps/validation/coverage_seed_42.yaml
uv run python main_a2c.py start-sweep sweeps/validation/coverage_seed_43.yaml
```

(`start-sweep` refuses a dirty tree; commit first or pass `--allow-dirty`.)

## Render every plot (from the repo root, after the runs finish)

```bash
uv run python plots/plot_custom_sweep_curves.py coverage_seed_42,coverage_seed_43 curves.png
uv run python plots/plot_custom_sweep_bar.py coverage_seed_42,coverage_seed_43 bar.png
uv run python plots/plot_eval_table.py eval_table.png coverage_seed_42
uv run python plots/plot_id_mpn_curve.py coverage_seed_42 id_mpn.png 7 IntervalDiscrimination-v0
uv run python plots/plot_layer_heatmap.py coverage_seed_42 heatmap.png mpn
uv run python plots/plot_trial_appendix.py PerceptualDecisionMaking-v0 trial.png 0 coverage_seed_42
uv run python plots/plot_episode_rewards.py     # no args; uses old run-name convention — may show no data
# visualize needs one mpn run; coverage_seed_42-0012 is the first (GoNogo, 1 layer, 64-dim)
uv run python plots/visualize_mpn_m_matrix.py --experiment coverage_seed_42-0012 --num-episodes 1 --output mmatrix.gif
```

## Smoke-test the eval and render commands

Confirm the `eval` and `render` subcommands load a trained run and work end-to-end:

```bash
uv run python main_a2c.py eval --experiment-name coverage_seed_42-0012 --experiments-dir results/coverage_seed_42/experiments
uv run python main_a2c.py render --experiment-name coverage_seed_42-0012 --experiments-dir results/coverage_seed_42/experiments --output render.png
```

## What to check in the outputs

The commands running without error is only half of it — open each output and confirm the
*structure* is right. Training is tiny, so curves are noisy and many runs don't learn;
you're checking that each plot renders its pieces, not that anything learned well.

| output | what a correct render shows |
|---|---|
| `curves.png` | grid of envs (rows) × model types (cols); each populated cell has a smoothed reward curve with a shaded ±std band (the band = the two seed versions aggregated); empty cells say "no data" |
| `bar.png` | one bar group per env, four bars (models) each, with ±std error bars across the two versions |
| `eval_table.png` | table of envs × models showing `mean ± std`; best-in-row cells green, failed (<0.1) red, missing "—" |
| `id_mpn.png` / `id_mpn_2trials.png` | MPN (green) vs MPN-frozen (red) curve on IntervalDiscrimination, beside 1 (and 2) sample-trial panels |
| `heatmap.png` | one heatmap per env (num_layers × hidden_dim) of best MPN reward; coloured cells, "—" for missing |
| `trial.png` | three trial columns for the best MPN on PerceptualDecisionMaking, with channel rows and a shaded decision period |
| `mmatrix.gif` | an animation of the MPN M-matrix evolving across the episode's frames |
| `render.png` | three stacked panels — observations / actions / rewards — for one episode |
| `eval` (stdout) | per-episode rewards then a `Mean reward: X ± Y` summary |
| `plot_episode_rewards.png` | **expected to be empty** with sweep data — known limitation: it filters on the old `a2c_run*` naming |
