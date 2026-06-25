"""
Each experiment writes its own metrics.jsonl (one JSON line per eval step)
and config.json.  This script uses an in-memory DuckDB to glob all those
files on demand, so queries always reflect the current state of running
experiments with no locking or concurrency issues.

Usage:
    # Live summary — works while jobs are running
    python query_experiments.py list
    python query_experiments.py best --env GoNogo-v0
    python query_experiments.py compare --model-type mpn
    python query_experiments.py sql "SELECT ..."
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from mpn_rl.experiment import find_experiment_files

# ---------------------------------------------------------------------------
# Live DuckDB connection (reads files directly, always up-to-date)
# ---------------------------------------------------------------------------


def _sql_list(files: list[Path]) -> str:
    return "[" + ", ".join("'" + str(f) + "'" for f in files) + "]"


def _live_con(experiments_dir: Path | None) -> duckdb.DuckDBPyConnection:
    """Return an in-memory DuckDB with views over live experiment files.

    Scans config.json/metrics.jsonl for every experiment — ad-hoc experiments
    under experiments/ and sweep experiments under results/<name>/experiments/ —
    or a single flat experiments_dir when one is given.
    """
    con = duckdb.connect()

    metrics_files = find_experiment_files("metrics.jsonl", experiments_dir)
    config_files = find_experiment_files("config.json", experiments_dir)

    if metrics_files:
        con.execute(f"""
            CREATE VIEW training_history AS
            SELECT experiment_name, frame, reward, length, loss, epsilon,
                   oracle_reward, pct_oracle
            FROM read_ndjson(
                {_sql_list(metrics_files)},
                columns = {{
                    experiment_name: 'VARCHAR',
                    frame:           'INTEGER',
                    reward:          'DOUBLE',
                    length:          'INTEGER',
                    loss:            'DOUBLE',
                    epsilon:         'DOUBLE',
                    oracle_reward:   'DOUBLE',
                    pct_oracle:      'DOUBLE'
                }},
                ignore_errors = true
            )
        """)
    else:
        con.execute("""
            CREATE VIEW training_history AS
            SELECT NULL::VARCHAR as experiment_name, NULL::INTEGER as frame,
                   NULL::DOUBLE  as reward,          NULL::INTEGER as length,
                   NULL::DOUBLE  as loss,            NULL::DOUBLE  as epsilon,
                   NULL::DOUBLE  as oracle_reward,   NULL::DOUBLE  as pct_oracle
            WHERE false
        """)

    if config_files:
        con.execute(f"""
            CREATE VIEW experiments AS
            SELECT *
            FROM read_json_auto(
                {_sql_list(config_files)},
                ignore_errors = true
            )
        """)
    else:
        con.execute("""
            CREATE VIEW experiments AS
            SELECT NULL::VARCHAR as experiment_name WHERE false
        """)

    return con


def _query(sql: str, experiments_dir: Path | None) -> pd.DataFrame:
    con = _live_con(experiments_dir)
    return con.execute(sql).fetchdf()


# ---------------------------------------------------------------------------
# Query commands (all use live DuckDB file scanning)
# ---------------------------------------------------------------------------


def cmd_list(args):
    df = _query(
        """
        SELECT
            e.experiment_name,
            COALESCE(e.algorithm, 'dqn') AS algorithm,
            e.model_type,
            e.env_name,
            e.learning_rate  AS lr,
            MAX(h.frame)     AS latest_frame,
            ROUND(MAX(h.reward), 2) AS best_reward
        FROM experiments e
        LEFT JOIN training_history h USING (experiment_name)
        GROUP BY e.experiment_name, e.algorithm, e.model_type, e.env_name, e.learning_rate
        ORDER BY MAX(h.frame) DESC NULLS LAST
    """,
        args.experiments_dir,
    )
    print(df.to_string(index=False))


def cmd_best(args):
    filters = []
    if args.env:
        filters.append(f"e.env_name = '{args.env}'")
    if args.algorithm:
        filters.append(f"COALESCE(e.algorithm, 'dqn') = '{args.algorithm}'")
    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    df = _query(
        f"""
        SELECT
            e.experiment_name,
            COALESCE(e.algorithm, 'dqn') AS algorithm,
            e.model_type,
            e.env_name,
            e.learning_rate                AS lr,
            ROUND(MAX(h.reward), 4)        AS best_reward,
            ROUND(AVG(h.reward), 4)        AS avg_reward
        FROM experiments e
        JOIN training_history h USING (experiment_name)
        {where}
        GROUP BY e.experiment_name, e.algorithm, e.model_type, e.env_name, e.learning_rate
        ORDER BY best_reward DESC
        LIMIT {args.limit}
    """,
        args.experiments_dir,
    )
    print(df.to_string(index=False))


def cmd_compare(args):
    filters = []
    if args.env:
        filters.append(f"e.env_name = '{args.env}'")
    if args.model_type:
        filters.append(f"e.model_type = '{args.model_type}'")
    if args.algorithm:
        filters.append(f"COALESCE(e.algorithm, 'dqn') = '{args.algorithm}'")
    where = ("WHERE " + " AND ".join(filters)) if filters else ""

    df = _query(
        f"""
        SELECT
            COALESCE(e.algorithm, 'dqn')              AS algorithm,
            e.model_type,
            e.env_name,
            e.learning_rate                           AS lr,
            e.hidden_dim,
            COUNT(DISTINCT e.experiment_name)         AS num_experiments,
            ROUND(AVG(h.reward), 4)                   AS avg_reward,
            ROUND(STDDEV(h.reward), 4)                AS std_reward
        FROM experiments e
        JOIN training_history h USING (experiment_name)
        {where}
        GROUP BY e.algorithm, e.model_type, e.env_name, e.learning_rate, e.hidden_dim
        ORDER BY e.env_name, avg_reward DESC
    """,
        args.experiments_dir,
    )
    print(df.to_string(index=False))


def cmd_today(args):
    today = datetime.now().strftime("%Y-%m-%d")
    rows = []

    for config_path in sorted(
        find_experiment_files("config.json", args.experiments_dir)
    ):
        with open(config_path) as f:
            config = json.load(f)

        created_at = config.get("created_at", "")
        if not created_at.startswith(today):
            continue

        exp_name = config.get("experiment_name", config_path.parent.name)
        algorithm = config.get("algorithm", "dqn")
        model_type = config.get("model_type", "?")
        utd = config.get("utd", "?")
        lr = config.get("learning_rate", "?")
        hidden_dim = config.get("hidden_dim", "?")
        num_layers = config.get("num_layers", "?")

        metrics_path = config_path.parent / "metrics.jsonl"
        if not metrics_path.exists():
            rows.append(
                dict(
                    experiment_name=exp_name,
                    algorithm=algorithm,
                    model_type=model_type,
                    utd=utd,
                    lr=lr,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    latest_frame=None,
                    last3_avg_reward=None,
                    last10_avg_reward=None,
                    total_avg_reward=None,
                    best_reward=None,
                )
            )
            continue

        rewards, frames = [], []
        with open(metrics_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    rewards.append(entry["reward"])
                    frames.append(entry["frame"])
                except (json.JSONDecodeError, KeyError):
                    continue

        if not rewards:
            rows.append(
                dict(
                    experiment_name=exp_name,
                    algorithm=algorithm,
                    model_type=model_type,
                    utd=utd,
                    lr=lr,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    latest_frame=None,
                    last3_avg_reward=None,
                    last10_avg_reward=None,
                    total_avg_reward=None,
                    best_reward=None,
                )
            )
            continue

        last3 = rewards[-3:]
        last10 = rewards[-10:]
        rows.append(
            dict(
                experiment_name=exp_name,
                algorithm=algorithm,
                model_type=model_type,
                utd=utd,
                lr=lr,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                latest_frame=frames[-1],
                last3_avg_reward=round(sum(last3) / len(last3), 4),
                last10_avg_reward=round(sum(last10) / len(last10), 4),
                total_avg_reward=round(sum(rewards) / len(rewards), 4),
                best_reward=round(max(rewards), 4),
            )
        )

    if not rows:
        print(f"No experiments found for today ({today}).")
        return

    df = pd.DataFrame(rows).sort_values(["algorithm", "model_type", "experiment_name"])
    print(f"Experiments from {today}:\n")
    print(df.to_string(index=False))


def cmd_sql(args):
    df = _query(args.query, args.experiments_dir)
    print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Query MPN-RL experiments (live file scanning)"
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=None,
        help="Scan a single experiments dir; default scans "
        "experiments/ and results/*/experiments/",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "list", help="List all experiments with latest frame and best reward"
    )

    p_best = sub.add_parser("best", help="Best reward per experiment")
    p_best.add_argument("--env", default=None)
    p_best.add_argument(
        "--algorithm", default=None, help="Filter by algorithm: dqn or a2c"
    )
    p_best.add_argument("--limit", type=int, default=20)

    p_compare = sub.add_parser(
        "compare", help="Aggregate stats grouped by model/hyperparams"
    )
    p_compare.add_argument("--env", default=None)
    p_compare.add_argument("--model-type", default=None)
    p_compare.add_argument(
        "--algorithm", default=None, help="Filter by algorithm: dqn or a2c"
    )

    sub.add_parser("today", help="Today's experiments: last-10 and total avg reward")

    p_sql = sub.add_parser("sql", help="Run a raw SQL query against live files")
    p_sql.add_argument("query")

    args = parser.parse_args()

    if args.command == "today":
        cmd_today(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "best":
        cmd_best(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "sql":
        cmd_sql(args)


if __name__ == "__main__":
    main()
