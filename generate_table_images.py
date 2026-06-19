"""
Generate LaTeX table images from comparison JSON results.

For sweep environments (many models per type): Top-10 tables per model type,
    2 tables per page (LSTM+RNN on page 1, MPN+MPN-Frozen on page 2).
For single-model environments: One summary table comparing all model types.

Usage:
    python generate_table_images.py                  # Generate all
    python generate_table_images.py --env gonogo      # Generate one
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

RESULTS_DIR = Path("results")
IMAGES_DIR = Path("images")

# Display names for environments
ENV_DISPLAY_NAMES = {
    "ContextDecisionMaking-v0": "ContextDecisionMaking",
    "DelayComparison-v0": "DelayComparison",
    "DelayMatchSample-v0": "DelayMatchSample",
    "DelayMatchSampleDistractor1D-v0": "DelayMatchSampleDistractor1D",
    "DelayPairedAssociation-v0": "DelayPairedAssociation",
    "GoNogo-v0": "GoNogo",
    "IntervalDiscrimination-v0": "IntervalDiscrimination",
    "MultiSensoryIntegration-v0": "MultiSensoryIntegration",
    "PerceptualDecisionMaking-v0": "PerceptualDecisionMaking",
    "PerceptualDecisionMakingDelayResponse-v0": "PerceptualDecisionMakingDelayResponse",
    "ProbabilisticReasoning-v0": "ProbabilisticReasoning",
}

MODEL_TYPE_ORDER = ["lstm", "rnn", "mpn", "mpn-frozen"]
MODEL_TYPE_LABELS = {
    "lstm": "LSTM",
    "rnn": "RNN",
    "mpn": "MPN",
    "mpn-frozen": "MPN-Frozen",
}


def load_results(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def group_models_by_type(results):
    """Group non-random models by model_type."""
    groups = {}
    for name, data in results["models"].items():
        if name == "random":
            continue
        # Skip 50k frame experiments
        hp = data.get("hyperparameters", {})
        tf = hp.get("total_frames")
        if tf is not None and int(tf) == 50000:
            continue
        mt = data["model_type"]
        if mt not in groups:
            groups[mt] = []
        groups[mt].append((name, data))
    # Sort each group by mean reward descending
    for mt in groups:
        groups[mt].sort(key=lambda x: x[1]["cumulative_reward"]["mean"], reverse=True)
    return groups


def fmt_reward(val):
    """Format reward value for LaTeX."""
    if val < 0:
        return f"$-${abs(val):.2f}"
    return f"{val:.2f}"


def fmt_pct(mean_reward, random_mean):
    """Format percentage improvement vs random."""
    if random_mean == 0:
        return "$-$"
    pct = ((mean_reward - random_mean) / abs(random_mean)) * 100
    if pct >= 0:
        return f"$+{pct:.0f}\\%$"
    else:
        return f"$-{abs(pct):.0f}\\%$"


def fmt_frames(total_frames):
    """Format frame count (e.g., 200000 -> 200k)."""
    if total_frames is None:
        return "—"
    tf = int(total_frames)
    if tf >= 1_000_000:
        return f"{tf / 1_000_000:.1f}M"
    elif tf >= 1000:
        return f"{tf // 1000}k"
    return str(tf)


def generate_sweep_table(model_type, entries, env_name, random_mean, top_n=10):
    """Generate a LaTeX table for a sweep model type (top N configs)."""
    is_mpn = model_type in ("mpn", "mpn-frozen")
    label = MODEL_TYPE_LABELS[model_type]
    total_count = len(entries)
    entries = entries[:top_n]

    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")

    if is_mpn:
        lines.append("\\begin{tabular}{clcccccc}")
    else:
        lines.append("\\begin{tabular}{clcccc}")

    lines.append("\\toprule")

    if is_mpn:
        lines.append(
            "Rank & Learning Rate & $\\eta$ & $\\lambda$ & Frames & Mean Reward & Std & \\% vs Random \\\\"
        )
    else:
        lines.append(
            "Rank & Learning Rate & Frames & Mean Reward & Std & \\% vs Random \\\\"
        )
    lines.append("\\midrule")

    for rank, (name, data) in enumerate(entries, 1):
        hp = data.get("hyperparameters", {})
        lr = hp.get("learning_rate", "—")
        if isinstance(lr, float):
            lr = f"{lr:.4f}"
        frames = fmt_frames(hp.get("total_frames"))
        mean = data["cumulative_reward"]["mean"]
        std = data["cumulative_reward"]["std"]
        pct = fmt_pct(mean, random_mean)

        mean_str = fmt_reward(mean)
        if rank == 1:
            mean_str = f"\\textbf{{{mean_str}}}"

        if is_mpn:
            eta = hp.get("eta", "—")
            lam = hp.get("lambda_decay", "—")
            if isinstance(eta, float):
                eta = f"{eta:.2f}"
            if isinstance(lam, float):
                lam = f"{lam:.2f}"
            lines.append(
                f"{rank} & {lr} & {eta} & {lam} & {frames} & {mean_str} & {std:.2f} & {pct} \\\\"
            )
        else:
            lines.append(
                f"{rank} & {lr} & {frames} & {mean_str} & {std:.2f} & {pct} \\\\"
            )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    caption = f"Top {min(top_n, total_count)} {label} configurations on {env_name} (of {total_count} total). Random baseline mean: {random_mean:.2f}."
    if is_mpn and model_type == "mpn":
        caption += (
            " $\\eta$ and $\\lambda$ columns show plasticity-specific hyperparameters."
        )
    elif is_mpn and model_type == "mpn-frozen":
        caption += " $\\eta$ and $\\lambda$ are fixed architectural parameters (plasticity frozen during training)."

    lines.append(f"\\caption{{{caption}}}")
    env_short = env_name.replace("-v0", "").lower()
    lines.append(f"\\label{{tab:{env_short}-{model_type}}}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def generate_single_table(results, env_name):
    """Generate a single comparison table for environments with one model per type."""
    random_mean = results["models"]["random"]["cumulative_reward"]["mean"]
    groups = group_models_by_type(results)

    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Model & Params & Mean Reward & Std & \\% vs Random \\\\")
    lines.append("\\midrule")

    # Find best mean for bolding
    best_mean = -float("inf")
    for mt in MODEL_TYPE_ORDER:
        if mt in groups and groups[mt]:
            m = groups[mt][0][1]["cumulative_reward"]["mean"]
            if m > best_mean:
                best_mean = m

    for mt in MODEL_TYPE_ORDER:
        if mt not in groups or not groups[mt]:
            continue
        name, data = groups[mt][0]
        label = MODEL_TYPE_LABELS[mt]
        params = f"{data['parameter_count']:,}"
        mean = data["cumulative_reward"]["mean"]
        std = data["cumulative_reward"]["std"]
        pct = fmt_pct(mean, random_mean)

        mean_str = fmt_reward(mean)
        if mean == best_mean:
            mean_str = f"\\textbf{{{mean_str}}}"

        lines.append(f"{label} & {params} & {mean_str} & {std:.2f} & {pct} \\\\")

    lines.append("\\midrule")
    # Random baseline row
    rd = results["models"]["random"]
    lines.append(
        f"Random & — & {fmt_reward(rd['cumulative_reward']['mean'])} & {rd['cumulative_reward']['std']:.2f} & — \\\\"
    )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    env_short = env_name.replace("-v0", "").lower()
    lines.append(
        f"\\caption{{Model comparison on {env_name}. Random baseline mean: {rd['cumulative_reward']['mean']:.2f}.}}"
    )
    lines.append(f"\\label{{tab:{env_short}-comparison}}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def generate_latex_document(tables):
    """Wrap tables in a full LaTeX document."""
    preamble = [
        "\\documentclass[11pt]{article}",
        "\\usepackage[margin=0.5in]{geometry}",
        "\\usepackage{booktabs}",
        "\\usepackage{amsmath}",
        "\\raggedbottom",
        "\\pagestyle{empty}",
        "",
        "\\begin{document}",
        "",
    ]
    closing = ["", "\\end{document}", ""]
    return "\n".join(preamble + tables + closing)


def compile_latex(tex_path, output_dir):
    """Compile LaTeX to PDF and convert to PNG images."""
    tex_path = Path(tex_path)
    output_dir = Path(output_dir)

    # Compile LaTeX
    result = subprocess.run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            "-output-directory",
            str(output_dir),
            str(tex_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  LaTeX compilation warning (may still produce output):")
        # Only print last few lines of error
        err_lines = result.stdout.strip().split("\n")
        for line in err_lines[-5:]:
            print(f"    {line}")

    pdf_path = output_dir / tex_path.with_suffix(".pdf").name
    if not pdf_path.exists():
        print(f"  ERROR: PDF not generated for {tex_path.name}")
        return []

    # Convert PDF pages to PNG
    stem = tex_path.stem
    subprocess.run(
        ["pdftoppm", "-png", "-r", "300", str(pdf_path), str(output_dir / stem)],
        capture_output=True,
    )

    # Collect generated PNGs
    pngs = sorted(output_dir.glob(f"{stem}-*.png"))
    # Also handle single-page case (pdftoppm may name it differently)
    single = output_dir / f"{stem}.png"
    if single.exists() and single not in pngs:
        pngs.insert(0, single)

    return pngs


def process_environment(json_path):
    """Process a single environment JSON and generate table images."""
    results = load_results(json_path)
    env_name = results["metadata"]["environment"]
    env_short = env_name.replace("-v0", "").lower()
    display_name = ENV_DISPLAY_NAMES.get(env_name, env_name)

    print(f"\nProcessing: {env_name}")

    groups = group_models_by_type(results)
    random_mean = results["models"]["random"]["cumulative_reward"]["mean"]

    # Determine if sweep or single-model
    total_trained = sum(len(v) for v in groups.values())
    is_sweep = total_trained > len(groups)  # More models than model types

    tables = []
    if is_sweep:
        # Sweep: top-10 tables per model type, 2 per page
        page_pairs = [("lstm", "rnn"), ("mpn", "mpn-frozen")]
        for i, (mt1, mt2) in enumerate(page_pairs):
            if mt1 in groups:
                tables.append(
                    generate_sweep_table(mt1, groups[mt1], env_name, random_mean)
                )
            if mt2 in groups:
                tables.append(
                    generate_sweep_table(mt2, groups[mt2], env_name, random_mean)
                )
            # Add page break between pairs (but not after the last pair)
            if i < len(page_pairs) - 1:
                tables.append("\\newpage")
    else:
        # Single-model: one comparison table
        tables.append(generate_single_table(results, env_name))

    # Generate LaTeX document
    tex_content = generate_latex_document(tables)
    tex_filename = f"{env_short}_tables"
    tex_path = IMAGES_DIR / f"{tex_filename}.tex"

    IMAGES_DIR.mkdir(exist_ok=True)
    with open(tex_path, "w") as f:
        f.write(tex_content)
    print(f"  Wrote: {tex_path}")

    # Compile and convert
    pngs = compile_latex(tex_path, IMAGES_DIR)
    for png in pngs:
        print(f"  Generated: {png}")

    return pngs


def main():
    parser = argparse.ArgumentParser(
        description="Generate table images from comparison results"
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Environment short name (e.g., 'gonogo', 'delaycomparison'). If omitted, generates all.",
    )
    args = parser.parse_args()

    json_files = sorted(RESULTS_DIR.glob("compare_*.json"))
    if not json_files:
        print("No result JSON files found in results/")
        return

    if args.env:
        # Filter to matching environment
        target = args.env.lower().replace("-", "").replace("_", "")
        json_files = [
            f
            for f in json_files
            if target in f.stem.replace("compare_", "").replace("_", "")
        ]
        if not json_files:
            print(f"No result file found matching: {args.env}")
            return

    all_pngs = []
    for json_path in json_files:
        pngs = process_environment(json_path)
        all_pngs.extend(pngs)

    print(f"\nDone! Generated {len(all_pngs)} images in {IMAGES_DIR}/")


if __name__ == "__main__":
    main()
