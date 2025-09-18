from __future__ import annotations

import argparse
import sys
from typing import List
import runpy


def _run_module_with_argv(module: str, argv: List[str]) -> int:
    """Run a module as if via `python -m module` with a custom argv.

    Returns process-like exit code (0 on success).
    """
    old_argv = sys.argv
    try:
        sys.argv = [module, *argv]
        runpy.run_module(module, run_name="__main__")
        return 0
    finally:
        sys.argv = old_argv


def main() -> None:
    description = (
        "Credit risk prediction CLI with training, dry runs, feature selection, "
        "column dictionary generation, and Weights & Biases utilities."
    )
    epilog = (
        "Examples:\n"
        "  Train (from YAML config):\n"
        "    python -m src.cli train --config configs/default.yaml\n\n"
        "  Dry run (no artifacts):\n"
        "    python -m src.cli dryrun --config configs/default.yaml\n\n"
        "  Feature selection (mutual information):\n"
        "    python -m src.cli select --config configs/default.yaml --method mi\n\n"
        "  Column dictionary from CSV (optionally override csv path):\n"
        "    python -m src.cli gen-column-dict --config configs/default.yaml\n"
        "    python -m src.cli gen-column-dict --config configs/default.yaml --csv data/raw/samples/thesis_data_sample_10k.csv\n\n"
        "  W&B login and downloads:\n"
        "    export WANDB_API_KEY=...; export WANDB_ENTITY=your_entity\n"
        "    python -m src.cli wandb-login\n"
        "    python -m src.cli pull-run --run your_entity/loan-risk-mlp/abcd1234\n"
        "    python -m src.cli pull-all --entity your_entity --project loan-risk-mlp\n\n"
        "Environment:\n"
        "  WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT control W&B behavior.\n"
        "  See configs/default.yaml for training and tracking options."
    )

    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    # Train
    sp_train = subparsers.add_parser(
        "train",
        help="Train the PyTorch MLP using a YAML config",
        description=(
            "Train the model per config. Supports CPU-only mode, notes tagging, and optional "
            "post-training W&B file/artifact download into the local run folder."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
    )

    # Dry run
    sp_dry = subparsers.add_parser(
        "dryrun",
        help="Run a dry training pass without writing artifacts",
        description=(
            "Loads data, builds the pipeline, runs a short training/eval, and prints a JSON summary. "
            "Useful for quick sanity checks."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
    )

    # Feature selection
    sp_select = subparsers.add_parser(
        "select",
        help="Run feature selection (mi or l1) and save artifacts",
        description=(
            "Ranks features and builds a compact subset targetting a fraction of full AUC. "
            "Artifacts (ranking/results/curves) are saved under reports/selection/<method>/."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
    )

    # Column dictionary
    sp_dict = subparsers.add_parser(
        "gen-column-dict",
        help="Generate Markdown column dictionary (types, missingness, categories)",
        description=(
            "Scans a CSV and produces a Markdown table summarizing column types, missingness, and top categories."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
    )

    # W&B helpers
    sp_wb_login = subparsers.add_parser(
        "wandb-login",
        help="Login to Weights & Biases using WANDB_API_KEY",
        description="Logs into W&B using environment variables (WANDB_API_KEY, WANDB_ENTITY).",
        add_help=False,
    )

    sp_wb_pull = subparsers.add_parser(
        "pull-run",
        help="Download files/artifacts for a single W&B run",
        description=(
            "Downloads all files and artifacts for a given W&B run into a local folder. "
            "Accepts run as entity/project/run_id, project/run_id (with WANDB_ENTITY), or run_id (with env defaults)."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
    )

    sp_wb_all = subparsers.add_parser(
        "pull-all",
        help="Download files/artifacts for all runs in a project",
        description=(
            "Lists runs via the W&B public API for ENTITY/PROJECT and downloads each run's files and artifacts "
            "into local_runs/<run_id>/wandb by default (configurable)."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
    )

    # If no subcommand, show help
    if len(sys.argv) <= 1:
        parser.print_help(sys.stderr)
        sys.exit(2)

    # Parse only the subcommand; forward the rest
    args, remainder = parser.parse_known_args()
    cmd = args.command

    # Strip the leading '--' if users separate args explicitly
    if remainder and remainder[0] == "--":
        remainder = remainder[1:]

    # Dispatch
    if cmd == "train":
        sys.exit(_run_module_with_argv("src.cli.train", remainder))
    if cmd == "dryrun":
        sys.exit(_run_module_with_argv("src.cli.dryrun", remainder))
    if cmd == "select":
        sys.exit(_run_module_with_argv("src.cli.select", remainder))
    if cmd == "gen-column-dict":
        sys.exit(_run_module_with_argv("src.cli.gen_column_dict", remainder))
    if cmd == "wandb-login":
        sys.exit(_run_module_with_argv("src.cli.wandb_login", remainder))
    if cmd == "pull-run":
        sys.exit(_run_module_with_argv("src.cli.wandb_pull", remainder))
    if cmd == "pull-all":
        sys.exit(_run_module_with_argv("src.cli.wandb_pull_all", remainder))

    parser.error(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
