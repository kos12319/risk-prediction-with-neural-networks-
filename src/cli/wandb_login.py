from __future__ import annotations

import os
import argparse

from src.training.wandb_sync import login_from_env


def main():
    parser = argparse.ArgumentParser(description="Login to Weights & Biases using environment variables")
    parser.add_argument("--quiet", action="store_true", help="Do not print status messages")
    args = parser.parse_args()

    ok = login_from_env()
    if not args.quiet:
        if ok:
            ent = os.environ.get("WANDB_ENTITY") or os.environ.get("WB_ENTITY") or "(no entity set)"
            print(f"W&B login successful. Entity: {ent}")
        else:
            print("W&B login failed. Set WANDB_API_KEY in environment.")


if __name__ == "__main__":
    main()
