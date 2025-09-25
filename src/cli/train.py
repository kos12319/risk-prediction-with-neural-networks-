from __future__ import annotations

import sys


def main() -> None:
    msg = (
        "Unified CLI entry 'src.cli.train' is deprecated.\n"
        "Use Makefile targets or backend-specific CLIs instead:\n"
        "  - make train               # PyTorch backend\n"
        "  - make automl-h2o          # H2O AutoML backend\n"
        "Or call modules explicitly:\n"
        "  python -m src.cli.pytorch.train --config configs/pytorch_default.yaml\n"
        "  python -m src.cli.h2o.train     --config configs/h2o_default.yaml\n"
    )
    sys.stderr.write(msg)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
