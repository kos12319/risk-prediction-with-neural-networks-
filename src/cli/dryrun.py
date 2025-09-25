from __future__ import annotations

import sys


def main() -> None:
    msg = (
        "Unified CLI entry 'src.cli.dryrun' is deprecated.\n"
        "Use Makefile targets or backend-specific CLIs instead:\n"
        "  - make dryrun              # PyTorch backend smoke test\n"
        "  - make dryrun-h2o          # H2O backend smoke test\n"
        "Or call modules explicitly:\n"
        "  python -m src.cli.pytorch.dryrun --config configs/pytorch_default.yaml\n"
        "  python -m src.cli.h2o.dryrun     --config configs/h2o_default.yaml\n"
    )
    sys.stderr.write(msg)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
