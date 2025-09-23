from __future__ import annotations

import logging
import os
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DEFAULT_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: Optional[str] = None, *, force: bool = True) -> None:
    """Configure root logging once per process.

    Level can be overridden via argument or the ``LOG_LEVEL`` env var. Subsequent
    calls replace prior configuration when ``force`` is true (Python >=3.8).
    """

    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")

    if isinstance(level, str):
        resolved_level = getattr(logging, level.upper(), logging.INFO)
    else:
        resolved_level = int(level)

    logging.basicConfig(level=resolved_level, format=_DEFAULT_FORMAT, datefmt=_DEFAULT_DATE_FMT, force=force)

