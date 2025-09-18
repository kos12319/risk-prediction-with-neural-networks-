from __future__ import annotations

import os
import platform


def apply_safe_env() -> None:
    """Apply conservative env settings to avoid BLAS/Accelerate crashes on macOS and ensure headless plotting.

    - Limit thread counts for common BLAS backends
    - Force headless Matplotlib backend
    - Set cache dirs to project-local folders if not set
    - On Apple Silicon, hint OpenBLAS core type
    """
    # BLAS thread caps
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    # Prefer sequential MKL to avoid OpenMP SHM usage in constrained envs
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    # OpenMP runtime knobs to avoid SHM and forking issues
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_PROC_BIND", "FALSE")

    # Headless plotting
    os.environ.setdefault("MPLBACKEND", "Agg")

    # Local cache/config dirs to avoid permission issues
    os.environ.setdefault("XDG_CACHE_HOME", ".cache")
    os.environ.setdefault("MPLCONFIGDIR", ".mplcache")

    # Apple Silicon hint for OpenBLAS
    try:
        if platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}:
            os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
    except Exception:
        pass
