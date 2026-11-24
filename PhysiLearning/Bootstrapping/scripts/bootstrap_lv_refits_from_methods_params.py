#!/usr/bin/env python3
from __future__ import annotations

import sys

from bootstrap_lv_refits import main


def add_default(flag: str, value: str) -> None:
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


if __name__ == "__main__":
    add_default("--seed-mode", "agent_training_frozen")
    add_default("--outdir", "outputs/bootstrap_n20_methods_seed_maxiter1000")
    raise SystemExit(main())
