#!/usr/bin/env python3
"""CLI shim for the skill-card specialization candidate proposer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.clean_self_distill.propose import main


if __name__ == "__main__":
    main()
