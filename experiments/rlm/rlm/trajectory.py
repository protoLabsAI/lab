"""Append a Trajectory as one JSONL line.

We write whole-row append, not streaming, so a crashed session produces no
partial row. For long sessions where that matters, swap to per-turn JSONL with
a session_end sentinel.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from rlm.schema import Trajectory


def write_trajectory(traj: Trajectory, dirpath: str) -> Path:
    Path(dirpath).mkdir(parents=True, exist_ok=True)
    # One file per day keeps grep/replay manageable; session_id distinguishes within.
    from datetime import datetime, timezone

    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = Path(dirpath) / f"trajectories-{day}.jsonl"
    line = traj.model_dump_json()
    # O_APPEND on POSIX is atomic for writes < PIPE_BUF (4096); our rows are larger
    # so use lockless append-only and accept rare interleaving — acceptable for SFT.
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
    return path
