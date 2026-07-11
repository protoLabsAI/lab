"""House bots for the live mythx world — two band strategies riding every wipe.

The live world (mythx-world.service) serves `builtin:skirmish:<seed>` forever;
this client keeps both teams played so the world is never empty: raider on
team A, hoarder on team B (the ladder's headline rivalry). Each band rejoins
after every wipe with a fresh brain, per the operator protocol. Replace a bot
with your own program by stopping this service and claiming its controllers.

Requires the mythxengine-sdk repo's `.venv-mmo` (mythx_sdk installed) and its
`examples/` on the path.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

SDK = Path(os.environ.get("MYTHX_SDK_ROOT", str(Path.home() / "dev/mythxengine-sdk")))
sys.path.insert(0, str(SDK / "examples"))
import band as B  # noqa: E402

ADDR = os.environ.get("MYTHX_WORLD_ADDR", "127.0.0.1:47420")
TEAMS = {
    "raider": ("operator_a", B.RaiderBand),
    "hoarder": ("operator_b", B.HoarderBand),
}


def ride(name: str, faction: str, cls) -> None:
    """Play seasons forever: a fresh brain per season; back off while the
    server is down (deploys, reboots) instead of spinning."""
    seasons = 0
    while True:
        brain = cls()
        played = []

        def pawn(cid: str) -> None:
            try:
                B.connect_and_serve(
                    B._Member(brain, cid), ADDR, cid,
                    actor_entity_id=cid, pack_versions=B.PACK_VERSIONS,
                )
                played.append(cid)
            except Exception:
                pass

        controllers = [f"{faction}_{i}" for i in range(3)]
        threads = [threading.Thread(target=pawn, args=(c,), daemon=True) for c in controllers]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if played:
            seasons += 1
            print(f"[{name}] season {seasons} done", flush=True)
        else:
            time.sleep(30)  # server unreachable — wait it out


def main() -> None:
    riders = [
        threading.Thread(target=ride, args=(n, f, c), daemon=True)
        for n, (f, c) in TEAMS.items()
    ]
    for r in riders:
        r.start()
    for r in riders:
        r.join()


if __name__ == "__main__":
    main()
