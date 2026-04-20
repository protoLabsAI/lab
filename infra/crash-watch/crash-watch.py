#!/usr/bin/env python3
"""
Crash-watch telemetry logger.

Writes one CSV row per second with GPU/CPU/NVMe/mem metrics, fsync'd on every
write so a hard power loss leaves the tail on disk up to T-1s.

Usage:
    python3 crash-watch.py                      # log to /mnt/scratch/logs/crash-watch/<ts>.csv
    python3 crash-watch.py --out /path/out.csv  # custom path
    python3 crash-watch.py --interval 0.5       # faster polling
"""

import argparse
import csv
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

GPU_FIELDS = [
    "power.draw",
    "temperature.gpu",
    "clocks.sm",
    "clocks.mem",
    "pstate",
    "clocks_throttle_reasons.active",
    "utilization.gpu",
    "memory.used",
]

HEADER = (
    ["ts", "wall"]
    + [f"gpu{i}_{f.replace('.', '_')}" for i in (0, 1) for f in GPU_FIELDS]
    + [
        "cpu_tctl", "cpu_ccd1", "cpu_ccd2",
        "mb_cpu", "mb_cpu_pkg", "mb_motherboard", "mb_vrm",
        "nvme0_c", "nvme1_c", "nvme2_c", "nvme3_c",
        "load1", "mem_avail_kb", "mem_free_kb",
    ]
)


def read_gpu():
    query = ",".join(GPU_FIELDS)
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            text=True, timeout=2,
        )
    except Exception:
        return [""] * (len(GPU_FIELDS) * 2)
    rows = [r.strip() for r in out.strip().splitlines()]
    result = []
    for i in range(2):
        if i < len(rows):
            result.extend([c.strip() for c in rows[i].split(",")])
        else:
            result.extend([""] * len(GPU_FIELDS))
    return result


_SENSORS_PAT = re.compile(r"^\s*([^:]+?):\s*\+?([\-\d.]+)")


def read_sensors():
    try:
        out = subprocess.check_output(["sensors"], text=True, timeout=2)
    except Exception:
        return {}
    blocks = {}
    current = None
    for line in out.splitlines():
        if not line.strip():
            current = None
            continue
        if current is None and not line.startswith(" ") and "-" in line:
            current = line.strip()
            blocks[current] = {}
            continue
        if current:
            m = _SENSORS_PAT.match(line)
            if m:
                blocks[current][m.group(1)] = m.group(2)
    return blocks


def pick(blocks, adapter_prefix, key):
    for name, fields in blocks.items():
        if name.startswith(adapter_prefix) and key in fields:
            return fields[key]
    return ""


def read_loadavg():
    try:
        return Path("/proc/loadavg").read_text().split()[0]
    except Exception:
        return ""


def read_meminfo():
    avail = free = ""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                avail = line.split()[1]
            elif line.startswith("MemFree:"):
                free = line.split()[1]
    except Exception:
        pass
    return avail, free


def collect_row():
    ts = time.time()
    wall = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    row = [f"{ts:.3f}", wall]
    row.extend(read_gpu())

    blocks = read_sensors()
    row.append(pick(blocks, "k10temp", "Tctl"))
    row.append(pick(blocks, "k10temp", "Tccd1"))
    row.append(pick(blocks, "k10temp", "Tccd2"))
    row.append(pick(blocks, "asusec", "CPU"))
    row.append(pick(blocks, "asusec", "CPU Package") or pick(blocks, "asusec", "CPU_Package"))
    row.append(pick(blocks, "asusec", "Motherboard"))
    row.append(pick(blocks, "asusec", "VRM"))

    nvme_temps = []
    for name in sorted(k for k in blocks if k.startswith("nvme-pci-")):
        nvme_temps.append(blocks[name].get("Composite", ""))
    nvme_temps += [""] * (4 - len(nvme_temps))
    row.extend(nvme_temps[:4])

    row.append(read_loadavg())
    avail, free = read_meminfo()
    row.append(avail)
    row.append(free)
    return row


def main():
    p = argparse.ArgumentParser()
    default_out = f"/mnt/scratch/logs/crash-watch/{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    p.add_argument("--out", default=default_out)
    p.add_argument("--interval", type=float, default=1.0)
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stop = {"flag": False}

    def handler(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    print(f"[crash-watch] logging to {out_path} every {args.interval}s (pid {os.getpid()})", flush=True)

    with open(out_path, "w", newline="", buffering=1) as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        f.flush()
        os.fsync(f.fileno())

        next_tick = time.time()
        while not stop["flag"]:
            try:
                row = collect_row()
                w.writerow(row)
                f.flush()
                os.fsync(f.fileno())
            except Exception as e:
                print(f"[crash-watch] collect error: {e}", file=sys.stderr, flush=True)
            next_tick += args.interval
            sleep_for = max(0.0, next_tick - time.time())
            time.sleep(sleep_for)

    print(f"[crash-watch] stopped, log at {out_path}", flush=True)


if __name__ == "__main__":
    main()
