#!/usr/bin/env python3
"""
resource_monitor.py

Universal resource monitor for any shell command.

Usage:
    python resource_monitor.py "pytest -ra -v -s -c argopy/tests/pytest.ini"
    python resource_monitor.py "git push"
    python resource_monitor.py "pip install -r requirements.txt"

Requires:
    pip install psutil
"""

import subprocess
import threading
import time
import statistics
import psutil
import json
import sys
import shlex
from datetime import datetime


SAMPLE_INTERVAL = 1.0
OUTPUT_JSON = "resource_summary.json"


def format_seconds(seconds):
    mins, sec = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)

    if hrs > 0:
        return f"{hrs:02d}h{mins:02d}m{sec:02d}s"
    elif mins > 0:
        return f"{mins:02d}m{sec:02d}s"
    return f"{sec}s"


def gb(value):
    return round(value / (1024 ** 3), 2)


def monitor(stop_event, samples):
    while not stop_event.is_set():
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()

        samples.append({
            "timestamp": time.time(),
            "cpu_percent": cpu,
            "ram_used_gb": gb(mem.used),
            "ram_available_gb": gb(mem.available),
            "ram_percent": mem.percent
        })

        time.sleep(SAMPLE_INTERVAL)


def run_command(command):
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    return process


def summarize(command, start_time, end_time, returncode, samples):
    duration = end_time - start_time

    cpu_values = [s["cpu_percent"] for s in samples] or [0]
    ram_values = [s["ram_used_gb"] for s in samples] or [0]
    ram_pct_values = [s["ram_percent"] for s in samples] or [0]

    summary = {
        "timestamp": datetime.now().isoformat(),
        "command": command,
        "duration_seconds": round(duration, 2),
        "duration_human": format_seconds(duration),
        "exit_code": returncode,

        "cpu": {
            "avg_percent": round(statistics.mean(cpu_values), 2),
            "max_percent": round(max(cpu_values), 2),
            "min_percent": round(min(cpu_values), 2),
        },

        "memory": {
            "avg_used_gb": round(statistics.mean(ram_values), 2),
            "peak_used_gb": round(max(ram_values), 2),
            "min_used_gb": round(min(ram_values), 2),
            "avg_percent": round(statistics.mean(ram_pct_values), 2),
            "max_percent": round(max(ram_pct_values), 2),
        },

        "samples_collected": len(samples)
    }

    return summary


def print_summary(summary):
    print("\n" + "=" * 50)
    print("RESOURCE SUMMARY")
    print("=" * 50)
    print(f"Command      : {summary['command']}")
    print(f"Duration     : {summary['duration_human']}")
    print(f"Exit code    : {summary['exit_code']}")
    print("")
    print("CPU")
    print(f"  Avg        : {summary['cpu']['avg_percent']} %")
    print(f"  Max        : {summary['cpu']['max_percent']} %")
    print("")
    print("Memory")
    print(f"  Avg used   : {summary['memory']['avg_used_gb']} GB")
    print(f"  Peak used  : {summary['memory']['peak_used_gb']} GB")
    print(f"  Avg usage  : {summary['memory']['avg_percent']} %")
    print(f"  Max usage  : {summary['memory']['max_percent']} %")
    print("")
    print(f"Samples      : {summary['samples_collected']}")
    print(f"Saved JSON   : {OUTPUT_JSON}")
    print("=" * 50)


def save_json(summary):
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print('python resource_monitor.py "your command here"')
        sys.exit(1)

    command = " ".join(sys.argv[1:])

    print(f"Launching command:\n{command}\n")

    samples = []
    stop_event = threading.Event()

    watcher = threading.Thread(target=monitor, args=(stop_event, samples))
    watcher.daemon = True

    start_time = time.time()
    watcher.start()

    process = run_command(command)
    process.wait()

    end_time = time.time()
    stop_event.set()
    watcher.join(timeout=2)

    summary = summarize(
        command=command,
        start_time=start_time,
        end_time=end_time,
        returncode=process.returncode,
        samples=samples
    )

    print_summary(summary)
    save_json(summary)

    sys.exit(process.returncode)


if __name__ == "__main__":
    main()