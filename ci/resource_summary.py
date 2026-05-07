import subprocess
import threading
import time
import statistics
import psutil
import json
import sys
import platform
import shutil
import os
from datetime import datetime


SAMPLE_INTERVAL = 1.0
OUTPUT_DIR = "ci/artifacts"
OUTPUT_JSON = f"{OUTPUT_DIR}/resource_summary.json"


# Helpers

def gb(value):
    return round(value / (1024 ** 3), 2)


def format_seconds(seconds):
    mins, sec = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)

    if hrs > 0:
        return f"{hrs:02d}h{mins:02d}m{sec:02d}s"
    elif mins > 0:
        return f"{mins:02d}m{sec:02d}s"

    return f"{sec}s"


def get_system_info():
    mem = psutil.virtual_memory()
    disk = shutil.disk_usage("/")

    return {
        "platform": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "cpu_physical": psutil.cpu_count(logical=False),
        "cpu_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": gb(mem.total),
        "ram_available_gb": gb(mem.available),
        "ram_percent": mem.percent,
        "disk_total_gb": gb(disk.total),
        "disk_free_gb": gb(disk.free),
    }


# ============================================================
# System summary mode
# ============================================================

def print_system_summary():
    info = get_system_info()

    print("\n" + "=" * 60)
    print("SYSTEM RESOURCE SUMMARY")
    print("=" * 60)
    print(f"Timestamp        : {datetime.now().isoformat()}")
    print(f"Platform         : {info['platform']}")
    print(f"Python           : {info['python_version']}")
    print(f"CPU physical     : {info['cpu_physical']}")
    print(f"CPU logical      : {info['cpu_logical']}")
    print(f"RAM total        : {info['ram_total_gb']} GB")
    print(f"RAM available    : {info['ram_available_gb']} GB")
    print(f"RAM usage        : {info['ram_percent']} %")
    print(f"Disk total       : {info['disk_total_gb']} GB")
    print(f"Disk free        : {info['disk_free_gb']} GB")
    print("=" * 60)


# ============================================================
# Monitoring
# ============================================================

def monitor(stop_event, samples):
    psutil.cpu_percent(interval=0.1)

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


# ============================================================
# Run command
# ============================================================

def run_command(command):
    return subprocess.Popen(
        command,
        shell=True,
        stdout=sys.stdout,
        stderr=sys.stderr
    )


# ============================================================
# Summary
# ============================================================

def summarize(command, start_time, end_time, returncode, samples):
    duration = end_time - start_time

    cpu_values = [s["cpu_percent"] for s in samples] or [0]
    ram_values = [s["ram_used_gb"] for s in samples] or [0]
    ram_pct_values = [s["ram_percent"] for s in samples] or [0]

    return {
        "timestamp": datetime.now().isoformat(),

        "system": get_system_info(),

        "command": command,
        "duration_seconds": round(duration, 2),
        "duration_minutes": format_seconds(duration),
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


def save_json(summary):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)


# ============================================================
# Main
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("python ci/resource_summary.py --system")
        print('python ci/resource_summary.py "pytest ..."')
        sys.exit(1)

    # --------------------------------------------------------
    # System mode
    # --------------------------------------------------------

    if sys.argv[1] == "--system":
        print_system_summary()
        sys.exit(0)

    # --------------------------------------------------------
    # Monitoring mode
    # --------------------------------------------------------

    command = " ".join(sys.argv[1:])

    print(f"\nLaunching command:\n{command}\n")

    samples = []
    stop_event = threading.Event()

    watcher = threading.Thread(
        target=monitor,
        args=(stop_event, samples)
    )

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

    save_json(summary)

    sys.exit(process.returncode)


if __name__ == "__main__":
    main()