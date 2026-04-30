import subprocess
import threading
import time
import statistics
import psutil
import json
import sys
import platform
import shutil
from datetime import datetime


SAMPLE_INTERVAL = 1.0
OUTPUT_JSON = "resource_summary.json"


# --------------------------------------------------
# Helpers
# --------------------------------------------------

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


def gha_group_start(title):
    print(f"::group::{title}")


def gha_group_end():
    print("::endgroup::")


# --------------------------------------------------
# System summary mode
# --------------------------------------------------

def print_system_summary():
    mem = psutil.virtual_memory()
    disk = shutil.disk_usage("/")
    cpu_phys = psutil.cpu_count(logical=False)
    cpu_log = psutil.cpu_count(logical=True)

    gha_group_start("System Resource Summary")

    print("=" * 60)
    print("SYSTEM RESOURCE SUMMARY")
    print("=" * 60)
    print(f"Timestamp        : {datetime.now().isoformat()}")
    print(f"Platform         : {platform.system()} {platform.release()}")
    print(f"Python           : {platform.python_version()}")
    print(f"CPU physical     : {cpu_phys}")
    print(f"CPU logical      : {cpu_log}")
    print(f"RAM total        : {gb(mem.total)} GB")
    print(f"RAM available    : {gb(mem.available)} GB")
    print(f"RAM usage        : {mem.percent} %")
    print(f"Disk total       : {gb(disk.total)} GB")
    print(f"Disk free        : {gb(disk.free)} GB")
    print("=" * 60)

    gha_group_end()


# --------------------------------------------------
# Monitoring
# --------------------------------------------------

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


# --------------------------------------------------
# Run command
# --------------------------------------------------

def run_command(command):
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    return process


# --------------------------------------------------
# Summary
# --------------------------------------------------

def summarize(command, start_time, end_time, returncode, samples):
    duration = end_time - start_time

    cpu_values = [s["cpu_percent"] for s in samples] or [0]
    ram_values = [s["ram_used_gb"] for s in samples] or [0]
    ram_pct_values = [s["ram_percent"] for s in samples] or [0]

    return {
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


def print_summary(summary):
    gha_group_start("Resource Usage Summary")

    print("=" * 60)
    print("RESOURCE SUMMARY")
    print("=" * 60)
    print(f"Command          : {summary['command']}")
    print(f"Duration         : {summary['duration_human']}")
    print(f"Exit code        : {summary['exit_code']}")
    print("")
    print("CPU")
    print(f"  Avg            : {summary['cpu']['avg_percent']} %")
    print(f"  Max            : {summary['cpu']['max_percent']} %")
    print(f"  Min            : {summary['cpu']['min_percent']} %")
    print("")
    print("Memory")
    print(f"  Avg used       : {summary['memory']['avg_used_gb']} GB")
    print(f"  Peak used      : {summary['memory']['peak_used_gb']} GB")
    print(f"  Min used       : {summary['memory']['min_used_gb']} GB")
    print(f"  Avg usage      : {summary['memory']['avg_percent']} %")
    print(f"  Max usage      : {summary['memory']['max_percent']} %")
    print("")
    print(f"Samples          : {summary['samples_collected']}")
    print(f"Saved JSON       : {OUTPUT_JSON}")
    print("=" * 60)

    gha_group_end()


def save_json(summary):
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("python resource_summary.py --system")
        print('python resource_summary.py "pytest ..."')
        sys.exit(1)

    if sys.argv[1] == "--system":
        print_system_summary()
        sys.exit(0)

    command = " ".join(sys.argv[1:])

    print(f"\nLaunching command:\n{command}\n")

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