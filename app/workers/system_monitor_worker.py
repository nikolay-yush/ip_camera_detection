import os
import time
import psutil


def system_monitor_worker(interval: float = 5.0):
    """
    Logs system resource utilization:
    - CPU (System): % of total CPU capacity across all cores (0-100%).
    - CPU (Cores): Sum of multi-core usage (100% = 1 full core).
    - RAM: Physical memory used by this process.
    - Threads: Total OS threads (Python app threads + C++ PyTorch/OpenCV worker pools).
    """
    process = psutil.Process(os.getpid())
    cpu_count = psutil.cpu_count(logical=True) or 1
    
    # Warm-up call for CPU percent calculation
    process.cpu_percent(interval=None)

    while True:
        time.sleep(interval)
        try:
            # Raw sum across all cores (e.g., 370.0%)
            raw_cpu = process.cpu_percent(interval=None)
            
            # Normalized overall CPU usage (0-100%)
            system_cpu_pct = raw_cpu / cpu_count
            
            # Resident Set Size (RAM in MB)
            ram_mb = process.memory_info().rss / (1024 * 1024)
            
            # Total native threads
            num_threads = process.num_threads()

            print(
                f"[MONITOR] "
                f"CPU (System): {system_cpu_pct:5.1f}% | "
                f"CPU (Cores): {raw_cpu:5.1f}% ({cpu_count} cores) | "
                f"RAM: {ram_mb:6.1f} MB | "
                f"Threads: {num_threads} (App + C++ Pool)"
            )
        except Exception as exc:
            print(f"[MONITOR] Metric extraction error: {exc}")