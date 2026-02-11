"""
Load test: 200 requests, 5 concurrent at a time.
Measures latency, throughput, and system resource usage.
"""
import time
import os
import subprocess
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "http://localhost:8000/tts"
TOTAL_REQUESTS = 100
CONCURRENCY = 5
TEST_TEXTS = [
    "Hello, this is a load test for Pocket TTS.",
    "Testing concurrent requests on the server.",
    "How fast can this model generate speech?",
    "The quick brown fox jumps over the lazy dog.",
    "Pocket TTS is designed to run efficiently on CPUs.",
]


def get_system_load():
    """Get current CPU and memory usage."""
    # CPU load averages
    load = os.getloadavg()
    # Memory via vm_stat
    result = subprocess.run(["vm_stat"], capture_output=True, text=True)
    lines = result.stdout.split("\n")
    page_size = 16384  # Apple Silicon page size
    free = 0
    active = 0
    for line in lines:
        if "Pages free" in line:
            free = int(line.split(":")[1].strip().rstrip(".")) * page_size
        if "Pages active" in line:
            active = int(line.split(":")[1].strip().rstrip(".")) * page_size
    return {
        "load_1m": load[0],
        "load_5m": load[1],
        "load_15m": load[2],
        "mem_active_gb": active / 1e9,
    }


def make_request(request_id):
    """Make a single TTS request and return timing info."""
    text = TEST_TEXTS[request_id % len(TEST_TEXTS)]
    start = time.time()

    response = requests.post(URL, data={"text": text}, stream=True)
    first_byte_time = None
    total_bytes = 0

    for chunk in response.iter_content(chunk_size=4096):
        if first_byte_time is None:
            first_byte_time = time.time() - start
        total_bytes += len(chunk)

    total_time = time.time() - start

    return {
        "id": request_id,
        "status": response.status_code,
        "first_byte_s": first_byte_time or 0,
        "total_time_s": total_time,
        "bytes": total_bytes,
    }


def main():
    print(f"{'='*60}")
    print(f"POCKET TTS LOAD TEST")
    print(f"{'='*60}")
    print(f"Total requests: {TOTAL_REQUESTS}")
    print(f"Concurrency:    {CONCURRENCY}")
    print()

    # System info before
    load_before = get_system_load()
    print(f"System load BEFORE: {load_before['load_1m']:.2f} / {load_before['load_5m']:.2f} / {load_before['load_15m']:.2f}")
    print(f"Memory active:      {load_before['mem_active_gb']:.1f} GB")
    print()

    results = []
    errors = 0
    start_all = time.time()

    print(f"Running {TOTAL_REQUESTS} requests with {CONCURRENCY} concurrent...")
    print()

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {executor.submit(make_request, i): i for i in range(TOTAL_REQUESTS)}

        completed = 0
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                completed += 1
                if result["status"] != 200:
                    errors += 1

                if completed % 20 == 0 or completed == TOTAL_REQUESTS:
                    elapsed = time.time() - start_all
                    rps = completed / elapsed
                    load = get_system_load()
                    print(f"  [{completed:3d}/{TOTAL_REQUESTS}] "
                          f"RPS: {rps:.1f} | "
                          f"CPU Load: {load['load_1m']:.1f} | "
                          f"Elapsed: {elapsed:.1f}s")
            except Exception as e:
                errors += 1
                completed += 1
                print(f"  ERROR on request {futures[future]}: {e}")

    total_time = time.time() - start_all

    # System info after
    load_after = get_system_load()

    # Compute stats
    times = [r["total_time_s"] for r in results]
    first_bytes = [r["first_byte_s"] for r in results]
    total_bytes_all = sum(r["bytes"] for r in results)

    times.sort()
    first_bytes.sort()

    print()
    print(f"{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Total time:         {total_time:.1f}s")
    print(f"Successful:         {len(results) - errors}/{TOTAL_REQUESTS}")
    print(f"Errors:             {errors}")
    print(f"Throughput:         {len(results)/total_time:.2f} req/s")
    print(f"Total data:         {total_bytes_all/1e6:.1f} MB")
    print()
    print(f"--- Latency (total response time) ---")
    print(f"  Min:    {min(times):.3f}s")
    print(f"  Avg:    {sum(times)/len(times):.3f}s")
    print(f"  P50:    {times[len(times)//2]:.3f}s")
    print(f"  P95:    {times[int(len(times)*0.95)]:.3f}s")
    print(f"  P99:    {times[int(len(times)*0.99)]:.3f}s")
    print(f"  Max:    {max(times):.3f}s")
    print()
    print(f"--- Time to first byte ---")
    print(f"  Min:    {min(first_bytes):.3f}s")
    print(f"  Avg:    {sum(first_bytes)/len(first_bytes):.3f}s")
    print(f"  P50:    {first_bytes[len(first_bytes)//2]:.3f}s")
    print(f"  P95:    {first_bytes[int(len(first_bytes)*0.95)]:.3f}s")
    print(f"  Max:    {max(first_bytes):.3f}s")
    print()
    print(f"--- System Load ---")
    print(f"  Before: {load_before['load_1m']:.2f} / {load_before['load_5m']:.2f} / {load_before['load_15m']:.2f}")
    print(f"  After:  {load_after['load_1m']:.2f} / {load_after['load_5m']:.2f} / {load_after['load_15m']:.2f}")
    print(f"  Peak CPU cores used (estimate): ~{load_after['load_1m']:.1f}")
    print(f"  Memory active: {load_after['mem_active_gb']:.1f} GB")


if __name__ == "__main__":
    main()
