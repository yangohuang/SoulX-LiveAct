"""Run a monitored, single-arm pageable/pinned weight-transfer comparison."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def available_mib() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise RuntimeError("MemAvailable absent from /proc/meminfo")


def gpu_process_mib(pid: int) -> int | None:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False, timeout=5,
    )
    if result.returncode:
        return None
    for line in result.stdout.splitlines():
        columns = [piece.strip() for piece in line.split(",")]
        if len(columns) == 2 and columns[0] == str(pid):
            return int(columns[1])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("pageable", "pinned"), required=True)
    parser.add_argument("--size", choices=("224*384", "416*720"), required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    headroom = available_mib()
    if headroom < 40 * 1024:
        raise RuntimeError(f"preflight MemAvailable {headroom} MiB < 40960 MiB")
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    request_path = prefix.with_name(prefix.name + "-request.json")
    output_path = prefix.with_suffix(".mp4")
    request = [{
        "prompt": "一个人在说话",
        "cond_image": "examples/image/1.png",
        "cond_audio": "/tmp/liveact-local-5s.wav",
        "output_path": str(output_path),
    }]
    request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n")
    cmd = [
        sys.executable, "generate.py", "--size", args.size,
        "--ckpt_dir", "checkpoints/LiveAct",
        "--wav2vec_dir", "checkpoints/chinese-wav2vec2-base",
        "--fps", "24", "--seed", "42", "--denoising_steps", "3",
        "--input_json", str(request_path),
        "--fp8_gemm", "--fp8_kv_cache", "--offload_cache", "--block_offload",
        "--t5_cpu", "--disable_compile", "--dura_print",
        "--resident_kv_steps", "1",
        "--fp8_cache_dir", "/tmp/liveact-fp8-cache",
        "--prompt_cache_dir", "/tmp/liveact-prompt-cache",
    ]
    if args.arm == "pinned":
        cmd.append("--pin_block_memory")
    env = os.environ.copy()
    env["USE_CHANNELS_LAST_3D"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    record = {
        "arm": args.arm, "size": args.size, "request": str(request_path),
        "command": cmd, "environment": {name: env[name] for name in ("USE_CHANNELS_LAST_3D", "CUDA_VISIBLE_DEVICES")},
        "preflight_mem_available_mib": headroom,
    }
    samples = []
    log_path = prefix.with_suffix(".log")
    started = time.monotonic()
    with log_path.open("w") as log:
        process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        low_count = 0
        try:
            while process.poll() is None:
                available = available_mib()
                gpu = gpu_process_mib(process.pid)
                sample = {"elapsed_s": round(time.monotonic() - started, 3),
                          "mem_available_mib": available, "gpu_process_mib": gpu}
                samples.append(sample)
                low_count = low_count + 1 if available < 8 * 1024 else 0
                if low_count >= 2:
                    record["aborted_reason"] = "MemAvailable below 8 GiB for two samples"
                    process.terminate()
                    break
                if len(samples) % 30 == 0:
                    print(f"{args.arm} {args.size}: {sample}", flush=True)
                time.sleep(1)
            try:
                record["returncode"] = process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                record["returncode"] = process.wait()
                record["aborted_reason"] = "process failed to terminate after safety stop"
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=20)
    record["elapsed_s"] = round(time.monotonic() - started, 3)
    record["samples"] = samples
    record["minimum_mem_available_mib"] = min(row["mem_available_mib"] for row in samples)
    gpu_samples = [row["gpu_process_mib"] for row in samples if row["gpu_process_mib"] is not None]
    record["maximum_sampled_gpu_process_mib"] = max(gpu_samples, default=None)
    prefix.with_name(prefix.name + "-resources.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"{args.arm} {args.size}: exit={record['returncode']} elapsed={record['elapsed_s']}s "
          f"minimum_available={record['minimum_mem_available_mib']}MiB "
          f"maximum_gpu={record['maximum_sampled_gpu_process_mib']}MiB", flush=True)
    if record["returncode"]:
        raise SystemExit(record["returncode"])


if __name__ == "__main__":
    main()
