import torch
import cv2
import time
import numpy as np
import os
import gc
import json
import urllib.request
import csv

# === CONFIGURATION ===
VIDEO_DIR = "/home/etri/data/video"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ITERATIONS = 50          # per-video 측정 반복
WARMUP_ITERS = 20        # 워밍업
OUTPUT_CSV = "x3d_benchmark_results_precise.csv"

MODELS_CONFIG = [
    ("x3d_xs", "x3d_xs", 4,  12, 182),
    ("x3d_xs",  "x3d_xs",  13, 6,  182),
    ("x3d_m",  "x3d_m",  16, 5,  256),
]

# ------------------------
# Utils
# ------------------------
def ensure_cuda():
    if DEVICE.type != "cuda":
        print("⚠️ CUDA가 아닙니다. CPU 측정은 의미가 많이 달라집니다.")
    else:
        torch.backends.cudnn.benchmark = True

def load_kinetics_labels():
    json_url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
    json_filename = "kinetics_classnames.json"
    if not os.path.exists(json_filename):
        try:
            urllib.request.urlretrieve(json_url, json_filename)
        except:
            return None
    with open(json_filename, "r") as f:
        data = json.load(f)
    return {int(v): k.replace('"', "") for k, v in data.items()}

def process_video(video_path, target_frames, target_size):
    """
    return: input tensor (1, 3, T, H, W) float32 on DEVICE
    전처리 시간은 main()에서 perf_counter로 별도 측정
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        scale = target_size / min(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        th, tw = frame.shape[:2]
        y0 = max(0, (th - target_size) // 2)
        x0 = max(0, (tw - target_size) // 2)
        frame = frame[y0:y0 + target_size, x0:x0 + target_size]

        # RGB
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    if len(frames) < 1:
        return None

    # temporal subsample
    indices = np.linspace(0, len(frames) - 1, target_frames).astype(int)
    clip = np.stack([frames[i] for i in indices], axis=0)  # (T, H, W, C)

    video_tensor = torch.from_numpy(clip).float() / 255.0
    video_tensor = video_tensor.permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, T, H, W)

    mean = torch.tensor([0.45, 0.45, 0.45], device=DEVICE).view(1, 3, 1, 1, 1)
    std  = torch.tensor([0.225, 0.225, 0.225], device=DEVICE).view(1, 3, 1, 1, 1)

    video_tensor = video_tensor.to(DEVICE, non_blocking=True)
    return (video_tensor - mean) / std

@torch.no_grad()
def warmup(model, x, warmup_iters=WARMUP_ITERS):
    model.eval()
    for _ in range(warmup_iters):
        _ = model(x)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

@torch.no_grad()
def measure_inference_cuda_events(model, x, iters=ITERATIONS):
    """
    CUDA 이벤트 기반
    - per-iter latency(ms) 분포: p50/p90/p99
    - total time(ms) 기반 throughput(clips/s)
    """
    model.eval()

    if DEVICE.type != "cuda":
        # CPU fallback (정확도 떨어짐)
        times = []
        t0 = time.perf_counter()
        for _ in range(iters):
            s = time.perf_counter()
            _ = model(x)
            e = time.perf_counter()
            times.append((e - s) * 1000.0)
        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000.0
        lat = np.array(times, dtype=np.float64)
        return _stats_from_latencies(lat, total_ms, iters)

    torch.cuda.synchronize()

    start_total = torch.cuda.Event(enable_timing=True)
    end_total   = torch.cuda.Event(enable_timing=True)

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    start_total.record()
    for i in range(iters):
        starts[i].record()
        _ = model(x)
        ends[i].record()
    end_total.record()

    torch.cuda.synchronize()

    total_ms = start_total.elapsed_time(end_total)
    lat_ms = np.array([starts[i].elapsed_time(ends[i]) for i in range(iters)], dtype=np.float64)

    return _stats_from_latencies(lat_ms, total_ms, iters)

def _stats_from_latencies(lat_ms: np.ndarray, total_ms: float, iters: int):
    lat_ms_sorted = np.sort(lat_ms)
    mean_ms = float(lat_ms_sorted.mean())
    p50_ms  = float(np.percentile(lat_ms_sorted, 50))
    p90_ms  = float(np.percentile(lat_ms_sorted, 90))
    p99_ms  = float(np.percentile(lat_ms_sorted, 99))

    throughput = float(iters / (total_ms / 1000.0))  # clips/s (batch=1 기준)
    return {
        "total_ms": float(total_ms),
        "mean_ms": mean_ms,
        "p50_ms": p50_ms,
        "p90_ms": p90_ms,
        "p99_ms": p99_ms,
        "throughput_clips_s": throughput,
    }

@torch.no_grad()
def predict_label(model, x, labels_map):
    out = model(x)
    top1 = int(torch.argmax(out, dim=1).item())
    if labels_map is None:
        return str(top1)
    return labels_map.get(top1, str(top1))

# ------------------------
# Main
# ------------------------
def main():
    ensure_cuda()

    video_files = sorted([
        f for f in os.listdir(VIDEO_DIR)
        if f.lower().endswith(('.mp4', '.avi', '.mkv'))
    ])

    if not video_files:
        print(f"❌ No videos found in {VIDEO_DIR}")
        return

    labels_map = load_kinetics_labels()

    detailed_rows = []
    summary_rows = []

    print(f"\n[X3D Precise Benchmark | Torch Hub]")
    print(f"Videos: {len(video_files)} | Iterations/video: {ITERATIONS} | Warmup: {WARMUP_ITERS}")
    print("-" * 170)
    print(f"{'Model':<8} | {'Video':<30} | {'Preproc(ms)':<11} | {'Mean(ms)':<9} | {'P50':<7} | {'P90':<7} | {'P99':<7} | {'Throughput(clips/s)':<19} | {'Peak VRAM(MB)':<13} | {'Pred':<18}")
    print("-" * 170)

    for model_name, hub_name, frames, sampling, size in MODELS_CONFIG:
        model_throughputs = []
        model_mean_lat = []
        model_p99_lat = []
        model_peak_mem = []

        try:
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

            # load model
            model = torch.hub.load('facebookresearch/pytorchvideo', hub_name, pretrained=True).to(DEVICE).eval()

            for video_file in video_files:
                video_path = os.path.join(VIDEO_DIR, video_file)

                # (A) 전처리 시간: CPU wall-time
                t0 = time.perf_counter()
                x = process_video(video_path, frames, size)
                if x is None:
                    continue
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                preproc_ms = (t1 - t0) * 1000.0

                # (B) 워밍업: (shape/데이터 동일)
                warmup(model, x, warmup_iters=WARMUP_ITERS)

                # (C) Peak VRAM 측정 리셋 (추론만 peak 보고 싶으면 여기서 reset)
                if DEVICE.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(DEVICE)

                # (D) 정확한 추론 측정
                stats = measure_inference_cuda_events(model, x, iters=ITERATIONS)

                # (E) peak vram (MB)
                if DEVICE.type == "cuda":
                    peak_mb = float(torch.cuda.max_memory_allocated(DEVICE) / (1024 * 1024))
                else:
                    peak_mb = 0.0

                # (F) 예측(측정 이후 1회만)
                pred = predict_label(model, x, labels_map)

                detailed_rows.append([
                    model_name, video_file,
                    round(preproc_ms, 2),
                    round(stats["mean_ms"], 4),
                    round(stats["p50_ms"], 4),
                    round(stats["p90_ms"], 4),
                    round(stats["p99_ms"], 4),
                    round(stats["throughput_clips_s"], 3),
                    round(peak_mb, 1),
                    pred
                ])

                print(f"{model_name:<8} | {video_file[:30]:<30} | "
                      f"{preproc_ms:<11.2f} | {stats['mean_ms']:<9.3f} | {stats['p50_ms']:<7.3f} | {stats['p90_ms']:<7.3f} | {stats['p99_ms']:<7.3f} | "
                      f"{stats['throughput_clips_s']:<19.2f} | {peak_mb:<13.1f} | {pred[:18]:<18}")

                model_throughputs.append(stats["throughput_clips_s"])
                model_mean_lat.append(stats["mean_ms"])
                model_p99_lat.append(stats["p99_ms"])
                model_peak_mem.append(peak_mb)

            # model summary
            if model_throughputs:
                summary_rows.append([
                    model_name,
                    round(float(np.mean(model_throughputs)), 3),
                    round(float(np.mean(model_mean_lat)), 4),
                    round(float(np.mean(model_p99_lat)), 4),
                    round(float(np.max(model_peak_mem)), 1),
                    "Success"
                ])
            else:
                summary_rows.append([model_name, 0, 0, 0, 0, "NoData"])

            del model
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            summary_rows.append([model_name, 0, 0, 0, 0, f"Error: {e}"])
            print(f"❌ Error on {model_name}: {e}")

    # --- Save CSV ---
    with open(OUTPUT_CSV, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model", "Video File",
            "Preproc_ms", "Mean_ms", "P50_ms", "P90_ms", "P99_ms",
            "Throughput_clips_s",
            "Peak_VRAM_MB",
            "Predicted_Class"
        ])
        writer.writerows(detailed_rows)

    print("\n" + "=" * 95)
    print(f"{'SUMMARY (Inference-only)':^95}")
    print("=" * 95)
    print(f"{'Model':<10} | {'Avg Throughput(clips/s)':<22} | {'Avg Mean(ms)':<13} | {'Avg P99(ms)':<12} | {'Max Peak(MB)':<12} | {'Status'}")
    print("-" * 95)
    for r in summary_rows:
        print(f"{r[0]:<10} | {r[1]:<22} | {r[2]:<13} | {r[3]:<12} | {r[4]:<12} | {r[5]}")
    print("=" * 95)
    print(f"✅ Detailed results saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    # numpy import here to avoid unused if someone trims code
    import numpy as np
    main()
