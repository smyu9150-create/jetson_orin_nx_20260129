import os
import time
import torch
import torch_tensorrt  # registers TRT classes for torch.jit.load
from movinets import MoViNet
from movinets.config import _C


def benchmark_model(model, input_tensor, name, num_runs=50, warmup=10):
    device = input_tensor.device
    print(f"[{name}] Warming up... ({warmup} iters)")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"[{name}] Benchmarking... ({num_runs} runs)")
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.time()

    avg_s = (end - start) / num_runs
    fps = 1.0 / avg_s
    return avg_s, fps


def build_base_a3(device):
    config = _C.MODEL.MoViNetA3
    model = MoViNet(config, causal=False, pretrained=False)
    model.to(device).eval()
    return model


def main():
    # === TRT 엔진이 요구하는 고정 shape ===
    B, C, T, H, W = 1, 3, 120, 256, 256
    TRT_MODEL_PATH = "movinet_a3_fp16_trt_full.ts"
    RUNS = 50
    WARMUP = 10
    # ====================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Comparison on: {device}")
    print(f"Input shape: {[B,C,T,H,W]}")
    print("-" * 90)

    if not os.path.exists(TRT_MODEL_PATH):
        print(f"File not found: {TRT_MODEL_PATH}")
        return

    # inputs
    x_fp32 = torch.randn(B, C, T, H, W, device=device, dtype=torch.float32)
    x_fp16 = x_fp32.half()

    results = {}

    # ---- Base A3 (FP32) ----
    try:
        print("Loading Base MoViNet-A3 (FP32)...")
        base_model = build_base_a3(device)
        t_base, fps_base = benchmark_model(
            base_model, x_fp32, "Base A3 (FP32)", RUNS, WARMUP
        )
        results["base"] = (t_base, fps_base)

        del base_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error benchmarking Base A3: {e}")

    print("-" * 90)

    # ---- TRT A3 (FP16) ----
    try:
        print("Loading TRT MoViNet-A3 (FP16)...")
        trt_model = torch.jit.load(TRT_MODEL_PATH, map_location=device)
        trt_model.to(device).eval()

        t_trt, fps_trt = benchmark_model(
            trt_model, x_fp16, "TRT A3 (FP16)", RUNS, WARMUP
        )
        results["trt"] = (t_trt, fps_trt)

        del trt_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error benchmarking TRT A3: {e}")

    # ---- Report ----
    print("\n" + "=" * 90)
    print(f"{'Model Version':<28} | {'Latency (ms)':>12} | {'FPS':>10} | {'Speedup':>10}")
    print("-" * 90)

    base_fps = results.get("base", (0.0, 0.0))[1]

    if "base" in results:
        t, fps = results["base"]
        print(f"{'Base MoViNet-A3 (FP32)':<28} | {t*1000:>12.2f} | {fps:>10.2f} | {'1.00x':>10}")

    if "trt" in results:
        t, fps = results["trt"]
        speedup = (fps / base_fps) if base_fps > 0 else 0.0
        print(f"{'TRT MoViNet-A3 (FP16)':<28} | {t*1000:>12.2f} | {fps:>10.2f} | {speedup:>9.2f}x")

    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
