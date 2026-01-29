import torch
import torch_tensorrt  # noqa: F401 (로드만 해도 엔진 실행에 필요할 수 있음)
import sys
import os
import glob
import time
import json
import urllib.request
import torchvision.transforms.functional as F
import numpy as np

# ----------------- [긴급 패치] -----------------
sys.modules["torchvision.transforms.functional_tensor"] = F
# ------------------------------------------------

from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample

# ==========================================
# [설정]
VIDEO_DIR = '/home/etri/data/video'      # 비디오 폴더
ENGINE_PATH = "x3d_xs_trt_fp16.ts"        # TorchScript/TRT 엔진 파일(.ts)
ITERATIONS = 50                          # 비디오당 반복 측정 횟수(측정용)
WARMUP_ITERS = 20                        # 워밍업 반복 횟수
PRINT_TOPK = 1                           # 결과 확인용
# ==========================================

device = "cuda"

def ensure_cuda():
    if not torch.cuda.is_available():
        print("❌ CUDA 사용 불가. Jetson/CUDA 환경 확인 필요.")
        sys.exit(1)

def load_labels():
    json_filename = "kinetics_classnames.json"
    if not os.path.exists(json_filename):
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json",
            json_filename
        )
    with open(json_filename, "r") as f:
        kinetics_id_to_classname = {}
        for k, v in json.load(f).items():
            kinetics_id_to_classname[v] = str(k).replace('"', "")
    return kinetics_id_to_classname

def build_transform():
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    return ApplyTransformToKey(
        key="video",
        transform=Compose([
            UniformTemporalSubsample(13),          # T=13
            Lambda(lambda x: x / 255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(size=182),
            CenterCropVideo(crop_size=(182, 182)),
        ]),
    )

@torch.no_grad()
def warmup(model, input_tensor, warmup_iters=WARMUP_ITERS):
    for _ in range(warmup_iters):
        _ = model(input_tensor)
    torch.cuda.synchronize()

@torch.no_grad()
def measure_inference_cuda_events(model, input_tensor, iters=ITERATIONS):
    """
    - per-iter latency: CUDA events로 측정 (ms)
    - total time: CUDA events로 측정 (ms)
    - throughput: iters / (total_ms/1000)
    """
    model.eval()

    # 전체 시간(throughput용)
    start_total = torch.cuda.Event(enable_timing=True)
    end_total = torch.cuda.Event(enable_timing=True)

    # per-iter latency 분포(p50/p90/p99용)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    torch.cuda.synchronize()

    start_total.record()
    for i in range(iters):
        starts[i].record()
        _ = model(input_tensor)
        ends[i].record()
    end_total.record()

    torch.cuda.synchronize()

    # total ms
    total_ms = start_total.elapsed_time(end_total)

    # per-iter ms
    lat_ms = np.array([starts[i].elapsed_time(ends[i]) for i in range(iters)], dtype=np.float64)

    # 통계
    mean_ms = float(lat_ms.mean())
    p50_ms  = float(np.percentile(lat_ms, 50))
    p90_ms  = float(np.percentile(lat_ms, 90))
    p99_ms  = float(np.percentile(lat_ms, 99))
    # throughput (clips/s) — total time 기반이 가장 정확
    throughput = float(iters / (total_ms / 1000.0))

    return {
        "total_ms": float(total_ms),
        "mean_ms": mean_ms,
        "p50_ms": p50_ms,
        "p90_ms": p90_ms,
        "p99_ms": p99_ms,
        "throughput_clips_s": throughput,
    }

def predict_label(model, input_tensor, id2name):
    with torch.no_grad():
        preds = model(input_tensor)
        probs = torch.softmax(preds, dim=1)
        topk = probs.topk(PRINT_TOPK)
        top1_idx = int(topk.indices[0][0])
        return id2name.get(top1_idx, str(top1_idx))

def main():
    ensure_cuda()

    # 1) 모델 로드
    print(f"📂 Loading Model from {ENGINE_PATH}...")
    if not os.path.exists(ENGINE_PATH):
        print("❌ 모델 파일이 없습니다!")
        sys.exit(1)

    model = torch.jit.load(ENGINE_PATH).eval().to(device)
    print("✅ Model Loaded!")

    # 2) 라벨 로드
    kinetics_id_to_classname = load_labels()

    # 3) 전처리
    transform = build_transform()
    clip_duration = (13 * 6) / 30  # 기존 설정 유지 (T=13, stride=6, fps=30 가정)

    # 4) 비디오 목록
    video_files = sorted(
        glob.glob(os.path.join(VIDEO_DIR, '*.mp4')) +
        glob.glob(os.path.join(VIDEO_DIR, '*.avi'))
    )
    if not video_files:
        print("❌ 비디오 파일이 없습니다.")
        sys.exit(1)

    # 5) 워밍업 (shape 맞춘 dummy)
    dummy = torch.randn(1, 3, 13, 182, 182, device=device, dtype=torch.float16)
    print(f"🔥 Warming up... ({WARMUP_ITERS} iters)")
    warmup(model, dummy, warmup_iters=WARMUP_ITERS)
    print("✅ Warm-up Done!\n")

    # 출력 헤더
    print(f"🚀 Benchmark Start | Videos: {len(video_files)} | Iters/video: {ITERATIONS} | Batch=1")
    print("-" * 140)
    print(f"{'Filename':<35} | {'Preproc(ms)':<11} | {'Mean(ms)':<9} | {'P50':<7} | {'P90':<7} | {'P99':<7} | {'Throughput(clips/s)':<19} | {'Prediction':<20}")
    print("-" * 140)

    # 글로벌 누적(추론만)
    global_total_ms = 0.0
    global_total_iters = 0
    global_mean_list = []

    for video_path in video_files:
        filename = os.path.basename(video_path)

        try:
            # (A) 전처리 시간(디코딩+transform) 측정: CPU wall time
            t0 = time.perf_counter()
            video = EncodedVideo.from_path(video_path)
            video_data = video.get_clip(start_sec=0, end_sec=clip_duration)
            video_data = transform(video_data)
            # 입력 준비
            inputs = video_data["video"].to(device).half()
            inputs = inputs[None, ...]  # (1, C, T, H, W)
            torch.cuda.synchronize()  # 전처리 이후 GPU sync (안정적으로 분리)
            t1 = time.perf_counter()
            preproc_ms = (t1 - t0) * 1000.0

            # (B) 추론 시간/분포 측정: CUDA events
            stats = measure_inference_cuda_events(model, inputs, iters=ITERATIONS)

            # (C) 예측 라벨(측정 후 1회만)
            pred_label = predict_label(model, inputs, kinetics_id_to_classname)

            print(f"{filename[:33]:<35} | "
                  f"{preproc_ms:<11.2f} | "
                  f"{stats['mean_ms']:<9.3f} | "
                  f"{stats['p50_ms']:<7.3f} | "
                  f"{stats['p90_ms']:<7.3f} | "
                  f"{stats['p99_ms']:<7.3f} | "
                  f"{stats['throughput_clips_s']:<19.2f} | "
                  f"{pred_label[:20]:<20}")

            global_total_ms += stats["total_ms"]
            global_total_iters += ITERATIONS
            global_mean_list.append(stats["mean_ms"])

        except Exception as e:
            print(f"{filename[:33]:<35} | Error: {e}")

    print("-" * 140)
    if global_total_iters > 0:
        global_throughput = global_total_iters / (global_total_ms / 1000.0)
        global_mean_ms = float(np.mean(global_mean_list)) if global_mean_list else float("nan")
        print(f"✅ GLOBAL (Inference-only, CUDA events)")
        print(f"   - Total clips: {global_total_iters}")
        print(f"   - Total inference time: {global_total_ms:.2f} ms")
        print(f"   - Global throughput: {global_throughput:.2f} clips/sec")
        print(f"   - Avg of per-video mean latency: {global_mean_ms:.3f} ms/clip")
        print("   * Note: 전처리 시간은 위 표 Preproc(ms)에 별도 측정됨(추론 throughput에 포함 X).")
    else:
        print("❌ 측정된 결과가 없습니다.")
    print("-" * 140)

if __name__ == "__main__":
    main()
