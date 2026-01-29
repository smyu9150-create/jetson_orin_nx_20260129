import os
import glob
import time
from dataclasses import dataclass

import cv2
import torch
from torch2trt import TRTModule


# ======================
# 설정 (여기만 바꾸면 됨)
# ======================
VIDEO_DIR = "/path/to/your/video_folder"   # <-- 영상 폴더 경로
TRT_PTH   = "trt_models/movinet_a0_base_trt.pth"  # <-- 변환된 TRTModule pth

# 네 엔진 입력 shape에 맞춰야 함!
T = 8          # 너 로그에서 input=...x8x... 이었음
H = 172
W = 172

STRIDE = 8     # 클립을 몇 프레임 간격으로 자를지 (8이면 non-overlap)
BATCH  = 1     # torch2trt/TRTModule은 보통 batch=1이 가장 안정적

WARMUP_ITERS = 10
BENCH_ITERS_PER_CLIP = 1   # 클립 1개당 forward 몇 번 반복할지 (간단 벤치면 1)

# 비디오 확장자
EXTS = ("*.mp4", "*.avi", "*.mov", "*.mkv")


@dataclass
class ClipResult:
    n_clips: int
    total_sec: float
    avg_ms: float
    fps_clips: float


def list_videos(folder: str):
    files = []
    for ext in EXTS:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


def read_video_frames_cv2(path: str):
    """
    비디오 전체 프레임을 BGR로 읽어서 리스트로 반환.
    (간단 버전: 메모리 많이 쓰는 대신 구현 쉬움)
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)  # BGR uint8
    cap.release()
    return frames


def preprocess_clip(frames_bgr, h: int, w: int):
    """
    frames_bgr: 길이 T, 각 원소는 (H,W,3) BGR uint8
    return: torch.Tensor [1,3,T,H,W] float32 (0~1)
    """
    # resize + BGR->RGB
    resized = []
    for f in frames_bgr:
        f2 = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)
        f2 = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)
        resized.append(f2)

    # (T,H,W,3) -> (1,3,T,H,W)
    x = torch.from_numpy(
        (torch.tensor(resized).numpy())  # 안전하게 numpy로
    )  # uint8
    # x shape: (T,H,W,3)
    x = x.permute(3, 0, 1, 2).contiguous()  # (3,T,H,W)
    x = x.unsqueeze(0)  # (1,3,T,H,W)
    x = x.float().div(255.0)
    return x.cuda(non_blocking=True)


@torch.no_grad()
def bench_on_video(model_trt: torch.nn.Module, video_path: str) -> ClipResult:
    frames = read_video_frames_cv2(video_path)
    n_frames = len(frames)
    if n_frames < T:
        print(f"  - skip (too short): {n_frames} frames < T={T}")
        return ClipResult(n_clips=0, total_sec=0.0, avg_ms=0.0, fps_clips=0.0)

    # 클립 인덱스 생성 (stride 간격)
    starts = list(range(0, n_frames - T + 1, STRIDE))
    n_clips = len(starts)

    # 워밍업 (첫 클립로)
    first_clip = preprocess_clip(frames[starts[0]:starts[0]+T], H, W)
    for _ in range(WARMUP_ITERS):
        _ = model_trt(first_clip)
    torch.cuda.synchronize()

    # 벤치
    t0 = time.time()
    total_runs = 0

    for s in starts:
        clip = preprocess_clip(frames[s:s+T], H, W)
        for _ in range(BENCH_ITERS_PER_CLIP):
            _ = model_trt(clip)
            total_runs += 1

    torch.cuda.synchronize()
    t1 = time.time()

    total_sec = t1 - t0
    # "클립 처리 FPS" (클립/초). BENCH_ITERS_PER_CLIP>1이면 runs 기준도 같이 참고 가능.
    fps_clips = n_clips / total_sec if total_sec > 0 else 0.0
    avg_ms = (total_sec / max(total_runs, 1)) * 1000.0

    return ClipResult(n_clips=n_clips, total_sec=total_sec, avg_ms=avg_ms, fps_clips=fps_clips)


def main():
    assert torch.cuda.is_available(), "CUDA not available"

    print("Loading TRTModule:", TRT_PTH)
    model_trt = TRTModule()
    # 경고 피하고 싶으면 weights_only=True (PyTorch 2.4+에서 지원)
    try:
        sd = torch.load(TRT_PTH, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(TRT_PTH, map_location="cpu")
    model_trt.load_state_dict(sd)
    model_trt.eval().cuda()

    videos = list_videos(VIDEO_DIR)
    if not videos:
        raise RuntimeError(f"No videos found in: {VIDEO_DIR}")

    print(f"Found {len(videos)} videos.")
    print(f"Engine expects input: [1,3,T,H,W] = [1,3,{T},{H},{W}], stride={STRIDE}")

    total_clips = 0
    total_time = 0.0

    for vp in videos:
        print(f"\n[VIDEO] {os.path.basename(vp)}")
        try:
            r = bench_on_video(model_trt, vp)
        except Exception as e:
            print("  - ERROR:", e)
            continue

        if r.n_clips == 0:
            continue

        total_clips += r.n_clips
        total_time += r.total_sec

        print(f"  clips: {r.n_clips}")
        print(f"  time : {r.total_sec:.3f} sec")
        print(f"  avg  : {r.avg_ms:.3f} ms / forward (per clip-run)")
        print(f"  fps  : {r.fps_clips:.2f} clips/sec")

    if total_clips > 0 and total_time > 0:
        print("\n==== SUMMARY ====")
        print(f"total clips: {total_clips}")
        print(f"total time : {total_time:.3f} sec")
        print(f"overall    : {total_clips/total_time:.2f} clips/sec")


if __name__ == "__main__":
    main()
