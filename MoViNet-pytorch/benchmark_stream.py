#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import gc
import cv2
import numpy as np
import torch

from movinets import MoViNet
from movinets.config import _C

# =========================
# Config
# =========================
VIDEO_DIR = "/home/etri/data/video"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Kinetics mean/std used commonly in MoViNet examples
MEAN = [0.43216, 0.394666, 0.37645]
STD  = [0.22803, 0.22145, 0.216989]

# Streaming: process every frame? (stride=1) or downsample for speed
STREAM_FRAME_STRIDE = 1   # 1이면 모든 프레임, 2이면 2프레임마다 1번 등

# Warmup iterations (GPU 안정화)
WARMUP_ITERS = 3

MODELS_CONFIG = [
    # name,                cfg,             T,   crop_size, causal(stream)
    ("A0-Base",   _C.MODEL.MoViNetA0,   50,  172, False),
    ("A0-Stream", _C.MODEL.MoViNetA0,   50,  172, True),
    ("A1-Base",   _C.MODEL.MoViNetA1,   50,  172, False),
    ("A1-Stream", _C.MODEL.MoViNetA1,   50,  172, True),
    ("A2-Base",   _C.MODEL.MoViNetA2,   50,  224, False),
    ("A2-Stream", _C.MODEL.MoViNetA2,   50,  224, True),
    ("A3-Base",   _C.MODEL.MoViNetA3,  120,  256, False),
    ("A4-Base",   _C.MODEL.MoViNetA4,   80,  290, False),
    ("A5-Base",   _C.MODEL.MoViNetA5,  120,  320, False),
]

# =========================
# Preprocess utils
# =========================
def short_side_resize_and_center_crop(frame_bgr, crop_size: int):
    """
    frame_bgr: HxWx3 BGR uint8 (OpenCV)
    return: crop_size x crop_size x 3 RGB uint8
    """
    h, w = frame_bgr.shape[:2]
    if h == 0 or w == 0:
        return None

    # short side -> crop_size
    if h < w:
        new_h = crop_size
        new_w = int(round(w * (crop_size / h)))
    else:
        new_w = crop_size
        new_h = int(round(h * (crop_size / w)))

    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # center crop to crop_size
    top = max(0, (new_h - crop_size) // 2)
    left = max(0, (new_w - crop_size) // 2)
    crop = resized[top:top + crop_size, left:left + crop_size]

    if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
        crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return rgb


def to_tensor_rgb_uint8(rgb, device, mean_t, std_t):
    """
    rgb: crop_size x crop_size x 3 uint8
    returns: (1,3,1,H,W) float32 normalized on device
    """
    x = torch.from_numpy(rgb).to(device=device, dtype=torch.float32) / 255.0  # H W C
    x = x.permute(2, 0, 1).unsqueeze(0)  # 1 3 H W
    x = x.unsqueeze(2)  # 1 3 1 H W  (T=1)
    x = (x - mean_t) / std_t
    return x


def sample_indices_uniform(total_frames: int, target_frames: int):
    """
    Uniform sampling indices in [0, total_frames-1] with target_frames points.
    """
    if total_frames <= 0:
        return None
    if target_frames <= 0:
        return np.array([], dtype=np.int64)
    if total_frames == 1:
        return np.zeros((target_frames,), dtype=np.int64)
    idx = np.linspace(0, total_frames - 1, target_frames).astype(np.int64)
    return idx


# =========================
# Video loaders
# =========================
def read_all_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        frames.append(fr)
    cap.release()
    return frames


def build_base_clip_tensor(frames_bgr, T: int, crop_size: int, device, mean_t, std_t):
    """
    frames_bgr: list of BGR uint8 frames (full video)
    Returns (1,3,T,crop,crop) float normalized
    """
    total = len(frames_bgr)
    if total == 0:
        return None

    idx = sample_indices_uniform(total, T)
    clip = []
    for i in idx:
        rgb = short_side_resize_and_center_crop(frames_bgr[int(i)], crop_size)
        if rgb is None:
            return None
        clip.append(rgb)

    arr = np.stack(clip, axis=0)  # T H W C
    x = torch.from_numpy(arr).to(device=device, dtype=torch.float32) / 255.0
    x = x.permute(3, 0, 1, 2).unsqueeze(0)  # 1 3 T H W
    x = (x - mean_t) / std_t
    return x


# =========================
# Streaming forward helper
# =========================
def stream_forward(model, frames_bgr, crop_size, device, mean_t, std_t, frame_stride=1):
    """
    Feed frames one-by-one, preserving internal state buffers.
    Returns: (num_frames_processed, elapsed_seconds)
    """
    if hasattr(model, "clean_activation_buffers"):
        model.clean_activation_buffers()

    processed = 0
    start = time.time()

    with torch.no_grad():
        for fi, fr in enumerate(frames_bgr):
            if frame_stride > 1 and (fi % frame_stride != 0):
                continue

            rgb = short_side_resize_and_center_crop(fr, crop_size)
            if rgb is None:
                continue

            x = to_tensor_rgb_uint8(rgb, device, mean_t, std_t)

            try:
                _ = model(x)
            except Exception:
                try:
                    _ = model(x.squeeze(2))  # (1,3,H,W)
                except Exception:
                    _ = model(x)

            processed += 1

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start
    return processed, elapsed


# =========================
# Pretty summary printer
# =========================
def print_summary_table(title, rows):
    print("\n" + "=" * 90)
    print(f"{title:^90}")
    print("=" * 90)
    print(f"{'Model Name':<15} | {'Global Avg FPS(fr/s)':<22} | {'Max Peak VRAM(MB)':<18} | {'Status'}")
    print("-" * 90)
    for r in rows:
        print(f"{r['name']:<15} | {r['avg_fps']:<22.2f} | {r['peak_vram']:<18.1f} | {r['status']}")
    print("=" * 90)


# =========================
# Benchmark
# =========================
def main():
    video_files = sorted([f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(".mp4")])
    if not video_files:
        print(f"No mp4 files found in: {VIDEO_DIR}")
        return

    mean_t = torch.tensor(MEAN, device=DEVICE, dtype=torch.float32).view(1, 3, 1, 1, 1)
    std_t  = torch.tensor(STD,  device=DEVICE, dtype=torch.float32).view(1, 3, 1, 1, 1)

    # ✅ split summaries
    final_summary_base = []
    final_summary_stream = []

    print(f"\n[Benchmarking on {DEVICE}]")
    print(f"VIDEO_DIR = {VIDEO_DIR}")
    print(f"STREAM_FRAME_STRIDE = {STREAM_FRAME_STRIDE}\n")

    for model_name, config, T, crop_size, is_stream in MODELS_CONFIG:
        print(f"\n>> Testing Model: {model_name}")
        print(f"{'Video File':<25} | {'FPS(fr/s)':<12} | {'Frames':<8} | {'Latency(s)':<10} | {'VRAM(MB)':<10}")
        print("-" * 80)

        fps_list = []
        peak_vram_mb = 0.0

        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(DEVICE)

            model = MoViNet(config, causal=is_stream, pretrained=True).to(DEVICE).eval()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Warmup
            with torch.no_grad():
                if is_stream:
                    if hasattr(model, "clean_activation_buffers"):
                        model.clean_activation_buffers()
                    dummy = torch.zeros((1, 3, 1, crop_size, crop_size), device=DEVICE, dtype=torch.float32)
                    for _ in range(WARMUP_ITERS):
                        _ = model(dummy)
                else:
                    dummy = torch.zeros((1, 3, T, crop_size, crop_size), device=DEVICE, dtype=torch.float32)
                    for _ in range(WARMUP_ITERS):
                        _ = model(dummy)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Per-video benchmark
            for vf in video_files:
                path = os.path.join(VIDEO_DIR, vf)

                frames_bgr = read_all_frames(path)
                if len(frames_bgr) == 0:
                    continue

                if is_stream:
                    processed_frames, elapsed = stream_forward(
                        model=model,
                        frames_bgr=frames_bgr,
                        crop_size=crop_size,
                        device=DEVICE,
                        mean_t=mean_t,
                        std_t=std_t,
                        frame_stride=STREAM_FRAME_STRIDE,
                    )
                    fps = (processed_frames / elapsed) if elapsed > 0 else 0.0
                    frames_used = processed_frames
                else:
                    x = build_base_clip_tensor(
                        frames_bgr=frames_bgr,
                        T=T,
                        crop_size=crop_size,
                        device=DEVICE,
                        mean_t=mean_t,
                        std_t=std_t,
                    )
                    if x is None:
                        continue

                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats(DEVICE)

                    start = time.time()
                    with torch.no_grad():
                        _ = model(x)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.time() - start

                    frames_used = T
                    fps = (T / elapsed) if elapsed > 0 else 0.0
                    del x

                if torch.cuda.is_available():
                    cur = torch.cuda.memory_allocated(DEVICE) / (1024 * 1024)
                    peak = torch.cuda.max_memory_allocated(DEVICE) / (1024 * 1024)
                    peak_vram_mb = max(peak_vram_mb, peak)
                else:
                    cur = 0.0

                fps_list.append(fps)
                print(f"{vf[:25]:<25} | {fps:<12.2f} | {frames_used:<8d} | {elapsed:<10.3f} | {cur:<10.1f}")

                del frames_bgr
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            avg_fps = float(np.mean(fps_list)) if fps_list else 0.0
            record = {
                "name": model_name,
                "avg_fps": avg_fps,
                "peak_vram": peak_vram_mb,
                "status": "Success"
            }

            # ✅ store in the right table
            if is_stream:
                final_summary_stream.append(record)
            else:
                final_summary_base.append(record)

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"!! {model_name} failed: {repr(e)}")
            record = {
                "name": model_name,
                "avg_fps": 0.0,
                "peak_vram": 0.0,
                "status": "OOM/Fail"
            }
            if is_stream:
                final_summary_stream.append(record)
            else:
                final_summary_base.append(record)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ✅ Final: two separate summary tables
    print_summary_table("BASE MODELS SUMMARY REPORT", final_summary_base)
    print_summary_table("STREAM MODELS SUMMARY REPORT", final_summary_stream)


if __name__ == "__main__":
    main()
