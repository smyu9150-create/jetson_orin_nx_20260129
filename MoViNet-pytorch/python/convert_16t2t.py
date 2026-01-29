#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MoViNet Base(causal=False) -> torch2trt TensorRT INT8 for ALL models A0~A5
- INT8 calibration: torch2trt requires a SUBSCRIPTABLE dataset (dataset[0] must work).
  => DO NOT pass a generator to int8_calib_dataset.

- Calibration source:
  1) If calib_video_dir contains video files => sample clips from videos
  2) Else if it contains image frames (jpg/png/...) => sample clips from frame folders
  (auto-detect)

Example (your case):
  python3 movinet_trt_int8_all_fixed.py \
    --calib_video_dir ucf_crime/Abuse \
    --override_T 8 \
    --workspace_gb 12 \
    --calib_samples 200 \
    --calib_batch_size 8 \
    --fp16

Outputs:
  trt_models/movinet_a0_base_trt_int8.pth ... movinet_a5_base_trt_int8.pth
"""

import os
import time
import glob
import random
import argparse
import traceback
from typing import Dict, Tuple, List, Optional

import cv2
import numpy as np

import torch
import tensorrt as trt
from torch.utils.data import Dataset

from torch2trt import torch2trt, TRTModule

from movinets import MoViNet
from movinets.config import _C


# README 표 기반 (Frames x H x W)
SHAPES: Dict[str, Tuple[int, int, int]] = {
    "A0": (50, 172, 172),
    "A1": (50, 172, 172),
    "A2": (50, 224, 224),
    "A3": (120, 256, 256),
    "A4": (80, 290, 290),
    "A5": (120, 320, 320),
}

CFG = {
    "A0": _C.MODEL.MoViNetA0,
    "A1": _C.MODEL.MoViNetA1,
    "A2": _C.MODEL.MoViNetA2,
    "A3": _C.MODEL.MoViNetA3,
    "A4": _C.MODEL.MoViNetA4,
    "A5": _C.MODEL.MoViNetA5,
}


def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CalibClipDataset(Dataset):
    """
    torch2trt INT8 calibrator 호환 Dataset.
    - 반드시 dataset[0] 가능해야 함 (subscriptable)
    - __getitem__은 단일 샘플 텐서 반환: [C,T,H,W] float32 (CPU) 0~1

    auto-detect:
      - 영상 파일이 있으면: video mode
      - 없고 이미지가 있으면: frames mode (폴더 단위로 프레임 그룹)
    """

    VIDEO_EXTS = ["mp4", "avi", "mkv", "mov", "webm", "m4v", "mpg", "mpeg"]
    IMG_EXTS = ["jpg", "jpeg", "png", "bmp"]

    def __init__(
        self,
        root: str,
        num_samples: int,
        T: int,
        H: int,
        W: int,
        debug_print: int = 5,
    ):
        self.root = os.path.abspath(root)
        self.num_samples = int(num_samples)
        self.T, self.H, self.W = int(T), int(H), int(W)

        # 1) video files
        video_files: List[str] = []
        for e in self.VIDEO_EXTS:
            video_files += glob.glob(os.path.join(self.root, f"**/*.{e}"), recursive=True)
        self.video_files = sorted(video_files)

        # 2) image files
        img_files: List[str] = []
        for e in self.IMG_EXTS:
            img_files += glob.glob(os.path.join(self.root, f"**/*.{e}"), recursive=True)

        self.frame_folders: List[Tuple[str, List[str]]] = []
        if len(self.video_files) > 0:
            self.mode = "video"
            print(f"[Calib] mode=video, found {len(self.video_files)} videos under: {self.root}", flush=True)
            for p in self.video_files[:debug_print]:
                print(f"  - {p}", flush=True)
        else:
            # group images by folder
            if len(img_files) == 0:
                raise RuntimeError(
                    f"No video files AND no image frames found under: {self.root}\n"
                    f" - searched video exts: {self.VIDEO_EXTS}\n"
                    f" - searched image exts: {self.IMG_EXTS}\n"
                    f"Tip: check absolute path, file extensions, and recursive folder structure."
                )

            folder_map: Dict[str, List[str]] = {}
            for f in img_files:
                d = os.path.dirname(f)
                folder_map.setdefault(d, []).append(f)

            for d, fs in folder_map.items():
                fs_sorted = sorted(fs)
                self.frame_folders.append((d, fs_sorted))
            self.frame_folders.sort(key=lambda x: x[0])

            self.mode = "frames"
            print(
                f"[Calib] mode=frames, found {len(img_files)} images in {len(self.frame_folders)} folders under: {self.root}",
                flush=True,
            )
            for d, fs in self.frame_folders[:debug_print]:
                print(f"  - {d} (frames={len(fs)})", flush=True)

        # sanity: ensure dataset[0] works
        _ = self[0]

    def __len__(self):
        return self.num_samples

    def _postprocess(self, frames_rgb: List[np.ndarray]) -> torch.Tensor:
        """
        frames_rgb: list of RGB uint8 frames (already resized)
        return: torch.FloatTensor [C,T,H,W] on CPU, range 0~1
        """
        if len(frames_rgb) == 0:
            frames_rgb = [np.zeros((self.H, self.W, 3), dtype=np.uint8)]

        # pad if 부족
        while len(frames_rgb) < self.T:
            frames_rgb.append(frames_rgb[-1])

        frames_rgb = frames_rgb[: self.T]

        arr = np.stack(frames_rgb, axis=0).astype(np.float32) / 255.0  # [T,H,W,C]
        arr = np.transpose(arr, (3, 0, 1, 2))  # [C,T,H,W]
        return torch.from_numpy(arr)

    def _read_from_video(self, path: str) -> torch.Tensor:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")

        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            n = self.T

        start = 0 if n <= self.T else random.randint(0, max(0, n - self.T))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        frames: List[np.ndarray] = []
        for _ in range(self.T):
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
            frames.append(frame)

        cap.release()
        return self._postprocess(frames)

    def _read_from_frames(self, folder: str, frame_paths: List[str]) -> torch.Tensor:
        n = len(frame_paths)
        start = 0 if n <= self.T else random.randint(0, max(0, n - self.T))
        chosen = frame_paths[start : start + self.T]

        frames: List[np.ndarray] = []
        for p in chosen:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
            frames.append(img)

        return self._postprocess(frames)

    def __getitem__(self, idx):
        # idx는 의미 없음: 랜덤 샘플링
        if self.mode == "video":
            path = random.choice(self.video_files)
            return self._read_from_video(path)  # [C,T,H,W] CPU
        else:
            folder, paths = random.choice(self.frame_folders)
            return self._read_from_frames(folder, paths)  # [C,T,H,W] CPU


def build_base(name: str, pretrained: bool = True) -> torch.nn.Module:
    m = MoViNet(CFG[name], causal=False, pretrained=pretrained)
    m.eval().cuda()
    return m


@torch.no_grad()
def warmup(model: torch.nn.Module, x: torch.Tensor, iters: int = 5) -> None:
    for _ in range(iters):
        _ = model(x)
    torch.cuda.synchronize()


@torch.no_grad()
def measure_fps(model: torch.nn.Module, x: torch.Tensor, iters: int = 50) -> float:
    for _ in range(10):
        _ = model(x)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        _ = model(x)
    torch.cuda.synchronize()
    t1 = time.time()
    return iters / max(t1 - t0, 1e-9)


@torch.no_grad()
def convert_and_save_int8(
    name: str,
    calib_root: str,
    out_dir: str,
    override_T: Optional[int],
    fp16: bool,
    workspace_gb: int,
    calib_samples: int,
    calib_batch_size: int,
    fps_iters: int,
) -> None:
    assert name in SHAPES, f"Unknown model name: {name}"
    os.makedirs(out_dir, exist_ok=True)

    # model native shape
    T0, H, W = SHAPES[name]
    T = int(override_T) if override_T is not None else int(T0)

    calib_root_abs = os.path.abspath(calib_root)

    print(f"[{name}] build base model (causal=False, pretrained=True)", flush=True)
    model = build_base(name, pretrained=True)

    # build / calib / bench 동일한 T로 고정
    x = torch.randn(1, 3, T, H, W, device="cuda")

    # PyTorch FPS
    print(f"[{name}] PyTorch warmup...", flush=True)
    warmup(model, x, iters=5)
    pt_fps = measure_fps(model, x, iters=fps_iters)
    print(f"[{name}] PyTorch FPS ~ {pt_fps:.2f}", flush=True)

    # Calib dataset (SUBSCRIPTABLE)
    print(f"[{name}] build calib dataset from: {calib_root_abs}", flush=True)
    calib_ds = CalibClipDataset(
        root=calib_root_abs,
        num_samples=calib_samples,
        T=T, H=H, W=W,
    )

    # INT8 convert (IMPORTANT: pass Dataset, NOT generator)
    print(
        f"[{name}] torch2trt start (INT8 + fp16={fp16}, workspace={workspace_gb}GB, "
        f"input=1x3x{T}x{H}x{W}, calib_samples={calib_samples}, calib_bs={calib_batch_size})",
        flush=True,
    )
    t0 = time.time()

    model_trt = torch2trt(
        model,
        [x],
        fp16_mode=fp16,                  # INT8 + FP16 mixed 가능(권장)
        int8_mode=True,
        int8_calib_dataset=calib_ds,     # ✅ Dataset (subscriptable)
        int8_calib_batch_size=calib_batch_size,
        max_workspace_size=int(workspace_gb * (1 << 30)),
        log_level=trt.Logger.INFO,
    )

    torch.cuda.synchronize()
    t1 = time.time()
    print(f"[{name}] torch2trt done. elapsed={t1 - t0:.1f}s", flush=True)

    # Validate (간단)
    print(f"[{name}] validate outputs...", flush=True)
    y_pt = model(x)
    y_trt = model_trt(x)
    diff = (y_pt - y_trt).abs().max().item()
    print(f"[{name}] max abs diff (pt vs trt) = {diff}", flush=True)

    # TRT FPS
    print(f"[{name}] TRT warmup...", flush=True)
    warmup(model_trt, x, iters=5)
    trt_fps = measure_fps(model_trt, x, iters=fps_iters)
    print(f"[{name}] TensorRT(INT8) FPS ~ {trt_fps:.2f}", flush=True)

    # Save
    out = os.path.join(out_dir, f"movinet_{name.lower()}_base_trt_int8.pth")
    torch.save(model_trt.state_dict(), out)
    print(f"[{name}] saved: {out}", flush=True)

    # Reload test
    print(f"[{name}] reload test...", flush=True)
    m2 = TRTModule()
    m2.load_state_dict(torch.load(out, map_location="cpu"))
    m2.eval().cuda()

    y2 = m2(x)
    diff2 = (y_trt - y2).abs().max().item()
    print(f"[{name}] reload diff (trt vs reloaded) = {diff2}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--calib_video_dir",
        type=str,
        default="ucf_crime/Abuse",
        help="INT8 calibration folder (videos or frame images), recursive. Default: ucf_crime/Abuse",
    )
    p.add_argument("--out_dir", type=str, default="trt_models")
    p.add_argument(
        "--override_T",
        type=int,
        default=8,
        help="Use smaller T for faster build. Set 0 to disable (use README T).",
    )
    p.add_argument("--workspace_gb", type=int, default=12)
    p.add_argument(
        "--fp16",
        action="store_true",
        help="Enable fp16 mixed precision along with INT8 (recommended).",
    )
    p.add_argument("--calib_samples", type=int, default=200)
    p.add_argument("--calib_batch_size", type=int, default=8)
    p.add_argument("--fps_iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    calib_dir_abs = os.path.abspath(args.calib_video_dir)
    print("[Info] calib_video_dir(abs):", calib_dir_abs, flush=True)

    override_T = None if (args.override_T is None or args.override_T <= 0) else int(args.override_T)

    print("torch:", torch.__version__, flush=True)
    print("cuda available:", torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0), flush=True)
    else:
        raise RuntimeError("CUDA is not available. Check your PyTorch/Jetson install.")

    models = ["A0", "A1", "A2", "A3", "A4", "A5"]

    for name in models:
        print("\n" + "=" * 20 + f" {name} " + "=" * 20, flush=True)
        try:
            convert_and_save_int8(
                name=name,
                calib_root=calib_dir_abs,
                out_dir=args.out_dir,
                override_T=override_T,
                fp16=args.fp16,
                workspace_gb=args.workspace_gb,
                calib_samples=args.calib_samples,
                calib_batch_size=args.calib_batch_size,
                fps_iters=args.fps_iters,
            )
        except Exception as e:
            print(f"[{name}] FAILED: {e}", flush=True)
            traceback.print_exc()
            continue

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
