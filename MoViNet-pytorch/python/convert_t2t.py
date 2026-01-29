#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import traceback
from typing import Dict, Tuple

import torch
import tensorrt as trt
from torch2trt import torch2trt, TRTModule

from movinets import MoViNet
from movinets.config import _C


# README 표에 나온 Base 입력 크기들(Frames x H x W)
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


def build_base(name: str, pretrained: bool = True) -> torch.nn.Module:
    """
    Base 모델 = causal=False (stream buffer 미사용)
    """
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
    """
    간단 FPS 측정 (forward만)
    """
    # warmup
    for _ in range(10):
        _ = model(x)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        _ = model(x)
    torch.cuda.synchronize()
    t1 = time.time()

    sec = max(t1 - t0, 1e-9)
    fps = iters / sec
    return fps


@torch.no_grad()
def convert_and_save(
    name: str,
    fp16: bool = True,
    workspace_gb: int = 8,
    out_dir: str = "trt_models",
    # 엔진 빌드 테스트용으로 프레임 수를 줄이고 싶으면 설정 (None이면 README 값 그대로)
    override_T: int | None = None,
    # FPS 측정 반복
    fps_iters: int = 50,
) -> None:
    assert name in SHAPES, f"Unknown model name: {name}"
    os.makedirs(out_dir, exist_ok=True)

    T, H, W = SHAPES[name]
    if override_T is not None:
        T = int(override_T)

    print(f"[{name}] build base model (causal=False, pretrained=True)", flush=True)
    model = build_base(name, pretrained=True)

    # 입력: [B, C, T, H, W]
    x = torch.randn(1, 3, 8, H, W, device="cuda")

    # PyTorch warmup + fps
    print(f"[{name}] PyTorch warmup...", flush=True)
    warmup(model, x, iters=5)
    pt_fps = measure_fps(model, x, iters=fps_iters)
    print(f"[{name}] PyTorch FPS ~ {pt_fps:.2f}", flush=True)

    # Convert
    print(
        f"[{name}] torch2trt start (fp16={fp16}, workspace={workspace_gb}GB, input=1x3x{T}x{H}x{W})",
        flush=True,
    )
    t0 = time.time()

    model_trt = torch2trt(
        model,
        [x],
        fp16_mode=fp16,
        max_workspace_size=int(workspace_gb * (1 << 30)),
        log_level=trt.Logger.INFO,
    )

    torch.cuda.synchronize()
    t1 = time.time()
    print(f"[{name}] torch2trt done. elapsed={t1 - t0:.1f}s", flush=True)

    # Execute + diff 체크
    print(f"[{name}] validate outputs...", flush=True)
    y_pt = model(x)
    y_trt = model_trt(x)
    diff = (y_pt - y_trt).abs().max().item()
    print(f"[{name}] max abs diff (pt vs trt) = {diff}", flush=True)

    # TRT warmup + fps
    print(f"[{name}] TRT warmup...", flush=True)
    warmup(model_trt, x, iters=5)
    trt_fps = measure_fps(model_trt, x, iters=fps_iters)
    print(f"[{name}] TensorRT FPS ~ {trt_fps:.2f}", flush=True)

    # Save
    out = os.path.join(out_dir, f"movinet_{name.lower()}_base_trt.pth")
    torch.save(model_trt.state_dict(), out)
    print(f"[{name}] saved: {out}", flush=True)

    # Load 테스트
    print(f"[{name}] reload test...", flush=True)
    m2 = TRTModule()
    m2.load_state_dict(torch.load(out, map_location="cpu"))
    m2.eval().cuda()

    y2 = m2(x)
    diff2 = (y_trt - y2).abs().max().item()
    print(f"[{name}] reload diff (trt vs reloaded) = {diff2}", flush=True)


def main():
    # ====== 사용자 옵션 ======
    # A0부터 성공 루트 잡고 싶으면 override_T=8 같은 식으로 먼저 테스트한 뒤 None으로 되돌려.
    override_T = 8  # 예: 8, 16, 32, 또는 None
    workspace_gb = 12   # A4/A5면 12~16으로 올려도 됨
    fp16 = True
    out_dir = "trt_models"
    fps_iters = 50
    # ========================

    # 빠르게 디바이스 확인
    print("torch:", torch.__version__, flush=True)
    print("cuda available:", torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0), flush=True)

    for name in ["A0", "A1", "A2", "A3", "A4", "A5"]:
        print("\n" + "=" * 20, name, "=" * 20, flush=True)
        try:
            convert_and_save(
                name,
                fp16=fp16,
                workspace_gb=workspace_gb,
                out_dir=out_dir,
                override_T=override_T,
                fps_iters=fps_iters,
            )
        except Exception as e:
            print(f"[{name}] FAILED: {e}", flush=True)
            traceback.print_exc()
            # 다음 모델로 계속 진행
            continue

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
