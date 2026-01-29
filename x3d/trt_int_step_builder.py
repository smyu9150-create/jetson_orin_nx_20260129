#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import time
import numpy as np
import torch
import torch.onnx
import tensorrt as trt

# =============================================================================
# [설정]
# =============================================================================
MODEL_NAME = "x3d_m"  # "x3d_xs", "x3d_s", "x3d_m"
TEMP_DATA_FOLDER = "./temp_calibration_tensors"

# ONNX/ENGINE 경로(절대경로로 안전하게)
ONNX_FILE_PATH = os.path.abspath(f"{MODEL_NAME}.onnx")
ENGINE_FILE_PATH = os.path.abspath(f"{MODEL_NAME}_int8.engine")
CALIBRATION_CACHE = os.path.abspath("calibration.cache")

# X3D 입력: (B, C, T, H, W)
# x3d_m: (1, 3, 16, 256, 256)
INPUT_SHAPE = (1, 3, 16, 256, 256)

# Calibration batch size (메모리 안전: 1 권장)
CALIBRATION_BATCH_SIZE = 1

# Max batch for TRT profile
MAX_BATCH = 1

# Workspace (필요 시 조절) - TRT 10에서도 유효
WORKSPACE_MB = 2048

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

print(f"Target Model: {MODEL_NAME}")
print(f"Target Input Shape: {INPUT_SHAPE}")
print(f"ONNX:   {ONNX_FILE_PATH}")
print(f"ENGINE: {ENGINE_FILE_PATH}")
print(f"Device: {device}")

# =============================================================================
# [Step 1] ONNX Export (고정 shape / dynamic_axes 없음)
# =============================================================================
def export_x3d_to_onnx(model_name: str, onnx_path: str, input_shape: tuple, opset: int = 13, overwrite: bool = False):
    print("\n" + "=" * 60)
    print("[Step 1] Checking/Exporting ONNX...")
    print("=" * 60)

    if os.path.exists(onnx_path) and not overwrite:
        print(f"✅ Using existing ONNX: {onnx_path}")
        return

    # ---- Load model (pytorchvideo hub 우선) ----
    try:
        from pytorchvideo.models import hub
        model_loader = getattr(hub, model_name)
        model = model_loader(pretrained=True)
        print("Loaded from pytorchvideo.models.hub")
    except Exception:
        model = torch.hub.load(
            "facebookresearch/pytorchvideo:main",
            model_name,
            pretrained=True,
            force_reload=False,
            trust_repo=True,
        )
        print("Loaded from torch.hub (facebookresearch/pytorchvideo:main)")

    # Export는 CPU에서 하는게 대체로 안정적
    model = model.eval().to("cpu")
    dummy = torch.randn(*input_shape, dtype=torch.float32, device="cpu")

    with torch.no_grad():
        try:
            torch.onnx.export(
                model,
                dummy,
                onnx_path,
                input_names=["input"],
                output_names=["output"],
                opset_version=opset,
                do_constant_folding=True,
                export_params=True,
            )
            print(f"✅ ONNX Export Success: {onnx_path}")
        except Exception as e:
            print(f"❌ ONNX Export Failed: {e}")
            sys.exit(1)

# =============================================================================
# [Step 2] INT8 Calibrator
#   - TEMP_DATA_FOLDER 안에 *.pt 텐서 파일이 있어야 함
#   - 각 파일 텐서는 (1,C,T,H,W) 또는 (C,T,H,W) 형태를 허용
# =============================================================================
class X3DEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_folder: str, batch_size: int, cache_file: str, device: torch.device):
        super().__init__()
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.cache_file = cache_file
        self.device = device

        self.files = sorted(glob.glob(os.path.join(data_folder, "*.pt")))
        self.current_idx = 0

        if not self.files:
            print(f"❌ Error: No calibration data found in: {data_folder}")
            print("   -> 먼저 calibration tensor (*.pt) 생성해줘야 함.")
            sys.exit(1)

        sample = torch.load(self.files[0], map_location="cpu")
        if sample.dim() == 5:
            # (1,C,T,H,W)
            c, t, h, w = sample.shape[1:]
        elif sample.dim() == 4:
            # (C,T,H,W)
            c, t, h, w = sample.shape
        else:
            print(f"❌ Unsupported tensor dim in {self.files[0]}: {sample.shape}")
            sys.exit(1)

        self.final_shape = (batch_size, c, t, h, w)
        self.device_input = torch.empty(self.final_shape, dtype=torch.float32, device=self.device)

        print(f"[Calibrator] Found {len(self.files)} files")
        print(f"[Calibrator] Using final batch shape: {self.final_shape}")

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_idx + self.batch_size > len(self.files):
            return None

        batch_tensors = []
        for i in range(self.batch_size):
            t = torch.load(self.files[self.current_idx + i], map_location="cpu")
            if t.dim() == 5:
                t = t.squeeze(0)  # (1,C,T,H,W)->(C,T,H,W)
            batch_tensors.append(t)

        self.current_idx += self.batch_size

        batch_stack = torch.stack(batch_tensors, dim=0).to(self.device, non_blocking=True)
        self.device_input.copy_(batch_stack)

        print(f"  [Calibrator] Feeding {self.current_idx}/{len(self.files)}")
        return [int(self.device_input.data_ptr())]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            return open(self.cache_file, "rb").read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

# =============================================================================
# [Step 3] Build TensorRT INT8 Engine (TRT10)
# =============================================================================
def build_int8_engine(onnx_path: str, engine_path: str):
    print("\n" + "=" * 60)
    print("[Step 3] Building TensorRT INT8 Engine...")
    print("=" * 60)

    # 캐시 삭제(권장)
    if os.path.exists(CALIBRATION_CACHE):
        os.remove(CALIBRATION_CACHE)

    calibrator = X3DEntropyCalibrator(
        data_folder=TEMP_DATA_FOLDER,
        batch_size=CALIBRATION_BATCH_SIZE,
        cache_file=CALIBRATION_CACHE,
        device=device
    )

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # workspace
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_MB * 1024 * 1024)
    except Exception:
        # 구버전 호환
        if hasattr(config, "max_workspace_size"):
            config.max_workspace_size = WORKSPACE_MB * 1024 * 1024

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("❌ ONNX Parse Failed. Errors:")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            sys.exit(1)

    # INT8 + Calibrator
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = calibrator

    # Optimization profile (Batch만 가변, 나머지 고정)
    profile = builder.create_optimization_profile()
    _, C, T, H, W = INPUT_SHAPE

    min_shape = (1, C, T, H, W)
    opt_shape = (1, C, T, H, W)
    max_shape = (MAX_BATCH, C, T, H, W)

    print(f"[Profile] Min:{min_shape} / Opt:{opt_shape} / Max:{max_shape}")
    profile.set_shape("input", min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    # Calibration profile 지정
    try:
        config.set_calibration_profile(profile)
    except Exception:
        # 일부 버전에서 API가 다를 수 있어 무시
        pass

    print("Building serialized network...")
    plan = builder.build_serialized_network(network, config)
    if not plan:
        print("❌ Build Failed!")
        sys.exit(1)

    with open(engine_path, "wb") as f:
        f.write(plan)

    print(f"✅ Success! Saved engine to: {engine_path}")

# =============================================================================
# [Step 4] Inference Verification (TRT10 execute_async_v3)
# =============================================================================
class TRTWrapper(torch.nn.Module):
    def __init__(self, engine_path: str):
        super().__init__()
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("Failed to deserialize engine")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

    def forward(self, x: torch.Tensor):
        # x shape must be within profile range
        self.context.set_input_shape("input", tuple(x.shape))

        out_dims = self.context.get_tensor_shape("output")
        output = torch.empty(tuple(out_dims), dtype=torch.float32, device=x.device)

        self.context.set_tensor_address("input", int(x.data_ptr()))
        self.context.set_tensor_address("output", int(output.data_ptr()))

        self.context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
        return output

def verify_engine(engine_path: str):
    print("\n" + "=" * 60)
    print("[Step 4] Inference Verification")
    print("=" * 60)

    trt_model = TRTWrapper(engine_path).to(device)
    dummy = torch.randn(*INPUT_SHAPE, device=device, dtype=torch.float32)

    with torch.no_grad():
        out = trt_model(dummy)

    print(f"✅ Output Shape: {tuple(out.shape)}")
    print("✅ Verification Complete.")

# =============================================================================
# MAIN
# =============================================================================
def main():
    # Step 1: ONNX 준비
    export_x3d_to_onnx(
        model_name=MODEL_NAME,
        onnx_path=ONNX_FILE_PATH,
        input_shape=INPUT_SHAPE,
        opset=13,
        overwrite=False,  # 필요시 True
    )

    if not os.path.exists(ONNX_FILE_PATH):
        print(f"❌ ONNX not found after export: {ONNX_FILE_PATH}")
        sys.exit(1)

    # Step 3: 엔진 빌드
    build_int8_engine(ONNX_FILE_PATH, ENGINE_FILE_PATH)

    # Step 4: 검증
    if os.path.exists(ENGINE_FILE_PATH):
        verify_engine(ENGINE_FILE_PATH)
    else:
        print(f"❌ Engine not found: {ENGINE_FILE_PATH}")
        sys.exit(1)

if __name__ == "__main__":
    main()
