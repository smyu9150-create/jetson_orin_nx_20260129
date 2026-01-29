#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import tensorrt as trt
import cv2
import time
import numpy as np
import os
import gc
import json
import urllib.request
import sys
import traceback

# MoViNet 라이브러리 (환경에 맞게 설치되어 있어야 함)
from movinets import MoViNet
from movinets.config import _C

# =========================
# === [설정 구역] =========
# =========================
VIDEO_DIR = "/home/etri/data/video"  # 실제 비디오 경로 확인 필수
OUTPUT_CSV = "movinet_comparison_results.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [중요] TensorRT 엔진 빌드 시 설정했던 프레임/해상도와 일치해야 합니다.
# 만약 엔진이 16프레임으로 빌드되었다면 64를 넣으면 에러가 납니다.
# 아래 코드를 실행하면 "Engine I/O Check" 로그에서 실제 엔진 스펙을 볼 수 있습니다.
FRAMES = 64        
RESOLUTION = 224   
BATCH_SIZE = 1     
ITERATIONS = 50    

MODELS_CONFIG = [
    ("MoViNet-A3-Base", "base", "torch"),
    ("TRT-FP16", "movinet_a3_fp16.engine", "engine"),
    ("TRT-INT8", "movinet_a3_int8.engine", "engine"),
]
# =========================

# -------------------------
# TensorRT Wrapper (Safe Version)
# -------------------------
class TRTModuleWrapper(torch.nn.Module):
    def __init__(self, engine_path: str):
        super().__init__()
        self.logger = trt.Logger(trt.Logger.WARNING)

        print(f"Loading Engine: {engine_path}")
        try:
            with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
        except Exception as e:
            raise RuntimeError(f"Engine 로드 실패: {e}")

        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        
        # [디버깅] 엔진 입력/출력 텐서 정보 확인
        print(f">>> [{engine_path}] Engine I/O Info:")
        self.input_name = None
        self.output_name = None
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)
            
            print(f"    - {'Input ' if mode == trt.TensorIOMode.INPUT else 'Output'} [{i}]: {name}, Shape={shape}, Dtype={dtype}")
            
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
                self.input_shape = tuple(shape)
                self.input_dtype = dtype
            else:
                self.output_name = name
                self.output_dtype = dtype

    def forward(self, x: torch.Tensor):
        # [핵심 수정 1] 메모리 연속성 강제 (Illegal memory access 방지)
        x = x.contiguous()

        # [핵심 수정 2] 엔진이 기대하는 Dtype으로 변환
        if self.input_dtype == trt.float16 and x.dtype != torch.float16:
            x = x.half()
        elif self.input_dtype == trt.float32 and x.dtype != torch.float32:
            x = x.float()

        # [핵심 수정 3] 입력 Shape 검증 (엔진과 입력이 다르면 경고)
        if self.input_shape:
            # 배치 차원(-1)을 제외하고 비교
            match = True
            for i, dim in enumerate(self.input_shape):
                if dim != -1 and x.shape[i] != dim:
                    match = False
            if not match:
                print(f"\n[CRITICAL WARNING] Input shape mismatch!")
                print(f"  Expected: {self.input_shape}")
                print(f"  Actual:   {x.shape}")
                print(f"  -> 엔진 빌드 시 설정한 해상도/프레임과 파이썬 코드가 다릅니다.\n")

        # 출력 텐서 준비
        out_shape_trt = self.engine.get_tensor_shape(self.output_name)
        # Dynamic shape 대응 (배치 사이즈 1 가정)
        out_shape = tuple([1 if s == -1 else s for s in out_shape_trt])
        
        dtype_torch = torch.float16 if self.output_dtype == trt.float16 else torch.float32
        out = torch.empty(out_shape, dtype=dtype_torch, device=x.device)

        # 주소 바인딩
        self.context.set_input_shape(self.input_name, tuple(x.shape))
        self.context.set_tensor_address(self.input_name, int(x.data_ptr()))
        self.context.set_tensor_address(self.output_name, int(out.data_ptr()))

        # 실행
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        return out

# -------------------------
# 유틸리티 함수
# -------------------------
def load_kinetics_labels():
    url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
    filename = "kinetics_classnames.json"
    if not os.path.exists(filename):
        try: urllib.request.urlretrieve(url, filename)
        except: return None
    with open(filename, "r") as f:
        data = json.load(f)
    
    clean_map = {}
    for k, v in data.items():
        try:
            if isinstance(v, int): clean_map[v] = str(k).replace('"', "")
            else: clean_map[int(k)] = str(v).replace('"', "")
        except: continue
    return clean_map

def process_video_batch(video_path, target_frames=FRAMES, target_size=RESOLUTION):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Resize & Center Crop
        h, w, _ = frame.shape
        scale = target_size / min(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        th, tw = frame.shape[:2]
        frame = frame[(th-target_size)//2:(th+target_size)//2, (tw-target_size)//2:(tw+target_size)//2]
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) < 1: return None

    # Temporal Sampling
    if len(frames) >= target_frames:
        indices = np.linspace(0, len(frames) - 1, target_frames).astype(int)
    else:
        indices = np.arange(len(frames))
        padding = [len(frames)-1] * (target_frames - len(frames))
        indices = np.concatenate([indices, padding])
        
    video_data = np.array([frames[i] for i in indices], dtype=np.float32) / 255.0
    
    # (T, H, W, C) -> (C, T, H, W)
    # [중요] permute 후 메모리가 섞이므로 .contiguous() 필수
    video_tensor = torch.tensor(video_data).permute(3, 0, 1, 2).contiguous()
    
    mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
    video_tensor = (video_tensor - mean) / std
    
    # 배치 차원 추가 및 최종 contiguous 확인
    return video_tensor.unsqueeze(0).contiguous().to(DEVICE).half()

# -------------------------
# 메인 루프
# -------------------------
def main():
    if not os.path.exists(VIDEO_DIR):
        print(f"Error: {VIDEO_DIR} not found.")
        return

    video_files = sorted([f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(('.mp4', '.avi', '.mkv'))])
    # 테스트용으로 10개만 먼저 보고 싶으면 주석 해제
    # video_files = video_files[:10]
    
    labels_map = load_kinetics_labels()
    final_summary = []

    print(f"\n[MoViNet Benchmark: Base vs TRT]")
    print(f"Settings: {FRAMES} Frames, {RESOLUTION}x{RESOLUTION} Resolution")
    print("-" * 140)
    print(f"{'Model':<16} | {'Video Name':<30} | {'FPS':<10} | {'Lat(ms)':<10} | {'VRAM(MB)':<10} | Pred")
    print("-" * 140)

    for name, path, m_type in MODELS_CONFIG:
        model = None
        try:
            # 메모리 정리
            gc.collect()
            torch.cuda.empty_cache()

            # 모델 로드
            if m_type == "torch":
                model = MoViNet(_C.MODEL.MoViNetA3, causal=False, pretrained=False).to(DEVICE).eval().half()
            elif m_type == "engine":
                if not os.path.exists(path):
                    print(f"Skipping {name}: {path} not found.")
                    continue
                model = TRTModuleWrapper(path)
            
            fps_list, mem_list = [], []

            # 비디오 루프
            for video_file in video_files:
                input_batch = process_video_batch(os.path.join(VIDEO_DIR, video_file))
                if input_batch is None: continue

                # Warmup
                try:
                    with torch.no_grad():
                        _ = model(input_batch)
                    torch.cuda.synchronize()
                except Exception as e:
                    print(f"\n[Error during warmup] {name}: {e}")
                    # 웜업 에러 시 해당 모델 스킵
                    break 

                # 측정 시작
                torch.cuda.reset_peak_memory_stats(DEVICE)
                start_time = time.time()
                last_output = None
                
                for _ in range(ITERATIONS):
                    with torch.no_grad():
                        last_output = model(input_batch)
                
                torch.cuda.synchronize()
                
                total_time = time.time() - start_time
                avg_fps = (BATCH_SIZE * ITERATIONS) / total_time
                avg_lat = (total_time / ITERATIONS) * 1000
                peak_mem = torch.cuda.max_memory_allocated(DEVICE) / (1024 * 1024)

                # 라벨 추출
                top1_label = "-"
                if labels_map and last_output is not None:
                    try:
                        if isinstance(last_output, tuple): logits = last_output[0]
                        else: logits = last_output
                        _, top1_class = torch.max(logits.flatten(), 0)
                        top1_label = labels_map.get(int(top1_class.item()), "Unknown")
                    except: pass

                fps_list.append(avg_fps)
                mem_list.append(peak_mem)
                print(f"{name:<16} | {video_file[:30]:<30} | {avg_fps:<10.2f} | {avg_lat:<10.2f} | {peak_mem:<10.1f} | {top1_label}")

            # 결과 집계
            if fps_list:
                final_summary.append({
                    "name": name, 
                    "fps": np.mean(fps_list), 
                    "lat": np.mean(1000 / np.array(fps_list)),
                    "mem": np.max(mem_list)
                })

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n[Critical Error on {name}]")
            traceback.print_exc()

    # 최종 표 출력
    if final_summary:
        print("\n" + "="*95)
        print(f"{'MOVINET PERFORMANCE SUMMARY':^95}")
        print("-" * 95)
        print(f"{'Model Name':<18} | {'Avg Throughput':<20} | {'Avg Latency':<18} | {'Max VRAM(MB)':<15}")
        print("-" * 95)
        for s in final_summary:
            print(f"{s['name']:<18} | {s['fps']:>10.2f} FPS        | {s['lat']:>10.2f} ms       | {s['mem']:>12.1f} MB")
        print("=" * 95)
    else:
        print("\n[Warning] No successful benchmarks recorded.")

if __name__ == "__main__":
    main()