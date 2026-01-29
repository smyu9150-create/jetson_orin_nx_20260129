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
import csv
import sys
import torchvision.transforms.functional as F_vision

# --- [시스템 패치] ---
sys.modules["torchvision.transforms.functional_tensor"] = F_vision

# =========================
# === [설정 구역] =========
# =========================
VIDEO_DIR = "/home/etri/data/video"
OUTPUT_CSV = "x3d_final_comparison_results.csv"
DEVICE = torch.device("cuda")

# [중요] 엔진이 고정 배치 1로 빌드되었으므로 1로 설정합니다.
BATCH_SIZE = 1  
ITERATIONS = 100

MODELS_CONFIG = [
    ("Base-x3d_m", "x3d_m", "hub"),
    ("TRT-FP16", "x3d_m_fp16.engine", "engine"),
    ("TRT-INT8", "x3d_m_int8.engine", "engine"),
]
# =========================

class TRTModuleWrapper(torch.nn.Module):
    def __init__(self, engine_path):
        super().__init__()
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self.stream = torch.cuda.Stream()

    def forward(self, x):
        # 엔진 출력 크기에 맞춰 동적으로 생성 (보통 400 클래스)
        out_shape = self.engine.get_tensor_shape(self.output_name)
        # 만약 고정 배치가 1이라면 out_shape[0]은 1이 됩니다.
        output = torch.empty(tuple(out_shape), dtype=torch.float32, device=DEVICE)
        
        self.context.set_input_shape(self.input_name, x.shape)
        self.context.set_tensor_address(self.input_name, int(x.data_ptr()))
        self.context.set_tensor_address(self.output_name, int(output.data_ptr()))
        
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        return output

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

def process_video_batch(video_path, target_frames=16, target_size=256):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        h, w, _ = frame.shape
        scale = target_size / min(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        th, tw = frame.shape[:2]
        frame = frame[(th-target_size)//2:(th+target_size)//2, (tw-target_size)//2:(tw+target_size)//2]
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) < 1: return None

    indices = np.linspace(0, len(frames) - 1, target_frames).astype(int)
    video_data = np.array([frames[i] for i in indices], dtype=np.float32) / 255.0
    video_tensor = torch.tensor(video_data).permute(3, 0, 1, 2)
    
    mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
    video_tensor = (video_tensor - mean) / std
    # 배치 사이즈 1로 고정
    return video_tensor.unsqueeze(0).to(DEVICE).half()

def main():
    if not os.path.exists(VIDEO_DIR):
        print(f"Error: {VIDEO_DIR} not found."); return

    video_files = sorted([f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(('.mp4', '.avi', '.mkv'))])
    labels_map = load_kinetics_labels()
    final_summary = []
    detailed_logs = []

    print(f"\n[X3D Final Benchmark: Base vs FP16 vs INT8]")
    print(f"Batch Size: {BATCH_SIZE} (Static) | Iterations: {ITERATIONS}")
    print("-" * 150)
    print(f"{'Model':<12} | {'Video Name':<35} | {'FPS':<10} | {'Lat(ms)':<10} | {'VRAM(MB)':<10} | Pred")
    print("-" * 150)

    for name, path, m_type in MODELS_CONFIG:
        try:
            gc.collect(); torch.cuda.empty_cache()
            if m_type == "hub":
                model = torch.hub.load('facebookresearch/pytorchvideo', path, pretrained=True).to(DEVICE).eval().half()
            else:
                if not os.path.exists(path):
                    print(f"Skipping {name}: {path} not found."); continue
                model = TRTModuleWrapper(path)

            fps_list, mem_list = [], []
            for video_file in video_files:
                input_batch = process_video_batch(os.path.join(VIDEO_DIR, video_file))
                if input_batch is None: continue

                with torch.no_grad(): _ = model(input_batch)
                torch.cuda.synchronize()

                torch.cuda.reset_peak_memory_stats(DEVICE)
                start_time = time.time()
                for _ in range(ITERATIONS):
                    with torch.no_grad(): output = model(input_batch)
                torch.cuda.synchronize()
                
                total_time = time.time() - start_time
                avg_fps = (BATCH_SIZE * ITERATIONS) / total_time
                avg_lat = (total_time / ITERATIONS) * 1000
                peak_mem = torch.cuda.max_memory_allocated(DEVICE) / (1024 * 1024)
                
                top1_label = "Unknown"
                if labels_map:
                    try:
                        _, top1_class = torch.max(output[0], 0)
                        top1_label = labels_map.get(int(top1_class.item()), "Unknown")
                    except: pass

                fps_list.append(avg_fps)
                mem_list.append(peak_mem)
                print(f"{name:<12} | {video_file[:35]:<35} | {avg_fps:<10.2f} | {avg_lat:<10.2f} | {peak_mem:<10.1f} | {top1_label}")
                detailed_logs.append([name, video_file, round(avg_fps, 2), round(avg_lat, 2), round(peak_mem, 1)])

            if fps_list: # 에러 방지를 위해 데이터가 있을 때만 추가
                final_summary.append({"name": name, "fps": np.mean(fps_list), "mem": np.max(mem_list)})
            del model
        except Exception as e:
            print(f"Error on {name}: {e}")

    # === 최종 비교 결과 요약표 출력 ===
    if final_summary:
        print("\n" + "="*95)
        print(f"{'X3D FINAL PERFORMANCE COMPARISON SUMMARY':^95}")
        print("-" * 95)
        print(f"{'Model Name':<15} | {'Avg Throughput':<20} | {'Avg Latency':<18} | {'Max VRAM(MB)':<15}")
        print("-" * 95)
        for s in final_summary:
            avg_lat_ms = 1000 / s['fps'] if s['fps'] > 0 else 0
            print(f"{s['name']:<15} | {s['fps']:>10.2f} FPS        | {avg_lat_ms:>10.2f} ms       | {s['mem']:>12.1f} MB")
        print("=" * 95)
    else:
        print("\n[Warning] No benchmark data to summarize.")

if __name__ == "__main__":
    main()