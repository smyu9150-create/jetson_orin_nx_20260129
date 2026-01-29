import torch
import torchvision
import cv2
import time
import numpy as np
import torch.nn.functional as F
import os
import gc
import json
import urllib.request
import csv

# === CONFIGURATION ===
VIDEO_DIR = "/home/etri/data/video"  # 비디오가 저장된 디렉토리
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ITERATIONS = 10  # 벤치마크를 몇 번 반복할지
OUTPUT_CSV = "x3d_benchmark_results.csv"  # 결과를 저장할 CSV 파일명

MODELS_CONFIG = [
    ("X3D-XS", "x3d_xs", 4, 12, 182),
    ("X3D-S", "x3d_xs", 13, 6, 182),
    ("X3D-M", "x3d_m", 16, 5, 256),
]

def load_kinetics_labels():
    json_url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
    json_filename = "kinetics_classnames.json"
    if not os.path.exists(json_filename):
        try: urllib.request.urlretrieve(json_url, json_filename)
        except: return None
    with open(json_filename, "r") as f:
        data = json.load(f)
    return {int(v): k.replace('"', "") for k, v in data.items()}

def process_video(video_path, target_frames, target_size):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
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
    video_tensor = torch.tensor(np.array([frames[i] for i in indices]), dtype=torch.float32) / 255.0
    video_tensor = video_tensor.permute(3, 0, 1, 2).unsqueeze(0)

    mean = torch.tensor([0.45, 0.45, 0.45], device=DEVICE).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.225], device=DEVICE).view(1, 3, 1, 1, 1)
    return (video_tensor.to(DEVICE) - mean) / std

def main():
    video_files = sorted([f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(('.mp4', '.avi', '.mkv'))])

    if not video_files:
        print(f"Error: No videos found in {VIDEO_DIR}")
        return

    labels_map = load_kinetics_labels()
    final_summary = []
    detailed_logs = []

    print(f"\n[X3D Full Batch Benchmark]")
    print(f"Testing {len(video_files)} videos x {ITERATIONS} iterations each.")
    print("-" * 100)
    print(f"{'Model':<10} | {'Video':<30} | {'Avg FPS':<10} | {'Peak MB'}")
    print("-" * 100)

    for model_name, hub_name, frames, sampling, size in MODELS_CONFIG:
        model_fps_list = []
        model_peak_mem_list = []

        try:
            gc.collect()
            torch.cuda.empty_cache()
            model = torch.hub.load('facebookresearch/pytorchvideo', hub_name, pretrained=True).to(DEVICE).eval()

            for video_file in video_files:
                input_tensor = process_video(os.path.join(VIDEO_DIR, video_file), frames, size)
                if input_tensor is None: continue

                # Warm-up
                with torch.no_grad(): _ = model(input_tensor)

                # Benchmark
                latencies = []
                torch.cuda.reset_peak_memory_stats(DEVICE)

                for _ in range(ITERATIONS):
                    start = time.time()
                    with torch.no_grad():
                        output = model(input_tensor)
                    torch.cuda.synchronize()
                    latencies.append(time.time() - start)

                peak_mem = torch.cuda.max_memory_allocated(DEVICE) / (1024 * 1024)
                avg_fps = 1.0 / (sum(latencies) / len(latencies))

                model_fps_list.append(avg_fps)
                model_peak_mem_list.append(peak_mem)

                top1_prob, top1_class = torch.max(output, 1)
                top1_label = labels_map.get(top1_class.item(), "Unknown")

                detailed_logs.append([model_name, video_file, top1_label, round(avg_fps, 2), round(peak_mem, 1)])
                print(f"{model_name:<10} | {video_file[:30]:<30} | {avg_fps:<10.2f} | {peak_mem:<10.1f}")

            if model_fps_list:
                final_summary.append({
                    "name": model_name, 
                    "fps": sum(model_fps_list)/len(model_fps_list), 
                    "peak": max(model_peak_mem_list),
                    "status": "Success"
                })
            del model

        except Exception as e:
            final_summary.append({"name": model_name, "fps": 0, "peak": 0, "status": "Error"})
            print(f"Error on {model_name}: {e}")

    # === Save to CSV ===
    with open(OUTPUT_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Video File', 'Predicted Class', 'Avg FPS', 'Peak VRAM (MB)'])
        writer.writerows(detailed_logs)

    # === Final Report ===
    print("\n" + "="*75)
    print(f"{'BATCH SUMMARY REPORT':^75}")
    print("="*75)
    print(f"{'Model Name':<15} | {'Global Avg FPS':<18} | {'Max Peak VRAM':<15} | {'Status'}")
    print("-" * 75)
    for res in final_summary:
        print(f"{res['name']:<15} | {res['fps']:<18.2f} | {res['peak']:<15.1f} | {res['status']}")
    print("="*75)
    print(f"Detailed results saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
