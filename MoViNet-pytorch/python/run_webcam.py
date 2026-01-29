import torch
import cv2
import os
import sys
import time
import argparse
import urllib.request  # <--- FIXED: Added missing import
from datetime import datetime
from movinets import MoViNet
from movinets.config import _C

# === Setup: User Input for A0-A5 ===
def get_user_input():
    print("\n" + "="*40)
    print("   MoViNet Performance Benchmark (A0 - A5)")
    print("="*40)
    print(" 0: MoViNet-A0 (Fastest, 172px)")
    print(" 1: MoViNet-A1 (Balanced, 172px)")
    print(" 2: MoViNet-A2 (Accurate, 224px)")
    print(" 3: MoViNet-A3 (Heavy, 256px)")
    print(" 4: MoViNet-A4 (Heavier, 290px)")
    print(" 5: MoViNet-A5 (Most Accurate, 320px)")
    
    while True:
        choice = input(">> Select Model (0-5): ").strip()
        if choice in ['0', '1', '2', '3', '4', '5']:
            return f"a{choice}"
        print("❌ Please enter a number between 0 and 5.")

# === Main Code ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['a0', 'a1', 'a2', 'a3', 'a4', 'a5'], help='Select Model Version')
    parser.add_argument('--source', type=str, help='Path to video file (MP4)')
    args = parser.parse_args()

    # 1. Model Selection
    if args.model and args.source:
        selected_model = args.model
        video_path = args.source
    else:
        selected_model = get_user_input()
        print("\n" + "="*40)
        print("   Input Video File Path")
        print("="*40)
        while True:
            video_path = input(">> Enter video file path: ").strip()
            video_path = video_path.replace("'", "").replace('"', "")
            if os.path.exists(video_path): break
            print(f"❌ Video file not found: {video_path}")

    # === Config Map ===
    model_configs = {
        'a0': (_C.MODEL.MoViNetA0, 172),
        'a1': (_C.MODEL.MoViNetA1, 172),
        'a2': (_C.MODEL.MoViNetA2, 224),
        'a3': (_C.MODEL.MoViNetA3, 256),
        'a4': (_C.MODEL.MoViNetA4, 290),
        'a5': (_C.MODEL.MoViNetA5, 320)
    }

    if selected_model not in model_configs:
        print(f"❌ Error: Model configuration for {selected_model} not found.")
        sys.exit()

    config, IMG_SIZE = model_configs[selected_model]
    
    # Common Setup
    LABEL_URL = "https://raw.githubusercontent.com/tensorflow/models/master/official/projects/movinet/files/kinetics_600_labels.txt"
    LABEL_FILE = "kinetics_600_labels.txt"

    # Result Path setup
    RESULT_DIR = "./result" # Simplified path for testing
    os.makedirs(RESULT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULT_FILE = os.path.join(RESULT_DIR, f"benchmark_result_{selected_model}_{timestamp}.txt")

    # 2. Load Labels
    if not os.path.exists(LABEL_FILE):
        print("⬇️ Downloading labels...")
        urllib.request.urlretrieve(LABEL_URL, LABEL_FILE)
    
    with open(LABEL_FILE, "r") as f:
        labels = [line.strip() for line in f.readlines()]

    # 3. Load Model
    print(f"🔄 Loading MoViNet-{selected_model.upper()} (Input: {IMG_SIZE}x{IMG_SIZE})...")

    try:
        # FIXED: Set causal=True for frame-by-frame streaming
        model = MoViNet(config, causal=True, pretrained=True)
        model.eval().to('cuda')
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit()

    # Calculate Parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Model Params: {total_params / 1e6:.2f} M")

    # 4. Video Processing
    print(f"🚀 Starting Video Inference... (Saving to '{RESULT_FILE}')")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Failed to open video file.")
        sys.exit()

    results_buffer = []
    
    # GPU Warmup
    print("   Running GPU warmup...")
    # FIXED: Use torch.no_grad()
    with torch.no_grad():
        model.clean_activation_buffers() # Reset before warmup
        dummy = torch.randn(1, 3, 1, IMG_SIZE, IMG_SIZE).to('cuda')
        for _ in range(5):
            model(dummy)
        model.clean_activation_buffers() # Reset after warmup
        torch.cuda.synchronize()

    start_time = time.time()
    frame_idx = 0

    # FIXED: Reset buffers ONCE before the actual video starts
    model.clean_activation_buffers()

    with torch.no_grad(): # <--- FIXED: Critical for memory and speed
        while True:
            ret, frame = cap.read()
            if not ret:
                break 

            frame_idx += 1
            
            # Preprocess
            img_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            input_tensor = torch.from_numpy(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)).float() / 255.0
            input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(1).unsqueeze(0).to('cuda')

            # Inference
            try:
                prediction = model(input_tensor)
            except RuntimeError as e:
                print(f"\n❌ OOM Error at frame {frame_idx}: {e}")
                break

            probs = torch.nn.functional.softmax(prediction[0], dim=0)
            top_prob, top_class = torch.topk(probs, 1)

            action = labels[top_class.item()]
            score = top_prob.item() * 100
            
            results_buffer.append(f"Frame {frame_idx:<5} | {action:<25} | {score:.1f}%")

            if frame_idx % 10 == 0:
                print(f"[{frame_idx}] {action} ({score:.1f}%)", end='\r')

            # FIXED: REMOVED clean_activation_buffers() from loop
            # We want the model to remember the history for accurate video prediction.

    torch.cuda.synchronize()
    end_time = time.time()

    # 5. Calc Stats
    total_time = end_time - start_time
    fps = frame_idx / total_time if total_time > 0 else 0

    print(f"\n✅ Completed! Total Frames: {frame_idx}")
    print(f"⚡ Average FPS: {fps:.2f} frames/sec")

    # 6. Save Report
    with open(RESULT_FILE, "w", encoding='utf-8') as f:
        f.write(f"=== MoViNet Benchmark Report ===\n")
        f.write(f"Model: MoViNet-{selected_model.upper()}\n")
        f.write(f"FPS: {fps:.2f}\n")
        f.write("-" * 50 + "\n")
        for line in results_buffer:
            f.write(line + "\n")

    print(f"📄 Report Saved: {RESULT_FILE}")
    cap.release()