import torch
import cv2
import os
import sys
import time
import argparse
import glob
import urllib.request
from datetime import datetime
from movinets import MoViNet
from movinets.config import _C

# === [USER SETTING] Default Path ===
DEFAULT_UCF_PATH = "/home/etri/ucfdata"

# === 1. Interactive Selection for UCF Dataset ===
def select_video_from_ucf(base_path):
    # 1. Check if path exists
    if not os.path.exists(base_path):
        print(f"❌ Error: The path '{base_path}' does not exist.")
        # Fallback to manual input
        return input(">> Please enter full video path: ").strip()

    # 2. List Categories (Subfolders)
    categories = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    
    if not categories:
        print(f"❌ No folders found in {base_path}")
        return input(">> Please enter full video path: ").strip()

    print("\n" + "="*40)
    print(f"   📂 UCF Dataset Found: {base_path}")
    print("="*40)
    for idx, cat in enumerate(categories):
        print(f" {idx}: {cat}")
    
    # 3. Select Category
    while True:
        try:
            c_idx = int(input(f">> Select Category (0-{len(categories)-1}): "))
            if 0 <= c_idx < len(categories):
                selected_category = categories[c_idx]
                break
        except ValueError:
            pass
        print("❌ Invalid selection.")

    # 4. List Videos in Category
    cat_path = os.path.join(base_path, selected_category)
    # Search for mp4, avi, mkv
    videos = sorted(glob.glob(os.path.join(cat_path, "*.mp4")) + 
                    glob.glob(os.path.join(cat_path, "*.avi")) + 
                    glob.glob(os.path.join(cat_path, "*.mkv")))

    if not videos:
        print(f"❌ No videos found in {selected_category}.")
        sys.exit()

    print(f"\n🎥 Videos in '{selected_category}':")
    # Show first 10 videos to avoid clutter
    limit = min(10, len(videos))
    for idx in range(limit):
        print(f" {idx}: {os.path.basename(videos[idx])}")
    if len(videos) > 10:
        print(f" ... and {len(videos)-10} more.")

    # 5. Select Video
    while True:
        try:
            v_idx = int(input(f">> Select Video (0-{len(videos)-1}): "))
            if 0 <= v_idx < len(videos):
                return videos[v_idx]
        except ValueError:
            pass
        print("❌ Invalid selection.")

# === 2. Model Selection ===
def get_model_input():
    print("\n" + "="*40)
    print("   MoViNet Model Selection")
    print("="*40)
    print(" 0: A0 (Fastest, 172px)")
    print(" 1: A1 (Balanced, 172px)")
    print(" 2: A2 (Accurate, 224px)")
    print(" 3: A3 (Heavy, 256px)")
    print(" 4: A4 (Heavier, 290px)")
    print(" 5: A5 (Best, 320px)")
    
    while True:
        choice = input(">> Select Model (0-5): ").strip()
        if choice in ['0', '1', '2', '3', '4', '5']:
            return f"a{choice}"
        print("❌ Invalid choice.")

# === Main Code ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['a0', 'a1', 'a2', 'a3', 'a4', 'a5'])
    parser.add_argument('--video', type=str, help='Path to input video')
    args = parser.parse_args()

    # --- Step 1: Resolve Video Path ---
    if args.video:
        video_path = args.video
    else:
        # Automatically use the hardcoded path logic
        video_path = select_video_from_ucf(DEFAULT_UCF_PATH)

    # --- Step 2: Resolve Model ---
    if args.model:
        selected_model = args.model
    else:
        selected_model = get_model_input()

    # --- Configuration ---
    model_configs = {
        'a0': (_C.MODEL.MoViNetA0, 172),
        'a1': (_C.MODEL.MoViNetA1, 172),
        'a2': (_C.MODEL.MoViNetA2, 224),
        'a3': (_C.MODEL.MoViNetA3, 256),
        'a4': (_C.MODEL.MoViNetA4, 290),
        'a5': (_C.MODEL.MoViNetA5, 320)
    }

    if selected_model not in model_configs:
        print(f"❌ Error: Model config {selected_model} not found.")
        sys.exit()

    config, IMG_SIZE = model_configs[selected_model]
    
    LABEL_URL = "https://raw.githubusercontent.com/tensorflow/models/master/official/projects/movinet/files/kinetics_600_labels.txt"
    LABEL_FILE = "kinetics_600_labels.txt"

    # Save Results
    RESULT_DIR = "results"
    os.makedirs(RESULT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULT_FILE = os.path.join(RESULT_DIR, f"result_{selected_model}_{timestamp}.txt")

    # --- Step 3: Load Resources ---
    # Load Labels
    if not os.path.exists(LABEL_FILE):
        print("⬇️ Downloading labels...")
        urllib.request.urlretrieve(LABEL_URL, LABEL_FILE)
    
    with open(LABEL_FILE, "r") as f:
        labels = [line.strip() for line in f.readlines()]

    # Load Model
    print(f"🔄 Loading MoViNet-{selected_model.upper()}...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        model = MoViNet(config, causal=False, pretrained=True)
        model.eval().to(device)
        model.clean_activation_buffers()
    except Exception as e:
        print(f"❌ Model Load Error: {e}")
        sys.exit()

    # Load Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        sys.exit()
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"✅ Processing: {os.path.basename(video_path)} ({total_frames} frames, {fps:.1f} FPS)")

    # --- Step 4: Run Inference ---
    print(f"🚀 Running... (Press Ctrl+C to stop)")
    
    FRAME_SKIP = 3  # Adjust this: 1=Accurate, 5=Fast
    results_buffer = []
    start_time = time.time()
    frame_idx = 0
    processed_count = 0

    try:
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret: break

                if frame_idx % FRAME_SKIP != 0:
                    frame_idx += 1
                    continue

                # Preprocess
                img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                input_tensor = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).float() / 255.0
                input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(1).unsqueeze(0).to(device)

                # Predict
                out = model(input_tensor)
                probs = torch.nn.functional.softmax(out[0], dim=0)
                top_prob, top_class = torch.topk(probs, 1)

                action = labels[top_class.item()]
                score = top_prob.item() * 100
                
                # Log
                t_sec = frame_idx / fps
                log = f"Time: {t_sec:5.1f}s | {action:<25} | {score:5.1f}%"
                results_buffer.append(log)
                
                # Console update
                sys.stdout.write(f"\r[{frame_idx}/{total_frames}] {action} ({score:.1f}%)")
                sys.stdout.flush()

                processed_count += 1
                frame_idx += 1

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    
    cap.release()
    total_time = time.time() - start_time
    
    # --- Step 5: Save Report ---
    with open(RESULT_FILE, "w") as f:
        f.write(f"Video: {video_path}\n")
        f.write(f"Model: {selected_model}\n")
        f.write("-" * 50 + "\n")
        for line in results_buffer:
            f.write(line + "\n")

    print(f"\n\n✅ Done! Saved to: {RESULT_FILE}")
    print(f"⚡ Avg Speed: {processed_count / total_time:.1f} FPS")