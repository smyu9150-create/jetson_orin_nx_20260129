import torch
import cv2
import os
import glob
import sys
import time
import argparse
import urllib.request
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
    # Updated choices to include a3, a4, a5
    parser.add_argument('--model', type=str, choices=['a0', 'a1', 'a2', 'a3', 'a4', 'a5'], help='Select Model Version')
    parser.add_argument('--source', type=str, help='Path to image folder')
    args = parser.parse_args()

    # 1. Model Selection
    if args.model and args.source:
        selected_model = args.model
        source_folder = args.source
    else:
        selected_model = get_user_input()
        print("\n" + "="*40)
        print("   Input Image Folder Path")
        print("="*40)
        while True:
            source_folder = input(">> Enter folder path: ").strip()
            source_folder = source_folder.replace("'", "").replace('"', "")
            if os.path.exists(source_folder): break
            print(f"❌ Folder not found.")

    # === [NEW] Dynamic Config & Resolution Map ===
    # MoViNet requires different input sizes for optimal performance
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
    RESULT_DIR = "/home/etri/MoViNet-pytorch/result"
    os.makedirs(RESULT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULT_FILE = os.path.join(RESULT_DIR, f"benchmark_result_{selected_model}_{timestamp}.txt")

    # 2. Search Images
    print(f"\n🔍 Searching for images in '{source_folder}'...")
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(source_folder, '**', ext), recursive=True))

    if not image_files:
        print("❌ No images found.")
        sys.exit()
    print(f"✅ Found {len(image_files)} images.")

    # 3. Load Labels
    if not os.path.exists(LABEL_FILE):
        urllib.request.urlretrieve(LABEL_URL, LABEL_FILE)
    with open(LABEL_FILE, "r") as f:
        labels = [line.strip() for line in f.readlines()]

    # 4. Load Model
    print(f"🔄 Loading MoViNet-{selected_model.upper()} (Input: {IMG_SIZE}x{IMG_SIZE})...")

    try:
        # Load specific configuration
        model = MoViNet(config, causal=False, pretrained=True)
        model.eval().to('cuda')
        model.clean_activation_buffers()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("💡 Note: Ensure your 'movinets' package version supports A3-A5.")
        sys.exit()

    # Calculate Parameters
    total_params = sum(p.numel() for p in model.parameters())
    params_str = f"{total_params / 1e6:.2f} M"
    print(f"📊 Model Params: {params_str}")

    # 5. Inference & Benchmark
    print(f"🚀 Starting Benchmark... (Saving to '{RESULT_FILE}')")
    
    # GPU Warmup (Critical for A5/A4 to allocate memory)
    print("   Running GPU warmup...")
    dummy = torch.randn(1, 3, 1, IMG_SIZE, IMG_SIZE).to('cuda')
    try:
        for _ in range(5):
            model(dummy)
            model.clean_activation_buffers()
        torch.cuda.synchronize()
    except RuntimeError as e:
        print(f"❌ OOM Error during warmup: {e}")
        print("💡 A5 requires significant VRAM. Try a smaller model or lower resolution.")
        sys.exit()

    results_buffer = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for i, img_path in enumerate(image_files):
            # Read Image
            frame = cv2.imread(img_path)
            if frame is None: continue

            # Preprocess (Resize to model specific IMG_SIZE)
            img_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            input_tensor = torch.from_numpy(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)).float() / 255.0
            input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(1).unsqueeze(0).to('cuda')

            # Inference
            try:
                prediction = model(input_tensor)
            except RuntimeError as e:
                print(f"\n❌ OOM Error at image {i}: {e}")
                break

            probs = torch.nn.functional.softmax(prediction[0], dim=0)
            top_prob, top_class = torch.topk(probs, 1)

            # Record Result
            action = labels[top_class.item()]
            score = top_prob.item() * 100
            
            # Print progress (Less frequent updates for A4/A5 as they are slower)
            if i % 5 == 0:
                print(f"[{i+1}/{len(image_files)}] Processing...", end='\r')

            folder_name = os.path.basename(os.path.dirname(img_path))
            file_name = os.path.basename(img_path)
            results_buffer.append(f"{folder_name:<20} | {file_name:<30} | {action:<25} | {score:.1f}%")
            
            # Important: Clean buffer for causal inference
            model.clean_activation_buffers()

    torch.cuda.synchronize()
    end_time = time.time()

    # 6. Calc Stats
    total_time = end_time - start_time
    processed_count = len(results_buffer)
    fps = processed_count / total_time if total_time > 0 else 0

    print(f"\n✅ Completed! Total Time: {total_time:.2f}s")
    print(f"⚡ Average FPS: {fps:.2f} frames/sec")

    # 7. Save Report
    with open(RESULT_FILE, "w", encoding='utf-8') as f:
        f.write(f"=== MoViNet Benchmark Report ===\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Model: MoViNet-{selected_model.upper()}\n")
        f.write(f"Resolution: {IMG_SIZE}x{IMG_SIZE}\n")
        f.write(f"Parameters: {params_str}\n")
        f.write(f"Total Images: {processed_count}\n")
        f.write(f"Average FPS: {fps:.2f}\n")
        f.write("="*90 + "\n")
        f.write(f"{'Folder':<20} | {'File':<30} | {'Prediction':<25} | {'Score'}\n")
        f.write("-" * 90 + "\n")
        for line in results_buffer:
            f.write(line + "\n")

    print(f"📄 Report Saved: {RESULT_FILE}")