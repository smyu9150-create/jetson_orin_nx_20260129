import torch
import cv2
import os
import glob
import sys
import time
import argparse
import urllib.request
import numpy as np
from collections import deque
from datetime import datetime
from torch2trt import TRTModule

# === 설정: 인터랙티브 모드 ===
def get_user_input():
    print("\n" + "="*40)
    print("   MoViNet TensorRT INT8 Benchmark")
    print("="*40)
    print(" 0: MoViNet-A0 (Fastest)")
    print(" 1: MoViNet-A1 (Balanced)")
    print(" 2: MoViNet-A2 (Higher Accuracy)") # A2 Option added
    
    while True:
        choice = input(">> Select Model (0, 1, 2): ").strip()
        if choice in ['0', '1', '2']:
            return f"a{choice}"
        print("❌ 0, 1, 2 중 하나를 입력하세요.")

# === 메인 코드 ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['a0', 'a1', 'a2'], help='모델 선택')
    parser.add_argument('--source', type=str, help='이미지 폴더 경로')
    args = parser.parse_args()

    # 1. 모델 선택
    if args.model and args.source:
        selected_model = args.model
        source_folder = args.source
    else:
        selected_model = get_user_input()
        print("\n" + "="*40)
        print("   분석할 이미지 폴더 경로 입력")
        print("="*40)
        while True:
            source_folder = input(">> 폴더 경로를 입력하세요: ").strip()
            source_folder = source_folder.replace("'", "").replace('"', "")
            if os.path.exists(source_folder): break
            print(f"❌ 폴더를 찾을 수 없습니다.")

    # 2. 설정 변수 (A2 resolution updated to 224)
    if selected_model == 'a0': 
        IMG_SIZE = 172
    elif selected_model == 'a1': 
        IMG_SIZE = 172
    elif selected_model == 'a2': 
        IMG_SIZE = 224
    
    # ★ 중요: convert_int8.py에서 설정한 프레임 수와 동일해야 합니다.
    CLIP_frames = 8  
    
    LABEL_URL = "https://raw.githubusercontent.com/tensorflow/models/master/official/projects/movinet/files/kinetics_600_labels.txt"
    LABEL_FILE = "kinetics_600_labels.txt"

    # ★ TensorRT INT8 엔진 파일 경로
    ENGINE_PATH = f"movinet_{selected_model}_trt_int8.pth"

    if not os.path.exists(ENGINE_PATH):
        print(f"❌ INT8 엔진 파일을 찾을 수 없습니다: {ENGINE_PATH}")
        print(f"👉 먼저 'convert_int8.py'를 실행하여 {selected_model.upper()} 엔진을 생성해주세요.")
        sys.exit()

    # === [Result Path & Timestamp] ===
    RESULT_DIR = os.path.join(os.getcwd(), "result")
    os.makedirs(RESULT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULT_FILE = os.path.join(RESULT_DIR, f"benchmark_trt_INT8_result_{selected_model}_{timestamp}.txt")

    # 3. 이미지 검색
    print(f"\n🔍 '{source_folder}' 이미지 검색 중...")
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(source_folder, '**', ext), recursive=True))
    
    image_files.sort()

    if not image_files:
        print("❌ 이미지를 찾을 수 없습니다.")
        sys.exit()
    print(f"✅ 총 {len(image_files)}장 발견.")

    # 4. 라벨 로드
    if not os.path.exists(LABEL_FILE):
        urllib.request.urlretrieve(LABEL_URL, LABEL_FILE)
    with open(LABEL_FILE, "r") as f:
        labels = [line.strip() for line in f.readlines()]

    # 5. TensorRT 모델 로드
    print(f"🔄 Loading INT8 Engine: {ENGINE_PATH} ...")
    try:
        model_trt = TRTModule()
        model_trt.load_state_dict(torch.load(ENGINE_PATH))
        model_trt = model_trt.cuda()
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        sys.exit()

    print(f"📊 모델 타입: MoViNet-{selected_model.upper()} TensorRT Engine (INT8)")

    # 6. 추론 및 FPS 측정
    print(f"🚀 벤치마크 시작... (결과는 '{RESULT_FILE}'에 저장)")
    
    frame_buffer = deque(maxlen=CLIP_frames)

    # GPU 워밍업
    print("🔥 GPU 워밍업 중...")
    dummy_input = torch.randn(1, 3, CLIP_frames, IMG_SIZE, IMG_SIZE).cuda()
    for _ in range(10):
        _ = model_trt(dummy_input)
    torch.cuda.synchronize()

    results_buffer = []
    start_time = time.time()
    
    with torch.no_grad():
        for i, img_path in enumerate(image_files):
            frame = cv2.imread(img_path)
            if frame is None: continue

            img_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            input_tensor = torch.from_numpy(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)).float() / 255.0
            input_tensor = input_tensor.permute(2, 0, 1)
            
            frame_buffer.append(input_tensor)

            # 버퍼가 부족할 경우 첫 프레임으로 채움
            while len(frame_buffer) < CLIP_frames:
                frame_buffer.append(input_tensor)

            input_batch = torch.stack(list(frame_buffer), dim=1).unsqueeze(0).cuda()

            # 추론
            prediction = model_trt(input_batch)
            
            probs = torch.nn.functional.softmax(prediction[0], dim=0)
            top_prob, top_class = torch.topk(probs, 1)

            action = labels[top_class.item()]
            score = top_prob.item() * 100
            
            if i % 10 == 0:
                current_fps = (i + 1) / (time.time() - start_time)
                print(f"[{i}/{len(image_files)}] Processing... {current_fps:.1f} FPS", end='\r')

            folder_name = os.path.basename(os.path.dirname(img_path))
            file_name = os.path.basename(img_path)
            results_buffer.append(f"{folder_name:<20} | {file_name:<30} | {action:<25} | {score:.1f}%")

    torch.cuda.synchronize()
    end_time = time.time()

    # 7. 성능 지표 계산
    total_time = end_time - start_time
    fps = len(image_files) / total_time

    print(f"\n✅ 완료! 총 소요시간: {total_time:.2f}초")
    print(f"⚡ 평균 FPS: {fps:.2f} frames/sec")

    # 8. 결과 파일 저장
    with open(RESULT_FILE, "w", encoding='utf-8') as f:
        f.write(f"=== MoViNet TensorRT INT8 Benchmark Report ===\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Model: MoViNet-{selected_model.upper()} (INT8 Quantized)\n")
        f.write(f"Input Shape: (1, 3, {CLIP_frames}, {IMG_SIZE}, {IMG_SIZE})\n")
        f.write(f"Total Images: {len(image_files)}\n")
        f.write(f"Average FPS: {fps:.2f}\n")
        f.write("="*90 + "\n")
        f.write(f"{'Folder':<20} | {'File':<30} | {'Prediction':<25} | {'Score'}\n")
        f.write("-" * 90 + "\n")
        for line in results_buffer:
            f.write(line + "\n")

    print(f"📄 리포트 저장 완료: {RESULT_FILE}")