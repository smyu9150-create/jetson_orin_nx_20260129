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

import tensorrt as trt

LABEL_URL = "https://raw.githubusercontent.com/tensorflow/models/master/official/projects/movinet/files/kinetics_600_labels.txt"
LABEL_FILE = "kinetics_600_labels.txt"


def get_user_input():
    print("\n" + "=" * 40)
    print("   MoViNet TensorRT Benchmark")
    print("=" * 40)
    print(" 0: MoViNet-A0 (Fastest)")
    print(" 1: MoViNet-A1 (Balanced)")
    print(" 2: MoViNet-A2 (Most Accurate)")

    while True:
        choice = input(">> Select Model (0, 1, 2): ").strip()
        if choice in ["0", "1", "2"]:
            return f"a{choice}"
        print("❌ 0, 1, 2 중 하나를 입력하세요.")


def find_images(source_folder: str):``
    image_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        image_files.extend(glob.glob(os.path.join(source_folder, "**", ext), recursive=True))
    image_files.sort()
    return image_files


def load_labels():
    if not os.path.exists(LABEL_FILE):
        print("⬇️ 라벨 파일 다운로드 중...")
        urllib.request.urlretrieve(LABEL_URL, LABEL_FILE)

    with open(LABEL_FILE, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


def preprocess_frame_bgr(frame_bgr, img_size: int):
    # Resize
    img = cv2.resize(frame_bgr, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # float32, 0~1
    img = img.astype(np.float32) / 255.0
    # (H,W,C) -> (C,H,W)
    img = np.transpose(img, (2, 0, 1))
    return img  # numpy float32 (3,H,W)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["a0", "a1", "a2"], help="모델 선택")
    parser.add_argument("--source", type=str, help="이미지 폴더 경로")
    parser.add_argument("--engine", type=str, default=None, help="엔진 경로 직접 지정 (기본: movinet_{model}_trt_fp16.pth)")
    parser.add_argument("--clip_frames", type=int, default=8, help="클립 프레임 수 (엔진 변환 시 값과 동일해야 함)")
    parser.add_argument("--warmup", type=int, default=10, help="GPU 워밍업 반복 횟수")
    parser.add_argument("--log_every", type=int, default=10, help="몇 장마다 진행상황 출력할지")
    parser.add_argument("--max_images", type=int, default=0, help="0이면 전체, 그 외 숫자면 해당 개수까지만 처리")
    parser.add_argument("--no_postprocess", action="store_true", help="softmax/topk/postprocess 생략(속도 측정용)")
    parser.add_argument("--save_scores", action="store_true", help="결과 파일에 예측/score까지 저장 (기본은 저장)")
    args = parser.parse_args()

    # 1) 모델/소스 입력
    if args.model and args.source:
        selected_model = args.model
        source_folder = args.source
    else:
        selected_model = get_user_input()
        print("\n" + "=" * 40)
        print("   분석할 이미지 폴더 경로 입력")
        print("=" * 40)
        while True:
            source_folder = input(">> 폴더 경로를 입력하세요: ").strip()
            source_folder = source_folder.replace("'", "").replace('"', "")
            if os.path.exists(source_folder):
                break
            print("❌ 폴더를 찾을 수 없습니다.")

    # 2) IMG_SIZE (엔진 변환 값과 동일해야 함)
    if selected_model in ["a0", "a1"]:
        img_size = 172
    else:
        img_size = 224

    clip_frames = args.clip_frames

    # 3) 엔진 경로
    engine_path = args.engine if args.engine else f"movinet_{selected_model}_trt_fp16.pth"
    if not os.path.exists(engine_path):
        print(f"❌ 엔진 파일을 찾을 수 없습니다: {engine_path}")
        print("👉 먼저 convert.py로 엔진을 생성하세요.")
        sys.exit(1)

    # 4) 결과 파일 경로
    result_dir = os.path.join(os.getcwd(), "result")
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(result_dir, f"benchmark_trt_{selected_model}_{timestamp}.txt")

    # 5) 이미지 로딩
    print(f"\n🔍 '{source_folder}' 이미지 검색 중...")
    image_files = find_images(source_folder)
    if not image_files:
        print("❌ 이미지를 찾을 수 없습니다.")
        sys.exit(1)

    if args.max_images and args.max_images > 0:
        image_files = image_files[: args.max_images]

    print(f"✅ 총 {len(image_files)}장 발견.")

    # 6) 라벨
    labels = load_labels()

    # 7) TRT 모델 로드
    print(f"🔄 Loading TensorRT Engine: {engine_path} ...")
    try:
        model_trt = TRTModule()
        model_trt.load_state_dict(torch.load(engine_path, weights_only=False))
        model_trt = model_trt.cuda().eval()
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        sys.exit(1)

    print("📊 모델 타입: TensorRT Engine (FP16)")
    print(f"🚀 벤치마크 시작... (결과는 '{result_file}'에 저장)")

    # 8) 워밍업
    print("🔥 GPU 워밍업 중...")
    dummy_input = torch.randn(1, 3, clip_frames, img_size, img_size, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model_trt(dummy_input)
    torch.cuda.synchronize()

    # 9) 버퍼(슬라이딩 윈도우)
    frame_buffer = deque(maxlen=clip_frames)

    # 10) 성능 측정용 (inference-only)
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    infer_ms_list = []

    # 11) 결과 저장용
    results_buffer = []

    # 12) 실시간 FPS 출력용
    start_time = time.time()
    last_print_t = start_time
    last_print_i = 0

    # 13) pinned memory 버퍼 (H2D 복사 약간 개선)
    #    (3,H,W) 형태 numpy를 torch tensor로 바꿔서 CPU에 pin 후 GPU로 non_blocking 전송
    use_post = not args.no_postprocess

    with torch.no_grad():
        for i, img_path in enumerate(image_files):
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            np_chw = preprocess_frame_bgr(frame, img_size)  # (3,H,W) float32
            cpu_tensor = torch.from_numpy(np_chw)  # CPU float32
            # pin memory (가능한 경우)
            cpu_tensor = cpu_tensor.pin_memory()

            # 버퍼에 CPU 텐서 저장 (GPU 메모리 절약)
            frame_buffer.append(cpu_tensor)

            # 초기 패딩
            while len(frame_buffer) < clip_frames:
                frame_buffer.append(cpu_tensor)

            # (clip_frames, 3, H, W) -> (3, clip_frames, H, W)
            # stack dim=0: (T, C, H, W)
            clip_cpu = torch.stack(list(frame_buffer), dim=0)         # (T, C, H, W)
            clip_cpu = clip_cpu.permute(1, 0, 2, 3).contiguous()      # (C, T, H, W)
            input_cpu = clip_cpu.unsqueeze(0)                         # (1, C, T, H, W)

            # H2D (non_blocking)
            input_batch = input_cpu.to(device="cuda", non_blocking=True)

            # ===== Inference-only timing (CUDA event) =====
            starter.record()
            prediction = model_trt(input_batch)
            ender.record()
            torch.cuda.synchronize()
            infer_ms = starter.elapsed_time(ender)
            infer_ms_list.append(infer_ms)

            action = "N/A"
            score = 0.0

            if use_post:
                probs = torch.nn.functional.softmax(prediction[0], dim=0)
                top_prob, top_class = torch.topk(probs, 1)
                idx = top_class.item()
                action = labels[idx] if 0 <= idx < len(labels) else f"class_{idx}"
                score = float(top_prob.item()) * 100.0

            # 로깅
            if args.log_every > 0 and (i % args.log_every == 0) and i > 0:
                now = time.time()
                dt = now - last_print_t
                inst_fps = (i - last_print_i) / dt if dt > 0 else 0.0
                print(f"[{i}/{len(image_files)}] Processing... {inst_fps:.1f} FPS", end="\r")
                last_print_t = now
                last_print_i = i

            # 결과 기록
            folder_name = os.path.basename(os.path.dirname(img_path))
            file_name = os.path.basename(img_path)

            if args.no_postprocess:
                results_buffer.append(f"{folder_name:<20} | {file_name:<30} | {'(postprocess off)':<25} | {'-':>6}")
            else:
                results_buffer.append(f"{folder_name:<20} | {file_name:<30} | {action:<25} | {score:5.1f}%")

    torch.cuda.synchronize()
    end_time = time.time()

    # 14) pipeline FPS (전체)
    total_time = end_time - start_time
    pipeline_fps = len(image_files) / total_time if total_time > 0 else 0.0

    # 15) inference-only FPS
    avg_infer_ms = float(np.mean(infer_ms_list)) if infer_ms_list else 0.0
    infer_fps = (1000.0 / avg_infer_ms) if avg_infer_ms > 0 else 0.0

    print("\n✅ 완료!")
    print(f"🕒 총 소요시간(pipeline): {total_time:.2f}초")
    print(f"⚡ 평균 FPS(pipeline): {pipeline_fps:.2f} frames/sec")
    print(f"⚡ 평균 FPS(inference-only): {infer_fps:.2f} clips/sec (avg {avg_infer_ms:.3f} ms/clip)")

    # 16) 저장
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("=== MoViNet TensorRT Benchmark Report ===\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Model: MoViNet-{selected_model.upper()} (TensorRT FP16 via torch2trt)\n")
        f.write(f"Engine: {engine_path}\n")
        f.write(f"Input Shape: (1, 3, {clip_frames}, {img_size}, {img_size})\n")
        f.write(f"Total Images: {len(image_files)}\n")
        f.write(f"Pipeline Time (sec): {total_time:.4f}\n")
        f.write(f"Average FPS (pipeline): {pipeline_fps:.4f}\n")
        f.write(f"Average Inference (ms/clip): {avg_infer_ms:.4f}\n")
        f.write(f"Average FPS (inference-only): {infer_fps:.4f}\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'Folder':<20} | {'File':<30} | {'Prediction':<25} | {'Score'}\n")
        f.write("-" * 100 + "\n")
        for line in results_buffer:
            f.write(line + "\n")

    print(f"📄 리포트 저장 완료: {result_file}")


if __name__ == "__main__":
    main()
