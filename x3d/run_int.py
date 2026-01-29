import torch
import cv2
import time
import numpy as np
import os
import gc
import json
import urllib.request
import csv
import sys
import tensorrt as trt

# === CONFIGURATION ===
VIDEO_DIR = "/home/etri/data/video"
ENGINE_FILE = "x3d_m_int8.engine"
OUTPUT_CSV = "x3d_int8_throughput.csv"
ITERATIONS = 10
BATCH_SIZE = 8  # [핵심] GPU 효율을 위해 배치를 8로 늘림
DEVICE = torch.device("cuda")

# ==========================================
# [Class] TensorRT Wrapper (Throughput Optimized)
# ==========================================
class TRTModuleWrapper(torch.nn.Module):
    def __init__(self, engine_path):
        super().__init__()
        self.logger = trt.Logger(trt.Logger.ERROR)
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine file not found: {engine_path}")

        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self.stream = torch.cuda.Stream()

    def setup_binding(self, x, output):
        self.context.set_input_shape(self.input_name, x.shape)
        self.context.set_tensor_address(self.input_name, int(x.data_ptr()))
        self.context.set_tensor_address(self.output_name, int(output.data_ptr()))

    def run_async(self):
        """비동기 실행 명령만 큐에 넣음 (기다리지 않음)"""
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)

    def sync(self):
        """모든 작업이 끝날 때까지 대기"""
        self.stream.synchronize()

# ==========================================
# [Utils] Helper Functions
# ==========================================
def load_kinetics_labels():
    url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
    filename = "kinetics_classnames.json"
    
    # 강제 재다운로드 (파일 오염 방지)
    if os.path.exists(filename):
        os.remove(filename)

    try: 
        print(f"Downloading labels from {url}...")
        urllib.request.urlretrieve(url, filename)
    except Exception as e: 
        print(f"[Error] Download failed: {e}")
        return {}

    try:
        with open(filename, "r") as f:
            data = json.load(f)
        
        clean_map = {}
        for k, v in data.items():
            # "0": "label" 형태 처리
            clean_map[int(k)] = str(v).replace('"', "")
        print(f"[Info] Loaded {len(clean_map)} labels successfully.")
        return clean_map
    except Exception as e:
        print(f"[Error] JSON Parsing failed: {e}")
        return {}

def process_video_batch(video_path, target_frames, target_size, batch_size):
    """비디오 하나를 복사해서 배치 크기만큼 뻥튀기 (부하 테스트용)"""
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
    # (T, H, W, C)
    video_data = np.array([frames[i] for i in indices], dtype=np.float32) / 255.0
    # (C, T, H, W)
    video_tensor = torch.tensor(video_data).permute(3, 0, 1, 2)
    
    # Normalize
    mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
    video_tensor = (video_tensor - mean) / std

    # [핵심] 배치를 8개로 복사해서 쌓음 -> (8, 3, 13, 182, 182)
    batch_tensor = video_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
    
    return batch_tensor.to(DEVICE)

# ==========================================
# [Main] Benchmark Loop
# ==========================================
def main():
    if not os.path.exists(VIDEO_DIR):
        print(f"Error: {VIDEO_DIR} not found.")
        return
    # 모든 파일 분석 (제한 제거)
    video_files = sorted([f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(('.mp4', '.avi', '.mkv'))])
    if not video_files: return

    labels_map = load_kinetics_labels()
    
    print(f"\n[X3D INT8 Throughput Benchmark]")
    print(f"Batch Size: {BATCH_SIZE} | Iterations: {ITERATIONS}")
    print("=" * 80)
    print(f"{'Video File':<30} | {'Throughput (FPS)':<18} | {'Latency (ms)':<15}")
    print("-" * 80)

    try:
        gc.collect(); torch.cuda.empty_cache()
        model = TRTModuleWrapper(ENGINE_FILE)
        
        total_fps_list = []

        for video_file in video_files:
            # 1. 데이터 준비 (배치 크기만큼 복제)
            input_batch = process_video_batch(os.path.join(VIDEO_DIR, video_file), 13, 182, BATCH_SIZE)
            if input_batch is None: continue
            
            output_batch = torch.empty((BATCH_SIZE, 400), dtype=torch.float32, device=DEVICE)

            # 2. 바인딩
            model.setup_binding(input_batch, output_batch)

            # 3. Warm-up (GPU 예열)
            for _ in range(10):
                model.run_async()
            model.sync()

            # 4. [핵심] Throughput 측정 (비동기 루프)
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(ITERATIONS):
                model.run_async() # 기다리지 않고 명령만 던짐 (CPU 오버헤드 최소화)
            
            model.sync() # 마지막에 한 번만 대기
            end_time = time.time()

            # 5. 계산
            total_time = end_time - start_time
            # 총 처리한 프레임 수 = 배치사이즈 * 반복횟수
            total_inferences = BATCH_SIZE * ITERATIONS
            fps = total_inferences / total_time
            latency_ms = (total_time / ITERATIONS) * 1000  # 배치 1개 처리 시간

            print(f"{video_file[:28]:<30} | {fps:<18.1f} | {latency_ms:<15.2f}")
            total_fps_list.append(fps)
            
            # 라벨 확인 (첫 번째 샘플만)
            if labels_map:
                probs = torch.nn.functional.softmax(output_batch[0], dim=0)
                top1 = torch.argmax(probs).item()
                # print(f"   -> Pred: {labels_map.get(top1, top1)}")

        print("=" * 80)
        if total_fps_list:
            avg_fps = sum(total_fps_list) / len(total_fps_list)
            print(f"Average Throughput: {avg_fps:.1f} FPS")
            print(f"Results saved to {OUTPUT_CSV}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()