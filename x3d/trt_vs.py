import torch
import torch.nn.functional as F
import torchvision
import time
import os
import glob
import torch_tensorrt  # [필수] TensorRT 엔진 로딩용

# ==========================================
# 설정 (Settings)
# ==========================================
VIDEO_DIR = "/home/etri/data/video"
BATCH_SIZE = 8       # 메모리가 부족하면 4로 줄이세요
NUM_FRAMES = 13      # X3D-S 모델 입력 프레임 수
INPUT_SIZE = 182     # 입력 해상도
WARMUP_ROUNDS = 5    
TEST_ROUNDS = 50     
# ==========================================

def load_base_model(device):
    print("\n[1] Loading Basic X3D-S model (FP32)...")
    try:
        from torchvision.models.video import x3d_s
        # torchvision 버전에 따라 weights 매개변수 처리
        try:
            model = x3d_s(weights='KINETICS400_V1')
        except:
            model = x3d_s(pretrained=True)
    except:
        # torchvision 로드 실패 시 torch.hub 사용
        model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', 
                               pretrained=True, verbose=False)
    
    return model.to(device).eval()

def preprocess_video(video_path, device):
    """
    영상 읽기 및 차원/타입 교정 함수
    """
    # 1. 영상 읽기
    try:
        # 기본적으로 (Time, Height, Width, Channel) 순서로 읽힘
        vframes, _, _ = torchvision.io.read_video(video_path, pts_unit='sec')
    except Exception as e:
        print(f"Skipping broken video {video_path}: {e}")
        return None

    # 2. [중요] 차원 순서 교정 (400 vs 3 에러 방지)
    # (T, H, W, C) -> (C, T, H, W)로 변경해야 함
    if vframes.shape[-1] == 3:  # 끝이 Channel(3)이면
        vframes = vframes.permute(3, 0, 1, 2)
    elif vframes.shape[1] == 3: # 혹시 (T, C, H, W)라면
        vframes = vframes.permute(1, 0, 2, 3)

    # 3. 프레임 샘플링 (13프레임 맞추기)
    total_frames = vframes.shape[1]
    if total_frames > NUM_FRAMES:
        indices = torch.linspace(0, total_frames - 1, NUM_FRAMES).long()
        vframes = vframes[:, indices, :, :]
    else:
        # 프레임 부족 시 반복해서 채움
        repeats = NUM_FRAMES // total_frames + 1
        vframes = vframes.repeat(1, repeats, 1, 1)[:, :NUM_FRAMES, :, :]

    # 4. 리사이즈 (182x182)
    vframes = vframes.to(device).float() / 255.0
    vframes = F.interpolate(vframes, size=(INPUT_SIZE, INPUT_SIZE), 
                            mode='bilinear', align_corners=False)
    
    # 5. 정규화
    mean = torch.tensor([0.45, 0.45, 0.45], device=device).view(3, 1, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.225], device=device).view(3, 1, 1, 1)
    vframes = (vframes - mean) / std

    return vframes

def prepare_batch(video_dir, batch_size, device):
    """배치 데이터 생성 (FP32 기본)"""
    print(f"\n[Data] Preparing batch of size {batch_size}...")
    
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mkv', '*.mov']:
        video_files.extend(glob.glob(os.path.join(video_dir, ext)))
    
    if not video_files:
        print(f"Error: {video_dir} 경로에 비디오가 없습니다.")
        return None

    batch_tensors = []
    for i in range(batch_size):
        file_path = video_files[i % len(video_files)]
        tensor = preprocess_video(file_path, device)
        if tensor is not None:
            batch_tensors.append(tensor)
    
    if not batch_tensors:
        print("Error: 전처리 가능한 비디오가 없습니다.")
        return None

    # (B, C, T, H, W) 형태로 합치기
    return torch.stack(batch_tensors)

def benchmark(model, input_tensor, name):
    print(f"[{name}] Warming up...")
    with torch.no_grad():
        for _ in range(WARMUP_ROUNDS):
            model(input_tensor)
    torch.cuda.synchronize()

    print(f"[{name}] Benchmarking ({TEST_ROUNDS} iters)...")
    start = time.time()
    with torch.no_grad():
        for _ in range(TEST_ROUNDS):
            model(input_tensor)
    torch.cuda.synchronize()
    end = time.time()

    fps = (input_tensor.shape[0] * TEST_ROUNDS) / (end - start)
    print(f"  -> FPS: {fps:.2f}")
    return fps

def main():
    if not torch.cuda.is_available():
        print("CUDA GPU Required.")
        return
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # 1. 데이터 준비 (기본 FP32)
    input_batch = prepare_batch(VIDEO_DIR, BATCH_SIZE, device)
    if input_batch is None: return

    # [핵심 수정] TRT 모델용 FP16 입력 별도 생성
    # TensorRT 엔진이 FP16으로 빌드되었으므로 입력도 반드시 .half()여야 함
    input_batch_fp16 = input_batch.clone().half()

    # 2. 모델 로드
    base_model = load_base_model(device)
    
    trt_path = 'x3d_s_trt_fp16.ts'
    if not os.path.exists(trt_path):
        print(f"Error: {trt_path} 파일이 없습니다.")
        return
    
    print(f"\n[2] Loading Optimized Model ({trt_path})...")
    trt_model = torch.jit.load(trt_path).to(device)

    print("\n=== Start Benchmark ===")
    
    # 3. 속도 측정
    # (1) Base Model -> FP32 입력 사용
    fps_base = benchmark(base_model, input_batch, "Basic X3D-S (FP32)")

    # (2) TRT Model -> [중요] FP16 입력 사용
    try:
        fps_trt = benchmark(trt_model, input_batch_fp16, "TRT X3D-S (FP16)")
    except RuntimeError as e:
        print(f"\n[Error] TRT 벤치마크 중 에러 발생: {e}")
        print("입력 텐서 타입이 맞지 않을 수 있습니다. (FP16 check)")
        return

    # 4. 결과 출력
    print("\n=== Final Results ===")
    print(f"Basic Model FPS : {fps_base:.2f}")
    print(f"TRT Model FPS   : {fps_trt:.2f}")
    print(f"Speedup         : {fps_trt/fps_base:.2f}x")

if __name__ == "__main__":
    main()