import torch
from torch2trt import torch2trt
from movinets.models import MoViNet
from movinets.config import _C

# ==========================================
# 1. 설정 (MoViNet A5에 맞춰 수정됨)
# ==========================================
# 변환할 모델 종류: 'a5' 설정
MODEL_VARIANT = 'a5' 

# 입력 크기 설정 (MoViNet 공식 입력 해상도)
# A0: 172, A1: 172, A2: 224, A3: 256, A4: 290, A5: 320
IMG_SIZE = 320 
NUM_FRAMES = 8   # 한 번에 추론할 프레임 수

print(f"[{MODEL_VARIANT}] 모델 준비 중... 해상도: {IMG_SIZE}x{IMG_SIZE}, 프레임: {NUM_FRAMES}")

# ==========================================
# 2. MoViNet 모델 로드
# ==========================================
if MODEL_VARIANT == 'a0':
    config = _C.MODEL.MoViNetA0
elif MODEL_VARIANT == 'a1':
    config = _C.MODEL.MoViNetA1
elif MODEL_VARIANT == 'a2':
    config = _C.MODEL.MoViNetA2
elif MODEL_VARIANT == 'a3':
    config = _C.MODEL.MoViNetA3
elif MODEL_VARIANT == 'a4':
    config = _C.MODEL.MoViNetA4
elif MODEL_VARIANT == 'a5':
    # A5 설정 추가
    config = _C.MODEL.MoViNetA5

# causal=False: 일반적인 비디오 클립 분류 모드
model = MoViNet(config, causal=False, pretrained=True)

# GPU로 이동 및 평가 모드
model.cuda().eval()

# ==========================================
# 3. 더미 데이터 생성 (B, C, Frames, H, W)
# ==========================================
# 배치 크기는 1로 고정
x = torch.ones((1, 3, NUM_FRAMES, IMG_SIZE, IMG_SIZE)).cuda()

# ==========================================
# 4. TensorRT 변환 (핵심)
# ==========================================
print("TensorRT 변환 시작... (A5는 모델이 커서 시간이 더 소요됩니다)")

# A5는 모델이 크므로 workspace 사이즈를 넉넉하게 잡는 것이 좋습니다.
# 기존 1<<25 (32MB) -> 1<<30 (1GB)로 상향 추천
model_trt = torch2trt(model, [x], fp16_mode=True, max_workspace_size=1<<30)

# ==========================================
# 5. 결과 저장
# ==========================================
save_name = f'movinet_{MODEL_VARIANT}_trt_fp16.pth'
torch.save(model_trt.state_dict(), save_name)

print(f"\n성공! 변환된 모델이 저장되었습니다: {save_name}")
print(f"입력 데이터 형태: {x.shape}")