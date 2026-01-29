import torch
import cv2
import glob
import os
import random
from torch2trt import torch2trt, DEFAULT_CALIBRATION_ALGORITHM
from movinets.models import MoViNet
from movinets.config import _C

# ==========================================
# 1. 설정
# ==========================================
MODEL_VARIANT = 'a2'


# ==========================================
# 3. 모델 및 데이터셋 준비
# ==========================================
if MODEL_VARIANT == 'a0':
    config = _C.MODEL.MoViNetA0
elif MODEL_VARIANT == 'a1':
    config = _C.MODEL.MoViNetA1
    IMG_SIZE = 172
elif MODEL_VARIANT == 'a2':
    config = _C.MODEL.MoViNetA2
    IMG_SIZE = 224


FRAMES = 8
CALIBRATION_BATCHES = 100  # 보정에 사용할 배치 수 (많을수록 정확도 유지, 변환 시간 증가)
DATA_PATH = "result/images" # ★ 실제 이미지가 있는 폴더 경로로 수정하세요!

# 경로가 없으면 사용자에게 입력을 받도록 함
if not os.path.exists(DATA_PATH):
    print(f"⚠️ 경고: '{DATA_PATH}' 폴더가 없습니다.")
    DATA_PATH = input(">> 학습/테스트용 이미지가 있는 폴더 경로를 입력하세요: ").strip().replace("'", "").replace('"', "")

print(f"[{MODEL_VARIANT}] INT8 변환 준비 중... 해상도: {IMG_SIZE}x{IMG_SIZE}")

# ==========================================
# 2. Calibration Dataset 클래스 정의
# ==========================================
# 실제 데이터를 로드하여 TensorRT에 공급하는 역할
class MoViNetCalibrationDataset:
    def __init__(self, folder_path, img_size, frames, num_batches):
        self.img_files = sorted(glob.glob(os.path.join(folder_path, '**', '*.jpg'), recursive=True) + 
                                glob.glob(os.path.join(folder_path, '**', '*.png'), recursive=True))
        self.img_size = img_size
        self.frames = frames
        self.num_batches = num_batches
        
        if len(self.img_files) < frames:
            raise ValueError(f"이미지가 너무 적습니다. 최소 {frames}장 이상 필요합니다.")
            
        print(f"✅ 총 {len(self.img_files)}장의 이미지를 발견했습니다. 보정 데이터셋 구성 중...")

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        # 랜덤하게 연속된 8장의 이미지를 뽑아서 배치 구성
        start_idx = random.randint(0, len(self.img_files) - self.frames - 1)
        clip = []
        for i in range(self.frames):
            img = cv2.imread(self.img_files[start_idx + i])
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            clip.append(img)
        
        # (Frames, H, W, C) -> (C, Frames, H, W) -> (1, C, Frames, H, W)
        tensor = torch.from_numpy(np.array(clip)).float() / 255.0
        tensor = tensor.permute(3, 0, 1, 2).unsqueeze(0).cuda()
        
        # torch2trt는 리스트 형태로 입력을 받음
        return [tensor]

# numpy 임포트 필요 (상단에 추가 안되어있다면)
import numpy as np



# 모델 로드
model = MoViNet(config, causal=False, pretrained=True).cuda().eval()

# 보정 데이터셋 생성
try:
    calib_dataset = MoViNetCalibrationDataset(DATA_PATH, IMG_SIZE, FRAMES, CALIBRATION_BATCHES)
except Exception as e:
    print(f"❌ 데이터셋 에러: {e}")
    exit()

# 더미 입력 (입력 크기 정의용)
dummy_input = torch.ones((1, 3, FRAMES, IMG_SIZE, IMG_SIZE)).cuda()

# ==========================================
# 4. INT8 변환 실행
# ==========================================
print(f"🚀 TensorRT INT8 변환 시작... (보정 작업으로 인해 시간이 조금 걸립니다)")
print(f" - Calibration Batches: {CALIBRATION_BATCHES}")

model_trt = torch2trt(
    model,
    [dummy_input],
    fp16_mode=True,       # FP16도 켜야 성능 최적화됨
    int8_mode=True,       # ★ INT8 모드 활성화
    int8_calib_dataset=calib_dataset, # 보정 데이터 공급
    int8_calib_algorithm=DEFAULT_CALIBRATION_ALGORITHM,
    max_workspace_size=1<<25
)

# ==========================================
# 5. 저장
# ==========================================
save_name = f'movinet_{MODEL_VARIANT}_trt_int8.pth'
torch.save(model_trt.state_dict(), save_name)

print(f"\n✅ 성공! INT8 모델 저장됨: {save_name}")
