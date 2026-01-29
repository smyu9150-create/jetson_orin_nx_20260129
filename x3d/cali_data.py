import torch
import os
import shutil
import sys

# ==========================================
# [설정] X3D-M 전용 스펙
# ==========================================
# 저장할 폴더 이름
SAVE_FOLDER = './temp_calibration_tensors'

# 생성할 샘플 개수 (Calibration에는 보통 50~100개면 충분합니다)
NUM_SAMPLES = 64

# X3D-M 입력 형태: (Batch, Channel, Time, Height, Width)
# Batch=1로 저장해야 나중에 유연하게 쓸 수 있습니다.
TARGET_SHAPE = (1, 3, 16, 256, 256) 
# ==========================================

def generate_dummy_data():
    print(f"="*60)
    print(f"[Generator] Starting Dummy Data Generation for X3D-M")
    print(f"Target Shape: {TARGET_SHAPE}")
    print(f"Save Folder : {SAVE_FOLDER}")
    print(f"="*60)

    # 1. 기존 폴더 초기화 (가장 중요: 이전 데이터 삭제)
    if os.path.exists(SAVE_FOLDER):
        print(f"Cleaning existing folder: {SAVE_FOLDER}...")
        shutil.rmtree(SAVE_FOLDER)
    
    os.makedirs(SAVE_FOLDER)
    print(f"Created new folder: {SAVE_FOLDER}")

    # 2. 데이터 생성 및 저장
    print(f"Generating {NUM_SAMPLES} samples...")
    
    for i in range(NUM_SAMPLES):
        # 랜덤 텐서 생성 (정규분포)
        # 실제 데이터와 비슷한 분포를 원하면 전처리된 이미지를 써야 하지만,
        # 엔진 빌드 테스트용으로는 randn으로도 충분합니다.
        tensor = torch.randn(*TARGET_SHAPE)
        
        file_path = os.path.join(SAVE_FOLDER, f"calib_batch_{i:04d}.pt")
        torch.save(tensor, file_path)
        
        if (i + 1) % 10 == 0:
            print(f"  Saved {i + 1}/{NUM_SAMPLES} files...")

    # 3. 검증
    print(f"="*60)
    files = os.listdir(SAVE_FOLDER)
    print(f"Successfully generated {len(files)} files.")
    
    # 첫 번째 파일 로드해서 확인
    test_tensor = torch.load(os.path.join(SAVE_FOLDER, files[0]))
    print(f"[Check] Saved Tensor Shape: {test_tensor.shape}")
    
    if test_tensor.shape == TARGET_SHAPE:
        print("✅ Data shape is CORRECT for x3d_m.")
    else:
        print("❌ Data shape is INCORRECT.")

if __name__ == "__main__":
    generate_dummy_data()