import sys
import torch
import json
import urllib.request
import torchvision.transforms.functional as F
import os
import glob
import torch_tensorrt
import tensorrt as trt

# ----------------- [긴급 패치] -----------------
sys.modules["torchvision.transforms.functional_tensor"] = F
# ------------------------------------------------

from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

# ==========================================
# [설정]
target_folder = '/home/etri/MoViNet-pytorch/ucf_crime/Abuse'
calibration_batch_size = 8  # 캘리브레이션 배치 크기
calibration_batches = 4
output_filename = "x3d_xs_int8.ts"

# [중요] 기존 캐시 파일 삭제 (이전 실패 데이터 제거)
if os.path.exists("calibration.cache"):
    os.remove("calibration.cache")
# ==========================================

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("Error: TensorRT requires CUDA GPU.")
    sys.exit()

# 1. 모델 로드
model_name = 'x3d_xs'
print(f"Loading {model_name} model...")
model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
model = model.eval().to(device)

# 2. 클래스 이름 로드
json_filename = "kinetics_classnames.json"
if not os.path.exists(json_filename):
    try:
        urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json", json_filename)
    except: pass
with open(json_filename, "r") as f:
    kinetics_classnames = json.load(f)
kinetics_id_to_classname = {v: str(k).replace('"', "") for k, v in kinetics_classnames.items()}

# 3. 데이터셋 준비
video_files = []
for ext in ['*.mp4', '*.avi', '*.mkv']:
    video_files.extend(glob.glob(os.path.join(target_folder, ext)))
print(f"Found {len(video_files)} videos.")
if len(video_files) == 0:
    print("Error: No videos found for calibration.")
    sys.exit()

# =========================================================================
# [핵심 1] Pickle/Copy 우회를 위한 전역 헬퍼
# =========================================================================
global_calibrator_instance = None
def get_global_calibrator():
    return global_calibrator_instance

# =========================================================================
# [핵심 2] Calibrator
# =========================================================================
class CalibrationDataLoader:
    def __init__(self, file_list, batch_size, limit_batches):
        self.file_list = file_list
        self.batch_size = batch_size
        self.limit_batches = limit_batches
        self.current_idx = 0
        self.batches_yielded = 0
        
        self.num_frames = 13
        self.mean = [0.45, 0.45, 0.45]
        self.std = [0.225, 0.225, 0.225]
        self.side_size = 182
        self.crop_size = 182
        self.sampling_rate = 6
        self.clip_duration = (self.num_frames * self.sampling_rate) / 30

        self.transforms = Compose([
            UniformTemporalSubsample(self.num_frames),
            NormalizeVideo(self.mean, self.std),
            ShortSideScale(size=self.side_size),
            CenterCropVideo(crop_size=(self.crop_size, self.crop_size)),
        ])

    def __iter__(self):
        self.current_idx = 0
        self.batches_yielded = 0
        return self

    def __next__(self):
        if self.batches_yielded >= self.limit_batches or self.current_idx >= len(self.file_list):
            raise StopIteration
        
        batch = []
        while len(batch) < self.batch_size and self.current_idx < len(self.file_list):
            try:
                v_path = self.file_list[self.current_idx]
                video = EncodedVideo.from_path(v_path)
                video_data = video.get_clip(start_sec=0, end_sec=self.clip_duration)
                
                vt = video_data["video"]
                vt = self.transforms.transforms[0](vt) 
                vt = vt / 255.0                        
                for t in self.transforms.transforms[1:]:
                    vt = t(vt)
                
                batch.append(vt)
            except Exception:
                pass
            self.current_idx += 1
        
        if not batch: raise StopIteration
        
        # 부족한 배치는 채우지 않고 있는 만큼만 보냄 (Shape mismatch 방지 위해 주의 필요)
        # 여기서는 간단히 TensorRT가 max_shape 범위 내면 처리 가능하므로 그대로 전송
        self.batches_yielded += 1
        return torch.stack(batch).to(device)

class SystemCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_loader, cache_file="calibration.cache"):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.data_loader = data_loader
        self.iter_loader = iter(data_loader)
        self.cache_file = cache_file
        self.current_batch = None 

    def __reduce__(self):
        return (get_global_calibrator, ())

    def get_batch_size(self):
        return self.data_loader.batch_size

    def get_batch(self, names):
        try:
            data = next(self.iter_loader)
            self.current_batch = data
            # [디버깅] 현재 배치 데이터가 들어가는지 확인
            print(f"  [Calibrator] Feeding batch shape: {data.shape}")
            return [int(data.data_ptr())]
        except StopIteration:
            return None

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            return open(self.cache_file, "rb").read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

# =========================================================================
# [핵심 3] 컴파일 (입력 쉐이프 수정됨)
# =========================================================================
print("Preparing Calibration...")
calib_loader = CalibrationDataLoader(video_files, calibration_batch_size, calibration_batches)

# 전역 등록
calibrator = SystemCalibrator(calib_loader)
global_calibrator_instance = calibrator 

print("Starting INT8 Compilation (Backend: Dynamo)...")

# [핵심 수정] inputs의 Batch Size 범위를 Calibration Batch Size까지 확장
inputs = [torch_tensorrt.Input(
    min_shape=(1, 3, 13, 182, 182),
    opt_shape=(calibration_batch_size, 3, 13, 182, 182), # 최적화 타겟
    max_shape=(calibration_batch_size, 3, 13, 182, 182), # 최대 허용 범위 (캘리브레이션 시 필요)
    dtype=torch.float
)]

try:
    # 컴파일
    trt_model = torch_tensorrt.compile(
        model,
        inputs=inputs,
        enabled_precisions={torch.int8},
        calibrator=calibrator,
        truncate_long_and_double=True,
        # debug=True # 필요시 디버그 정보 출력
    )
    print("Compilation Success!")

    print(f"Tracing optimized model for saving...")
    # 저장용 Tracing은 배치 1로 수행해도 됨 (추론 시 배치 1을 쓸 예정이라면)
    dummy_input = torch.randn(1, 3, 13, 182, 182).to(device)
    traced_optimized_model = torch.jit.trace(trt_model, dummy_input)
    
    print(f"Saving INT8 model to {output_filename}...")
    torch.jit.save(traced_optimized_model, output_filename)
    print("Save Complete.")

except Exception as e:
    print(f"\n[Fatal Error] Compilation Failed: {e}")
    # 캐시 파일 문제일 수 있으므로 안내
    print("Tip: If assert serialized_engine fails, try reducing batch size or checking input data.")
    sys.exit()

# =========================================================================
# 검증
# =========================================================================
print("-" * 60)
print(f"Testing loaded model: {output_filename}")
print("-" * 60)

try:
    loaded_model = torch.jit.load(output_filename)
    loaded_model = loaded_model.to(device)
    
    if len(video_files) > 0:
        v_path = video_files[0]
        filename = os.path.basename(v_path)
        
        video = EncodedVideo.from_path(v_path)
        video_data = video.get_clip(start_sec=0, end_sec=calib_loader.clip_duration)
        vt = video_data["video"]
        vt = calib_loader.transforms.transforms[0](vt)
        vt = vt / 255.0
        for t in calib_loader.transforms.transforms[1:]:
            vt = t(vt)
        
        # 추론 시에는 배치 1
        inputs = vt.unsqueeze(0).to(device)

        with torch.no_grad():
            preds = loaded_model(inputs)
        
        if isinstance(preds, (tuple, list)): preds = preds[0]
        
        probs = torch.nn.Softmax(dim=1)(preds)
        pred_label = kinetics_id_to_classname[int(probs.topk(1).indices[0])]
        
        print(f"File: {filename}")
        print(f"Prediction: {pred_label}")
        print("Model works correctly.")

except Exception as e:
    print(f"Inference Test Failed: {e}")