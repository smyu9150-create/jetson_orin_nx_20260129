import sys
import torch
import json
import urllib.request
import torchvision.transforms.functional as F
import os
import glob

# ----------------- [긴급 패치] -----------------
sys.modules["torchvision.transforms.functional_tensor"] = F
# ------------------------------------------------

from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda
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
# [설정] 분석할 폴더 경로를 여기에 입력하세요
target_folder = '/home/etri/MoViNet-pytorch/ucf_crime/Abuse'
# ==========================================

# 1. 모델 로드
model_name = 'x3d_xs' # xs s m l
print(f"Loading {model_name} model...")
model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")
model = model.eval()
model = model.to(device)

# 2. 클래스 이름 로드
json_url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
json_filename = "kinetics_classnames.json"
if not os.path.exists(json_filename):
    try:
        urllib.request.urlretrieve(json_url, json_filename)
    except:
        pass

with open(json_filename, "r") as f:
    kinetics_classnames = json.load(f)

kinetics_id_to_classname = {}
for k, v in kinetics_classnames.items():
    kinetics_id_to_classname[v] = str(k).replace('"', "")

# 3. 전처리 파이프라인 설정
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
model_transform_params = {
    "x3d_xs": {"side_size": 182, "crop_size": 182, "num_frames": 4, "sampling_rate": 12},
    "x3d_xs": {"side_size": 182, "crop_size": 182, "num_frames": 13, "sampling_rate": 6},
    "x3d_m": {"side_size": 256, "crop_size": 256, "num_frames": 16, "sampling_rate": 5},
}
transform_params = model_transform_params[model_name]
transform = ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(transform_params["num_frames"]),
            Lambda(lambda x: x / 255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(size=transform_params["side_size"]),
            CenterCropVideo(crop_size=(transform_params["crop_size"], transform_params["crop_size"])),
        ]
    ),
)
clip_duration = (transform_params["num_frames"] * transform_params["sampling_rate"]) / 30

# 4. 폴더 내 모든 비디오 파일 찾기
# mp4, avi, mkv 확장자를 모두 찾습니다.
video_files = []
for ext in ['*.mp4', '*.avi', '*.mkv']:
    video_files.extend(glob.glob(os.path.join(target_folder, ext)))

print(f"Found {len(video_files)} videos in {target_folder}")

# 5. 반복 추론 실행
print("-" * 60)
print(f"{'Filename':<30} | {'Top 1 Prediction':<25}")
print("-" * 60)

for video_path in video_files:
    filename = os.path.basename(video_path)
    
    try:
        # 비디오 로드 (앞부분 clip_duration 초 만큼만 자름)
        video = EncodedVideo.from_path(video_path)
        video_data = video.get_clip(start_sec=0, end_sec=clip_duration)
        video_data = transform(video_data)

        inputs = video_data["video"]
        inputs = inputs.to(device)

        with torch.no_grad():
            preds = model(inputs[None, ...])

        post_act = torch.nn.Softmax(dim=1)
        preds = post_act(preds)
        pred_classes = preds.topk(k=1).indices[0] # Top 1만 확인
        
        # 결과 매핑
        pred_label = kinetics_id_to_classname[int(pred_classes[0])]
        
        print(f"{filename[:28]:<30} | {pred_label:<25}")

    except Exception as e:
        print(f"{filename[:28]:<30} | Error: {str(e)[:20]}")

print("-" * 60)