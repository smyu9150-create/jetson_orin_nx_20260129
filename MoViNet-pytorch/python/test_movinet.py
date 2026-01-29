import torch
from movinets import MoViNet
from movinets.config import _C

# 1. 모델 설정 (가장 가벼운 a0 버전)
model = MoViNet(_C.MODEL.MoViNetA0, causal=True, pretrained=True)
model.eval()

# 2. GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 3. 더미 데이터 입력 (배치, 채널, 프레임, 높이, 너비)
# MoViNet은 (B, C, T, H, W) 형식을 주로 씁니다.
inputs = torch.randn(1, 3, 50, 172, 172).to(device)

# 4. 추론 실행
with torch.no_grad():
    output = model(inputs)

print(f"출력 크기: {output.shape}")
print(f"사용 디바이스: {device}")

