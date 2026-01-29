import torch
import torch_tensorrt  # 이게 핵심: tensorrt.Engine 타입 등록

model = torch.jit.load("x3d_xs_trt_fp16_nhwc.ts", map_location="cpu")

# 파라미터 수 (대부분 0일 수 있음: 엔진에 bake됨)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# 참고: forward 한 번 호출 가능 여부 확인(입력 shape은 네 모델에 맞게)
# x = torch.randn(1, 3, 13, 182, 182)  # 예시
# y = model(x)
# print(y.shape)
