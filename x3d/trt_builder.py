import torch
import torch_tensorrt
import sys
import torchvision.transforms.functional as F

sys.modules["torchvision.transforms.functional_tensor"] = F

def build():
    print("🚀 [Step 1] PyTorch 모델 로드...")
    model_name = 'x3d_s'
    
    # 1. 모델 로드
    model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
    
    # [핵심 수정 1] .to(memory_format=torch.channels_last_3d) 추가!
    # 모델의 가중치 저장 순서를 미리 GPU 친화적으로 바꿉니다.
    model = model.eval().cuda().half().to(memory_format=torch.channels_last_3d)

    print("🛠 [Step 2] Channels Last 최적화 컴파일 시작...")

    # X3D-S 입력 규격
    trt_input_shape = [1, 3, 13, 182, 182]

    # [핵심 수정 2] 입력 데이터 정의 시에도 format 명시 권장
    # (Torch-TensorRT가 알아서 처리하긴 하지만, 모델이 이미 변환되어 있어야 함)
    
    trt_model = torch_tensorrt.compile(
        model,
        inputs=[torch_tensorrt.Input(
            min_shape=trt_input_shape,
            opt_shape=trt_input_shape,
            max_shape=trt_input_shape,
            dtype=torch.half,
            name="input_video",
            # 입력 텐서의 메모리 포맷 힌트 제공
            format=torch.channels_last_3d 
        )],
        enabled_precisions={torch.half},
        truncate_long_and_double=True,
        workspace_size=1 << 30 # 메모리 조금 더 넉넉하게 (1GB)
    )
    
    print("📦 [Step 3] 저장 호환성을 위한 Trace...")
    
    # [핵심 수정 3] 더미 입력도 channels_last_3d로 생성
    dummy_input = torch.randn(trt_input_shape).cuda().half().to(memory_format=torch.channels_last_3d)
    traced_model = torch.jit.trace(trt_model, [dummy_input])

    print("💾 [Step 4] 저장 (x3d_xs_trt_fp16_nhwc.ts)...")
    torch.jit.save(traced_model, "x3d_xs_trt_fp16_nhwc.ts")
    
    print("✅ 최적화 빌드 완료!")

if __name__ == "__main__":
    build()