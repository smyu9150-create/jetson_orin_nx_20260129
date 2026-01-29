import torch
import torch_tensorrt
import gc
from movinets import MoViNet
from movinets.config import _C

def convert_and_save_a3_full_res():
    # 1. 설정 (A3 표준 사양: 120 Frames, 256 Resolution)
    device = torch.device("cuda")
    input_shape = (1, 3, 120, 256, 256) 
    save_path = "movinet_a3_fp16_trt_full.ts"

    print(f"--- MoViNet-A3 Full Spec Conversion ---")
    print(f"Target Shape: {input_shape}")

    # 2. 모델 로드 (메모리 절약을 위해 처음부터 half로 로드 시도)
    try:
        model = MoViNet(_C.MODEL.MoViNetA3, causal=False, pretrained=False)
        model.to(device).eval().half()
        
        # 3. Tracing 수행
        dummy_input = torch.randn(*input_shape).to(device).half()
        
        print("Step 1: Tracing model...")
        with torch.no_grad():
            traced_model = torch.jit.trace(model, dummy_input, strict=False)

        # [중요] Tracing 완료 후 원본 모델과 더미 입력을 즉시 제거하여 VRAM 확보
        del model
        del dummy_input
        gc.collect()
        torch.cuda.empty_cache()

        # 4. TensorRT 컴파일
        print("Step 2: Compiling with TensorRT (This may take a while)...")
        inputs = [torch_tensorrt.Input(shape=input_shape, dtype=torch.half)]
        
        trt_model = torch_tensorrt.ts.compile(
            traced_model,
            inputs=inputs,
            enabled_precisions={torch.float16},
            workspace_size=1 << 29, # 512MB
            truncate_long_and_double=True
        )

        # 5. 저장
        torch.jit.save(trt_model, save_path)
        print(f"Step 3: Done. Saved to: {save_path}")

        # 6. 간단 검증 (메모리 재할당)
        test_input = torch.randn(*input_shape).to(device).half()
        with torch.no_grad():
            _ = trt_model(test_input)
        print("Final: Inference test passed.")

    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "NvMap" in str(e):
            print("\n[메모리 부족] A3 풀 사양은 Jetson의 VRAM 한계를 넘을 수 있습니다.")
            print("해결책: 'sudo init 3'로 GUI를 끄고 실행하거나, 프레임 수를 64로 낮춰야 합니다.")
        else:
            print(f"Error: {e}")

if __name__ == "__main__":
    convert_and_save_a3_full_res()