import torch
import os
from movinets import MoViNet
from movinets.config import _C

def export_onnx():
    device = torch.device("cpu") # Export는 CPU가 더 안정적입니다.
    model_name = "movinet_a3"
    onnx_path = f"{model_name}.onnx"
    input_shape = (1, 3, 64, 224, 224)

    print(f"-> Loading {model_name}...")
    model = MoViNet(_C.MODEL.MoViNetA3, causal=False, pretrained=True).to(device).eval()
    
    dummy_input = torch.randn(*input_shape).to(device)

    print(f"-> Exporting to {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        do_constant_folding=True
    )
    print(f"✅ ONNX export complete: {onnx_path}")

if __name__ == "__main__":
    export_onnx()