import torch
import time
from movinets import MoViNet
from movinets.config import _C

def benchmark_movinets():
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running benchmark on: {device}\n")

    # Define model configurations based on the README table
    # Format: (Config Object, Name, Frames, Resolution)
    # Note: Input shape is (Batch, Channels, Frames, Height, Width)
    models_to_test = [
        (_C.MODEL.MoViNetA0, "MoViNet-A0-Base", 50, 172),
        (_C.MODEL.MoViNetA1, "MoViNet-A1-Base", 50, 172),
        (_C.MODEL.MoViNetA2, "MoViNet-A2-Base", 50, 224),
        (_C.MODEL.MoViNetA3, "MoViNet-A3-Base", 120, 256),
        (_C.MODEL.MoViNetA4, "MoViNet-A4-Base", 80, 290),
        (_C.MODEL.MoViNetA5, "MoViNet-A5-Base", 120, 320),
    ]

    print(f"{'Model':<20} | {'Input Shape (T,H,W)':<20} | {'Inference Time':<15} | {'FPS':<10}")
    print("-" * 75)

    for config, name, frames, resolution in models_to_test:
        try:
            # 1. Initialize Model
            # causal=False loads the Base model (standard 3D convolutions)
            model = MoViNet(config, causal=False, pretrained=False)
            model.to(device)
            model.eval()

            # 2. Create Dummy Input: (Batch=1, Channels=3, Frames, Height, Width)
            input_tensor = torch.randn(1, 3, frames, resolution, resolution).to(device)

            # 3. Warmup (crucial for CUDA to initialize buffers)
            with torch.no_grad():
                for _ in range(5):
                    _ = model(input_tensor)
            
            # 4. Benchmark Loop
            num_runs = 20
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = model(input_tensor)
            
            # Synchronize CUDA for accurate timing
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()

            # 5. Calculate Metrics
            total_time = end_time - start_time
            avg_time = total_time / num_runs
            fps = 1 / avg_time

            shape_str = f"{frames}x{resolution}x{resolution}"
            print(f"{name:<20} | {shape_str:<20} | {avg_time:.4f} sec      | {fps:.2f}")

            # Clean up memory
            del model
            del input_tensor
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{name:<20} | SKIPPED (OOM) - GPU memory insufficient for this size.")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            else:
                print(f"{name:<20} | ERROR: {e}")

if __name__ == "__main__":
    benchmark_movinets()
