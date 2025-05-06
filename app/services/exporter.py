# app/services/exporter.py
import torch
import os

def export_torchscript(model: torch.nn.Module, example_input: torch.Tensor, save_path: str) -> str:
    model.eval()
    try:
        traced = torch.jit.trace(model, example_input)
        torch.jit.save(traced, save_path)
        return save_path
    except Exception as e:
        raise RuntimeError(f"TorchScript export failed: {e}")

def export_onnx(model: torch.nn.Module, example_input: torch.Tensor, save_path: str) -> str:
    model.eval()
    try:
        torch.onnx.export(
            model,
            example_input,
            save_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output']
        )
        return save_path
    except Exception as e:
        raise RuntimeError(f"ONNX export failed: {e}")


if __name__ == "__main__":
    from model_loader import load_model_from_file
    class tiny_model(torch.nn.Module):
        def __init__(self):
            super(tiny_model, self).__init__()
            self.fc = torch.nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(x)

    model = tiny_model()
    example_input = torch.randn(1, 10)
    save_path = "model.pt"
    onnx_path = "model.onnx"
    try:
        export_torchscript(model, example_input, save_path)
        print(f"Model exported to {save_path}")
    except RuntimeError as e:
        print(e)
    try:
        export_onnx(model, example_input, onnx_path)
        print(f"Model exported to {onnx_path}")
    except RuntimeError as e:
        print(e)