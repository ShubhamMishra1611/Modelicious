import torch
import torch.quantization

import os


def apply_dynamic_quantization(model: torch.nn.Module, save_path: str)->str:
    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    torch.save(quantized_model.state_dict(), save_path)
    return save_path

def apply_static_quantization(model: torch.nn.Module, example_input: torch.Tensor, save_path: str)->str:
    model.eval()
    model.fuse_model() # model has to implement this method
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    model(example_input)
    torch.quantization.convert(model, inplace=True)
    torch.save(model.state_dict(), save_path)
    return save_path


if __name__ == "__main__":
    # Example usage
    class TinyMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(10, 20)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(20, 1)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

        def fuse_model(self):
            torch.quantization.fuse_modules(self, [['fc1', 'relu']], inplace=True)

    model = TinyMLP()
    example_input = torch.randn(1, 10)
    apply_dynamic_quantization(model, "quantized_model.pth")
    apply_static_quantization(model, example_input, "static_quant_model.pth")