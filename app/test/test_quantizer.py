from ..services import quantizer

import torch
import torch.nn as nn
import os

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def test_dynamic_quantization():
    model = TinyMLP()
    path = "quantized_model.pth"
    quantizer.apply_dynamic_quantization(model, path)
    assert os.path.exists(path)

def test_static_quantization():
    class QuantModel(TinyMLP):
        def fuse_model(self): pass  # Override for compatibility

    model = QuantModel()
    example_input = torch.randn(1, 10)
    path = "static_quant_model.pth"
    quantizer.apply_static_quantization(model, example_input, path)
    assert os.path.exists(path)