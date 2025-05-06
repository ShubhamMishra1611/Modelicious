import torch
from services import pruner
from test_quantizer import TinyMLP

def test_magnitude_pruning():
    model = TinyMLP()
    pruned_model = pruner.apply_magnitude_pruning(model, 0.5)
    for name, module in pruned_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            assert hasattr(module, 'weight_mask')  # Confirm pruning applied
