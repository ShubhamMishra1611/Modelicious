import torch
import torch.nn.utils.prune as prune


def apply_magnitude_prune(model: torch.nn.Module, amount: float = 0.3):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
    return model

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

    model = TinyMLP()
    pruned_model = apply_magnitude_prune(model, amount=0.5)
    print(pruned_model)  # Check the pruned model structure
    for name, module in pruned_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            print(f"Layer {name} has weight mask: {module.weight_mask}")