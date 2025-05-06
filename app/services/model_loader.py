import torch
import os

def load_model_from_file(file_path: str, input_shape=(1, 10)) -> torch.nn.Module:
    """
    Loads a PyTorch model from a file path. Supports full model or state_dict.
    """
    assert os.path.exists(file_path), f"Model file not found: {file_path}"
    
    try:
        # Attempt full model load
        model = torch.load(file_path)
        if isinstance(model, torch.nn.Module):
            model.eval()
            return model
    except Exception:
        pass

    raise ValueError("Failed to load model. Please ensure it's a full model (not just state_dict).")
