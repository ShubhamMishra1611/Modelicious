import yaml
import importlib
import torch
from pathlib import Path


class model_loading_error(Exception):
    """Custom exception for model loading errors."""
    pass


class ModelLoader:
    def __init__(self, manifest_path: str):
        self.manifest_path = Path(manifest_path)
        self.manifest_data = self._load_manifest()
        print(self.manifest_data)
        self.models = {}  # name -> model instance

    def _load_manifest(self):
        if not self.manifest_path.exists():
            return {"models": []}
        with open(self.manifest_path, 'r') as f:
            return yaml.safe_load(f)

    def save_manifest(self):
        with open(self.manifest_path, 'w') as f:
            yaml.safe_dump(self.manifest_data, f)

    def load_model_by_name(self, name: str):
        for config in self.manifest_data.get("models", []):
            if config['name'] == name:
                return self._load_model(config)
        raise ValueError(f"Model {name} not found in manifest.")

    def register_and_load_model(self, config: dict, persist: bool = False):
        if config["name"] in self.models:
            raise ValueError(f"Model {config['name']} already loaded.")
        model = self._load_model(config)
        self.manifest_data["models"].append(config)
        if persist:
            self.save_manifest()
        return model

    def unload_model(self, name: str):
        if name in self.models:
            del self.models[name]

    def get_loaded_models(self):
        return list(self.models.keys())

    def _load_model(self, config: dict):
        model_type = config['type']
        loader_map = {
            "torchscript": self._load_torchscript,
            "class_pth": self._load_class_pth,
            "class_state_dict": self._load_class_state_dict,
            "predefined": self._load_predefined_model,
        }
        if model_type not in loader_map:
            raise ValueError(f"Unsupported model type: {model_type}")
        model = loader_map[model_type](config)
        self.models[config["name"]] = model
        return model

    def _load_torchscript(self, config):
        path = config['script_path']
        return torch.jit.load(path)

    def _load_class_pth(self, config):
        model = torch.load(config['weights_path'], weights_only=False)
        return model

    def _load_class_state_dict(self, config):
        cls = self._import_class(config['class_path'])
        model = cls()
        state_dict = torch.load(config['state_dict_path'])
        model.load_state_dict(state_dict)
        return model

    def _load_predefined_model(self, config):
        module_path, class_name = config['source'].rsplit('.', 1)
        module = importlib.import_module(module_path)
        model_fn = getattr(module, class_name)
        return model_fn(pretrained=config.get("pretrained", False))

    def _import_class(self, dotted_path):
        module_path, class_name = dotted_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
