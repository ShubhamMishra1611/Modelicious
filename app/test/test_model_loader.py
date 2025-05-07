import pytest
from model_loader_service import ModelLoader
import torch

@pytest.fixture(scope="module")
def loader():
    return ModelLoader(manifest_path="manifest.yml")

test_tensor = torch.randn(1, 10)
test_resnet_input_tesnor = torch.randn(1, 3, 224, 224)
# print(test_tensor)
def test_load_torchscript(loader):
    config = {
        "name": "torchscript_test",
        "type": "torchscript",
        "script_path": "models/torchscript_model.pt"
    }
    model = loader.register_and_load_model(config)
    assert model is not None
    assert config["name"] in loader.models
    print(model(test_tensor))  # Ensure the model can process input
    assert model(test_tensor).shape == (1, 2)  # Adjust based on your model's output shape

def test_load_class_pth(loader):
    config = {
        "name": "full_model_test",
        "type": "class_pth",
        "class_path": "dummy_model.DummyModel",
        "weights_path": "models/dummy_model_full.pth"
    }
    model = loader.register_and_load_model(config)
    assert model is not None
    assert config["name"] in loader.models
    print(model(test_tensor))  # Ensure the model can process input
    assert model(test_tensor).shape == (1, 2)

def test_load_state_dict(loader):
    config = {
        "name": "state_dict_test",
        "type": "class_state_dict",
        "class_path": "dummy_model.DummyModel",
        "state_dict_path": "models/dummy_model_sd.pth"
    }
    model = loader.register_and_load_model(config)
    assert model is not None
    assert config["name"] in loader.models
    print(model(test_tensor))  # Ensure the model can process input
    assert model(test_tensor).shape == (1, 2)

def test_load_predefined(loader):
    config = {
        "name": "predefined_test",
        "type": "predefined",
        "source": "torchvision.models.resnet18",
        "pretrained": False
    }
    model = loader.register_and_load_model(config)
    assert model is not None
    assert config["name"] in loader.models
    print(model(test_resnet_input_tesnor))  # Ensure the model can process input
    assert model(test_resnet_input_tesnor).shape == (1, 1000) 

def test_load_torchscript_from_manifest(loader):
    model = loader.load_model_by_name("torchscript_test")
    assert model is not None
    assert "torchscript_test" in loader.models
    print(model(test_tensor))  # Ensure the model can process input
    assert model(test_tensor).shape == (1, 2)

def test_load_class_pth_from_manifest(loader):
    model = loader.load_model_by_name("full_model_test")
    assert model is not None
    assert "full_model_test" in loader.models
    print(model(test_tensor))  # Ensure the model can process input
    assert model(test_tensor).shape == (1, 2)

def test_load_state_dict_from_manifest(loader):
    model = loader.load_model_by_name("state_dict_test")
    assert model is not None
    assert "state_dict_test" in loader.models
    print(model(test_tensor))  # Ensure the model can process input
    assert model(test_tensor).shape == (1, 2)

def test_load_predefined_from_manifest(loader):
    model = loader.load_model_by_name("predefined_test")
    assert model is not None
    assert "predefined_test" in loader.models
    print(model(test_resnet_input_tesnor))  # Ensure the model can process input
    assert model(test_resnet_input_tesnor).shape == (1, 1000)


