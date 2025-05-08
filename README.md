# How to ?

To run simply do

    uvicorn app.main:app --reload  


## 1. Load Model and Manifest


```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/model/load-model' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model_folder": "D:/learning/LLMserve/models",
  "manifest_path": "D:/learning/LLMserve/manifest.yml"
}'
```

**Response:**
```json
{
  "message": "Model loader initialized successfully",
  "available_models": []
}
```

---

## 2. Check Available Models

After loading, verify which models are available: (like manifest is loaded but not all the models, only the models loaded in memory will be shown)

```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/model/models' \
  -H 'accept: application/json'
```

**Response (example):**
```json
{
  "available_models": []
}
```

---

## 3. Predict Using a Loaded Model

Once the model (e.g., `full_model_test`) (name was written in the manifest file... look at the example manifest file)is available, make predictions using the following:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/model/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model_name": "full_model_test",
  "input_tensor": [0.1, 0.5, -0.3, 1.2, 0.0, 2.1, -1.0, 0.8, 0.9, -0.2]
}'
```

**Response:**
```json
{
  "output": [
    [
      0.3848043382167816,
      0.03474080562591553
    ]
  ]
}
```

---

## 4. Confirm Loaded Models

Once a model is properly loaded, the `/models` endpoint should reflect it:

```json
{
  "available_models": [
    "full_model_test"
  ]
}
```

---
## Example Manifest.yml

```yaml
models:
  - name: torchscript_test
    type: torchscript
    script_path: D:/learning/LLMserve/models/torchscript_model.pt

  - name: full_model_test
    type: class_pth
    class_path: dummy_model.DummyModel
    weights_path: models/dummy_model_full.pth

  - name: state_dict_test
    type: class_state_dict
    class_path: dummy_model.DummyModel
    state_dict_path: models/dummy_model_sd.pth

  - name: predefined_test
    type: predefined
    source: torchvision.models.resnet18
    pretrained: false

```

---
Rn the code can only handle following way to load the model

- **TorchScript**  
  Loads a scripted model using `torch.jit.load(script_path)`.

- **Class .pth**  
  Loads a fully serialized model object using `torch.load(weights_path)`.

- **Class + State Dict**  
  Dynamically imports the model class, initializes it, and loads weights using `model.load_state_dict()`.

- **Predefined**  
  Loads a predefined model from a module (e.g., `torchvision.models`) using a callable with `pretrained` option.



---
Exporting Module
#### Export Functions

- **`export_torchscript(model, example_input, save_path)`**  
  Traces the model using `torch.jit.trace`.

- **`export_onnx(model, example_input, save_path)`**  
  Converts the model to the ONNX format





