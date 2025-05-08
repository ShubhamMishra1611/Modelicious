from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
from app.services.model_loader import ModelLoader
from pathlib import Path

router = APIRouter()

# Global model loader instance
model_loader = None

class LoadModelRequest(BaseModel):
    model_folder: str
    manifest_path: str = "manifest.yml"

class PredictRequest(BaseModel):
    model_name: str
    input_tensor: list

@router.post("/load-model")
def load_model(req: LoadModelRequest):
    global model_loader
    try:
        model_folder = Path(req.model_folder)
        manifest_path = model_folder / req.manifest_path
        
        if not model_folder.exists():
            raise HTTPException(status_code=400, detail=f"Model folder {req.model_folder} does not exist")
        if not manifest_path.exists():
            raise HTTPException(status_code=400, detail=f"Manifest file {manifest_path} does not exist")
            
        model_loader = ModelLoader(str(manifest_path))
        print(f"Loaded model from {model_folder} with manifest {manifest_path}")
        print(model_loader)
        return {
            "message": "Model loader initialized successfully",
            "available_models": model_loader.get_loaded_models()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/predict")
def predict(req: PredictRequest):
    global model_loader
    if model_loader is None:
        raise HTTPException(status_code=400, detail="Model loader not initialized. Please load a model first.")
    
    try:
        model = model_loader.load_model_by_name(req.model_name)
        input_tensor = torch.tensor([req.input_tensor], dtype=torch.float32)
        with torch.no_grad():
            output = model(input_tensor)
        return {"output": output.tolist()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
def list_models():
    global model_loader
    if model_loader is None:
        raise HTTPException(status_code=400, detail="Model loader not initialized. Please load a model first.")
    return {"available_models": model_loader.get_loaded_models()}
