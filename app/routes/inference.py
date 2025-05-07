from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
from app.services import model_loader

router = APIRouter()

# Global in-memory model (for demo)
loaded_model = None

class LoadModelRequest(BaseModel):
    model_path: str

class PredictRequest(BaseModel):
    input_tensor: list

@router.post("/load-model")
def load_model(req: LoadModelRequest):
    global loaded_model
    try:
        model = model_loader.load_model_from_file(req.model_path)
        loaded_model = model
        return {"message": "Model loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/predict")
def predict(req: PredictRequest):
    global loaded_model
    if loaded_model is None:
        raise HTTPException(status_code=400, detail="Model not loaded.")
    try:
        input_tensor = torch.tensor([req.input_tensor], dtype=torch.float32)
        with torch.no_grad():
            output = loaded_model(input_tensor)
        return {"output": output.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
