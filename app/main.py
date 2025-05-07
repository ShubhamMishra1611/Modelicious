from fastapi import FastAPI
from app.routes import inference

app = FastAPI(title="Model Server")
app.include_router(inference.router, prefix="/model", tags=["Model Inference"])