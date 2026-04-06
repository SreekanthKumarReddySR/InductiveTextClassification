from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.predict import predict_single

app = FastAPI(title="DBpedia Entity Classification API", description="GraphSAGE-based text classification for DBpedia-14 ontology", version="1.0.0")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    class_id: int
    class_name: str
    probabilities: list[float] = None

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        result = predict_single(request.text)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return FileResponse("app/static/index.html")

@app.get("/health")
async def health():
    return {"status": "healthy"}