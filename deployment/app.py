from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import os
import pickle
import logging
import uvicorn
from contextlib import asynccontextmanager
from train_ml_model import NaiveBayesHateSpeechDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

detector = None
class_names = ['Neither', 'Offensive Language', 'Hate Speech']

def load_model():
    global detector
    try:
        model_path = 'simple_hate_speech_detector.pkl'
        
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    detector = pickle.load(f)
                logger.info("Naive Bayes model loaded successfully")
                logger.info(f"Model classes: {detector.classes}")
                return True
            except Exception as e:
                logger.error(f"Error loading model from {model_path}: {e}")
                return False
        else:
            logger.error("Model file not found")
            return False
    except Exception as e:
        logger.error(f"Error in model initialization: {e}")
        return False

def get_class_names(model):
    if hasattr(model, 'classes'):
        return [name.replace('_', ' ').title() for name in model.classes]
    return class_names

class TextRequest(BaseModel):
    text: str

class BatchTextRequest(BaseModel):
    texts: list[str]

class PredictionResponse(BaseModel):
    text: str
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]

class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]
    total_processed: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Hate Speech Detection API...")
    success = load_model()
    if success:
        logger.info("✅ Model loaded successfully!")
    else:
        logger.error("❌ Failed to load model!")
    yield
    logger.info("Shutting down Hate Speech Detection API...")

app = FastAPI(title="Hate Speech Detection API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def home():
    return {
        "message": "Hate Speech Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": {
                "method": "POST",
                "description": "Predict hate speech for a single text",
                "body": {"text": "your text here"}
            },
            "/predict-batch": {
                "method": "POST", 
                "description": "Predict hate speech for multiple texts",
                "body": {"texts": ["text1", "text2", "text3"]}
            },
            "/health": {
                "method": "GET",
                "description": "Health check endpoint"
            }
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    model_loaded = detector is not None
    return HealthResponse(
        status="healthy" if model_loaded else "model not loaded",
        model_loaded=model_loaded,
        timestamp=str(pd.Timestamp.now())
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: TextRequest):
    if detector is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        text = request.text
        if not text or not isinstance(text, str):
            raise HTTPException(status_code=400, detail="Text must be a non-empty string")
        
        predictions = detector.predict_proba([text])
        model_class_names = get_class_names(detector)
        
        # Handle the Naive Bayes model's output
        probs = predictions[0]
        class_idx = probs.index(max(probs))
        confidence = max(probs)
        
        # Create probabilities dict based on model classes
        prob_dict = {}
        for i, class_name in enumerate(detector.classes):
            prob_dict[class_name] = float(probs[i])
        
        result = PredictionResponse(
            text=text,
            predicted_class=model_class_names[class_idx],
            confidence=float(confidence),
            probabilities=prob_dict
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchTextRequest):
    if detector is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        texts = request.texts
        if not isinstance(texts, list) or len(texts) == 0:
            raise HTTPException(status_code=400, detail="Texts must be a non-empty list")
        
        for i, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise HTTPException(status_code=400, detail=f"Text at index {i} must be a non-empty string")
        
        predictions = detector.predict_proba(texts)
        model_class_names = get_class_names(detector)
        results = []
        
        for i, text in enumerate(texts):
            probs = predictions[i]
            class_idx = probs.index(max(probs))
            confidence = max(probs)
            
            # Create probabilities dict based on model classes
            prob_dict = {}
            for j, class_name in enumerate(detector.classes):
                prob_dict[class_name] = float(probs[j])
            
            result = PredictionResponse(
                text=text,
                predicted_class=model_class_names[class_idx],
                confidence=float(confidence),
                probabilities=prob_dict
            )
            results.append(result)
        
        return BatchPredictionResponse(
            results=results,
            total_processed=len(results)
        )
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 