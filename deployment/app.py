from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import os
from model import HateSpeechDetector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

detector = None
class_names = ['Hate Speech', 'Offensive Language', 'Neither']

def load_model():
    global detector
    try:
        detector = HateSpeechDetector()
        model_path = 'simple_hate_speech_detector.pkl'
        
        if os.path.exists(model_path):
            detector.load_model(model_path)
            logger.info("Model loaded successfully")
            return True
        else:
            logger.warning("Model file not found")
            return False
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

@app.route('/')
def home():
    return jsonify({
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
    })

@app.route('/health')
def health():
    model_loaded = detector is not None
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded,
        "timestamp": pd.Timestamp.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict_single():
    if detector is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Text field is required"}), 400
        
        text = data['text']
        if not text or not isinstance(text, str):
            return jsonify({"error": "Text must be a non-empty string"}), 400
        
        predictions = detector.predict([text])
        class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        result = {
            "text": text,
            "predicted_class": class_names[class_idx],
            "confidence": float(confidence),
            "probabilities": {
                "hate_speech": float(predictions[0][0]),
                "offensive_language": float(predictions[0][1]),
                "neither": float(predictions[0][2])
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    if detector is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({"error": "Texts field is required"}), 400
        
        texts = data['texts']
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({"error": "Texts must be a non-empty list"}), 400
        
        for i, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                return jsonify({"error": f"Text at index {i} must be a non-empty string"}), 400
        
        predictions = detector.predict(texts)
        results = []
        
        for i, text in enumerate(texts):
            class_idx = np.argmax(predictions[i])
            confidence = np.max(predictions[i])
            
            result = {
                "text": text,
                "predicted_class": class_names[class_idx],
                "confidence": float(confidence),
                "probabilities": {
                    "hate_speech": float(predictions[i][0]),
                    "offensive_language": float(predictions[i][1]),
                    "neither": float(predictions[i][2])
                }
            }
            results.append(result)
        
        return jsonify({
            "results": results,
            "total_processed": len(results)
        })
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    model_loaded = load_model()
    
    if not model_loaded:
        logger.warning("Starting without pre-trained model")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 