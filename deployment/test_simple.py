from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({
        "message": "Hate Speech Detection API",
        "version": "1.0.0",
        "status": "Basic test mode - model not loaded",
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
    return jsonify({
        "status": "healthy",
        "model_loaded": False,
        "mode": "test",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict_single():
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Text field is required"}), 400
        
        text = data['text']
        if not text or not isinstance(text, str):
            return jsonify({"error": "Text must be a non-empty string"}), 400
        
        return jsonify({
            "text": text,
            "predicted_class": "Neither",
            "confidence": 0.5,
            "mode": "test",
            "probabilities": {
                "hate_speech": 0.1,
                "offensive_language": 0.2,
                "neither": 0.7
            }
        })
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict-batch', methods=['POST'])
def predict_batch():
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
        
        results = []
        for text in texts:
            result = {
                "text": text,
                "predicted_class": "Neither",
                "confidence": 0.5,
                "mode": "test",
                "probabilities": {
                    "hate_speech": 0.1,
                    "offensive_language": 0.2,
                    "neither": 0.7
                }
            }
            results.append(result)
        
        return jsonify({
            "results": results,
            "total_processed": len(results),
            "mode": "test"
        })
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False) 