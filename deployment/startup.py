import uvicorn
import os
from app import load_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    model_loaded = load_model()
    
    if not model_loaded:
        logger.warning("Starting without pre-trained model")
    
    port = int(os.environ.get('PORT', 8000))
    logger.info(f"Starting FastAPI server on port {port}")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    ) 