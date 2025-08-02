import os
import pickle
import numpy as np
from model import HateSpeechDetector

def test_model_loading():
    print("Testing model loading...")
    
    model_path = 'simple_hate_speech_detector.pkl'
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        return False
    
    print(f"Model file found: {model_path}")
    print(f"File size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    
    try:
        detector = HateSpeechDetector()
        detector.load_model(model_path)
        print("Model loaded successfully!")
        
        test_texts = ["Hello world", "I hate you", "This is bad"]
        predictions = detector.predict(test_texts)
        
        print("\nTest predictions:")
        for i, text in enumerate(test_texts):
            class_idx = np.argmax(predictions[i])
            confidence = np.max(predictions[i])
            classes = ['Hate Speech', 'Offensive Language', 'Neither']
            print(f"'{text}' -> {classes[class_idx]} (confidence: {confidence:.3f})")
        
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

if __name__ == "__main__":
    test_model_loading() 