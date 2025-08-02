import pickle
import numpy as np

def test_sklearn_model():
    model_path = 'simple_hate_speech_detector.pkl'
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"Model type: {type(model)}")
        print(f"Has predict_proba: {hasattr(model, 'predict_proba')}")
        
        test_texts = [
            "Hello world",
            "I hate you",
            "You are the worst person; you should just end yourself",
            "This is bad",
            "You are stupid and worthless"
        ]
        
        predictions = model.predict_proba(test_texts)
        
        print("\nTest predictions:")
        classes = ['Hate Speech', 'Offensive Language', 'Neither']
        for i, text in enumerate(test_texts):
            class_idx = np.argmax(predictions[i])
            confidence = np.max(predictions[i])
            print(f"'{text}' -> {classes[class_idx]} (confidence: {confidence:.3f})")
            print(f"  Probabilities: {predictions[i]}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_sklearn_model() 