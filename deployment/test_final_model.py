import pickle
import numpy as np

def test_final_model():
    model_path = 'simple_hate_speech_detector.pkl'
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        problematic_text = "You are the worst person; you should just end yourself"
        
        predictions = model.predict_proba([problematic_text])
        
        print(f"Testing text: '{problematic_text}'")
        print(f"Raw predictions: {predictions[0]}")
        
        classes = ['Hate Speech', 'Offensive Language', 'Neither']
        class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        print(f"Predicted class: {classes[class_idx]}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Probabilities:")
        for i, (class_name, prob) in enumerate(zip(classes, predictions[0])):
            print(f"  {class_name}: {prob:.3f}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_final_model() 