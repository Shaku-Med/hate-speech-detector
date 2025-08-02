import pickle
import os
from train_ml_model import NaiveBayesHateSpeechDetector

def test_model_loading():
    print("=== TESTING MODEL LOADING ===\n")
    
    try:
        model_path = 'simple_hate_speech_detector.pkl'
        
        if not os.path.exists(model_path):
            print(f"❌ Model file {model_path} not found!")
            return False
        
        print(f"✅ Model file found: {model_path}")
        
        with open(model_path, 'rb') as f:
            detector = pickle.load(f)
        
        print(f"✅ Model loaded successfully")
        print(f"Model type: {type(detector)}")
        print(f"Model classes: {detector.classes}")
        
        # Test prediction
        test_text = "You're so beautiful"
        predictions = detector.predict_proba([test_text])
        
        print(f"\n✅ Prediction test successful")
        print(f"Input: '{test_text}'")
        print(f"Output: {predictions[0]}")
        
        class_idx = predictions[0].index(max(predictions[0]))
        confidence = max(predictions[0])
        predicted_class = detector.classes[class_idx]
        
        print(f"Predicted: {predicted_class} (confidence: {confidence:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n🎉 Model loading test PASSED!")
        print("Your FastAPI app should work correctly now.")
    else:
        print("\n💥 Model loading test FAILED!")
        print("Please check the model file and try again.") 