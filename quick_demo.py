from simple_model import SimpleHateSpeechDetector

def run_quick_demo():
    print("Hate Speech Detector Quick Demo")
    print("=" * 50)
    
    try:
        detector = SimpleHateSpeechDetector()
        detector.load_model('simple_hate_speech_detector.pkl')
        print("Model loaded successfully!")
    except:
        print("Training new model...")
        from simple_model import train_simple_model
        detector, accuracy, report = train_simple_model()
    
    test_texts = [
        "I love this beautiful day!",
        "This is absolutely terrible and offensive",
        "We should eliminate all people of that race",
        "The weather is nice today",
        "You are such a stupid idiot",
        "Let's go for a walk in the park",
        "Death to all immigrants",
        "This movie was really good",
        "I hate everyone who thinks differently",
        "Have a wonderful day!",
        "Fuck you, you piece of shit",
        "The food at this restaurant is amazing",
        "Kill all the Jews",
        "I'm so happy to see you",
        "You're a worthless human being"
    ]
    
    print("\nTesting with various texts:")
    print("-" * 80)
    
    for i, text in enumerate(test_texts, 1):
        result = detector.predict_single(text)
        print(f"{i:2d}. Text: {text}")
        print(f"    Prediction: {result['predicted_class']}")
        print(f"    Confidence: {result['confidence']:.4f}")
        print(f"    Hate Speech: {result['probabilities']['hate_speech']:.3f}")
        print(f"    Offensive: {result['probabilities']['offensive_language']:.3f}")
        print(f"    Neither: {result['probabilities']['neither']:.3f}")
        print()
    
    print("Demo completed!")
    print("\nTo use interactively, run: python simple_predict.py --interactive")

if __name__ == "__main__":
    run_quick_demo() 