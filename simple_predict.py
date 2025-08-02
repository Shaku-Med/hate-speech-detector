import argparse
import sys
from simple_model import SimpleHateSpeechDetector

def interactive_mode(detector):
    print("Interactive Hate Speech Detection Mode")
    print("Type 'quit' to exit")
    print("-" * 50)
    
    while True:
        text = input("\nEnter text to analyze: ").strip()
        
        if text.lower() == 'quit':
            break
        
        if not text:
            continue
        
        try:
            result = detector.predict_single(text)
            print(f"\nText: {result['text']}")
            print(f"Predicted Class: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print("Probabilities:")
            for class_name, prob in result['probabilities'].items():
                print(f"  {class_name}: {prob:.4f}")
        except Exception as e:
            print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Simple Hate Speech Detection Predictor')
    parser.add_argument('--model', type=str, default='simple_hate_speech_detector.pkl',
                       help='Path to the trained model file')
    parser.add_argument('--text', type=str, help='Single text to predict')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    try:
        detector = SimpleHateSpeechDetector()
        detector.load_model(args.model)
        print(f"Model loaded successfully: {args.model}")
        
        if args.interactive:
            interactive_mode(detector)
        elif args.text:
            result = detector.predict_single(args.text)
            print(f"\nText: {result['text']}")
            print(f"Predicted Class: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print("Probabilities:")
            for class_name, prob in result['probabilities'].items():
                print(f"  {class_name}: {prob:.4f}")
        else:
            print("No input specified. Running in interactive mode...")
            interactive_mode(detector)
            
    except FileNotFoundError:
        print(f"Model file not found: {args.model}")
        print("Please train a model first using simple_model.py")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 