import numpy as np
import pandas as pd
from model import HateSpeechDetector
import argparse
import sys

class HateSpeechPredictor:
    def __init__(self, model_path):
        self.detector = HateSpeechDetector()
        self.detector.load_model(model_path)
        self.class_names = ['Hate Speech', 'Offensive Language', 'Neither']
        
    def predict_single(self, text):
        predictions = self.detector.predict([text])
        class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        return {
            'text': text,
            'predicted_class': self.class_names[class_idx],
            'confidence': confidence,
            'probabilities': {
                'hate_speech': float(predictions[0][0]),
                'offensive_language': float(predictions[0][1]),
                'neither': float(predictions[0][2])
            }
        }
    
    def predict_batch(self, texts):
        predictions = self.detector.predict(texts)
        results = []
        
        for i, text in enumerate(texts):
            class_idx = np.argmax(predictions[i])
            confidence = np.max(predictions[i])
            
            result = {
                'text': text,
                'predicted_class': self.class_names[class_idx],
                'confidence': confidence,
                'probabilities': {
                    'hate_speech': float(predictions[i][0]),
                    'offensive_language': float(predictions[i][1]),
                    'neither': float(predictions[i][2])
                }
            }
            results.append(result)
        
        return results
    
    def predict_from_file(self, file_path, output_path=None):
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                if 'text' in df.columns:
                    texts = df['text'].values
                elif 'tweet' in df.columns:
                    texts = df['tweet'].values
                else:
                    raise ValueError("CSV file must contain 'text' or 'tweet' column")
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts = [line.strip() for line in f if line.strip()]
            
            results = self.predict_batch(texts)
            
            if output_path:
                results_df = pd.DataFrame(results)
                results_df.to_csv(output_path, index=False)
                print(f"Results saved to {output_path}")
            
            return results
            
        except Exception as e:
            print(f"Error reading file: {e}")
            return None

def print_prediction_result(result):
    print(f"\nText: {result['text']}")
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("Probabilities:")
    for class_name, prob in result['probabilities'].items():
        print(f"  {class_name}: {prob:.4f}")

def interactive_mode(predictor):
    print("Interactive Hate Speech Detection Mode")
    print("Type 'quit' to exit")
    print("-" * 50)
    
    while True:
        text = input("\nEnter text to analyze: ").strip()
        
        if text.lower() == 'quit':
            break
        
        if not text:
            continue
        
        result = predictor.predict_single(text)
        print_prediction_result(result)

def main():
    parser = argparse.ArgumentParser(description='Hate Speech Detection Predictor')
    parser.add_argument('--model', type=str, default='hate_speech_detector_lstm.h5',
                       help='Path to the trained model file')
    parser.add_argument('--text', type=str, help='Single text to predict')
    parser.add_argument('--file', type=str, help='Path to file containing texts (CSV or TXT)')
    parser.add_argument('--output', type=str, help='Output file path for batch predictions')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    try:
        predictor = HateSpeechPredictor(args.model)
        print(f"Model loaded successfully: {args.model}")
        
        if args.interactive:
            interactive_mode(predictor)
        elif args.text:
            result = predictor.predict_single(args.text)
            print_prediction_result(result)
        elif args.file:
            results = predictor.predict_from_file(args.file, args.output)
            if results:
                print(f"\nProcessed {len(results)} texts:")
                for result in results[:5]:
                    print_prediction_result(result)
                if len(results) > 5:
                    print(f"... and {len(results) - 5} more results")
        else:
            print("No input specified. Running in interactive mode...")
            interactive_mode(predictor)
            
    except FileNotFoundError:
        print(f"Model file not found: {args.model}")
        print("Please train a model first using train.py")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 