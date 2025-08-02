import pandas as pd
import numpy as np
from model import HateSpeechDetector
from data_generator import create_enhanced_dataset
import os

def run_demo():
    print("Hate Speech Detector Demo")
    print("=" * 50)
    
    if not os.path.exists('enhanced_dataset.csv'):
        print("Creating enhanced dataset...")
        create_enhanced_dataset()
    
    print("Initializing model...")
    detector = HateSpeechDetector(
        max_vocab_size=10000,
        max_sequence_length=100,
        embedding_dim=128
    )
    
    df = pd.read_csv('enhanced_dataset.csv')
    sample_texts = df['tweet'].sample(n=1000).values
    sample_labels = df['class'].sample(n=1000).values
    
    print("Preparing data...")
    X_processed, y_processed = detector.prepare_data(sample_texts, sample_labels)
    
    print("Training LSTM model (demo mode - 5 epochs)...")
    X_train, X_val, y_train, y_val = X_processed[:800], X_processed[800:900], y_processed[:800], y_processed[800:900]
    X_test, y_test = X_processed[900:], y_processed[900:]
    
    history = detector.train(
        X_train, y_train, X_val, y_val,
        model_type='lstm',
        epochs=5,
        batch_size=32
    )
    
    print("Evaluating model...")
    accuracy, report, y_pred = detector.evaluate(X_test, y_test)
    
    print(f"\nDemo Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    print("\nTesting with example texts...")
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
        "Have a wonderful day!"
    ]
    
    predictions = detector.predict(test_texts)
    class_names = ['Hate Speech', 'Offensive Language', 'Neither']
    
    print("\nExample Predictions:")
    print("-" * 80)
    for i, text in enumerate(test_texts):
        pred_class = np.argmax(predictions[i])
        confidence = np.max(predictions[i])
        print(f"{i+1:2d}. Text: {text[:60]}...")
        print(f"    Prediction: {class_names[pred_class]}")
        print(f"    Confidence: {confidence:.4f}")
        print()
    
    print("Demo completed!")
    print("To train a full model, run: python train.py")
    print("To make predictions, run: python predict.py --interactive")

if __name__ == "__main__":
    run_demo() 