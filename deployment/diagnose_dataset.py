import pandas as pd
import numpy as np
import pickle
import os

def examine_dataset():
    print("=== DATASET DIAGNOSIS ===\n")
    
    csv_file = 'dataset_tweet.csv'
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        return
    
    try:
        df = pd.read_csv(csv_file)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        if 'tweet' in df.columns and 'class' in df.columns:
            texts = df['tweet'].fillna('').astype(str)
            labels = df['class']
        elif 'text' in df.columns and 'label' in df.columns:
            texts = df['text'].fillna('').astype(str)
            labels = df['label']
        else:
            print("Available columns:", df.columns.tolist())
            return
        
        print(f"\nLabel distribution:")
        print(labels.value_counts())
        print(f"\nLabel distribution (%):")
        print(labels.value_counts(normalize=True) * 100)
        
        print(f"\n=== SAMPLE TEXTS BY CLASS ===")
        for label in labels.unique():
            print(f"\n--- {label} ---")
            sample_texts = texts[labels == label].head(5).tolist()
            for i, text in enumerate(sample_texts, 1):
                print(f"{i}. {text[:100]}{'...' if len(text) > 100 else ''}")
        
        return texts, labels
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

def test_current_model():
    print("\n=== CURRENT MODEL TEST ===\n")
    
    model_path = 'simple_hate_speech_detector.pkl'
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        return
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        test_texts = [
            "You're so beautiful",
            "You are beautiful",
            "Hello world",
            "I love you",
            "Have a great day",
            "You are amazing",
            "This is wonderful",
            "I hate you",
            "You are stupid",
            "Go die",
            "You should kill yourself",
            "You're the worst",
            "I hope you suffer"
        ]
        
        predictions = model.predict_proba(test_texts)
        classes = model.classes_
        
        print(f"Model classes: {classes}")
        print("\nTest Results:")
        for i, text in enumerate(test_texts):
            class_idx = np.argmax(predictions[i])
            confidence = np.max(predictions[i])
            predicted_class = classes[class_idx]
            
            print(f"\nText: '{text}'")
            print(f"Predicted: {predicted_class} (confidence: {confidence:.3f})")
            print(f"Probabilities: {dict(zip(classes, predictions[i]))}")
            
    except Exception as e:
        print(f"Error testing model: {e}")

def analyze_text_patterns(texts, labels):
    print("\n=== TEXT PATTERN ANALYSIS ===\n")
    
    positive_words = ['beautiful', 'love', 'amazing', 'wonderful', 'great', 'good', 'nice', 'happy']
    negative_words = ['hate', 'stupid', 'kill', 'die', 'worst', 'terrible', 'awful']
    
    print("Analyzing positive words in different classes:")
    for word in positive_words:
        positive_matches = texts.str.contains(word, case=False, na=False)
        if positive_matches.any():
            print(f"\n'{word}' appears in:")
            for label in labels.unique():
                count = (positive_matches & (labels == label)).sum()
                total = (labels == label).sum()
                if count > 0:
                    print(f"  {label}: {count}/{total} ({count/total*100:.1f}%)")
    
    print("\nAnalyzing negative words in different classes:")
    for word in negative_words:
        negative_matches = texts.str.contains(word, case=False, na=False)
        if negative_matches.any():
            print(f"\n'{word}' appears in:")
            for label in labels.unique():
                count = (negative_matches & (labels == label)).sum()
                total = (labels == label).sum()
                if count > 0:
                    print(f"  {label}: {count}/{total} ({count/total*100:.1f}%)")

def main():
    texts, labels = examine_dataset()
    
    if texts is not None:
        analyze_text_patterns(texts, labels)
    
    test_current_model()
    
    print("\n=== RECOMMENDATIONS ===")
    print("1. Check if your dataset has mislabeled examples")
    print("2. Ensure 'beautiful', 'love', 'amazing' etc. are not labeled as offensive")
    print("3. Consider retraining with a cleaner dataset")
    print("4. Check for data leakage or preprocessing issues")

if __name__ == "__main__":
    main() 