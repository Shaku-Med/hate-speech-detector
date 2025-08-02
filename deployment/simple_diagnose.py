import csv
import pickle
import os

def examine_dataset():
    print("=== DATASET DIAGNOSIS ===\n")
    
    csv_file = 'dataset_tweet.csv'
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        return
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        print(f"Dataset has {len(rows)} rows")
        print(f"Columns: {list(rows[0].keys())}")
        
        if 'tweet' in rows[0] and 'class' in rows[0]:
            texts = [row['tweet'] for row in rows]
            labels = [row['class'] for row in rows]
        elif 'text' in rows[0] and 'label' in rows[0]:
            texts = [row['text'] for row in rows]
            labels = [row['label'] for row in rows]
        else:
            print("Available columns:", list(rows[0].keys()))
            return
        
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"\nLabel distribution:")
        for label, count in label_counts.items():
            percentage = (count / len(labels)) * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")
        
        print(f"\n=== SAMPLE TEXTS BY CLASS ===")
        for label in label_counts.keys():
            print(f"\n--- {label} ---")
            sample_count = 0
            for i, (text, text_label) in enumerate(zip(texts, labels)):
                if text_label == label and sample_count < 5:
                    print(f"{sample_count + 1}. {text[:100]}{'...' if len(text) > 100 else ''}")
                    sample_count += 1
        
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
            class_idx = predictions[i].argmax()
            confidence = predictions[i].max()
            predicted_class = classes[class_idx]
            
            print(f"\nText: '{text}'")
            print(f"Predicted: {predicted_class} (confidence: {confidence:.3f})")
            probs = dict(zip(classes, predictions[i]))
            print(f"Probabilities: {probs}")
            
    except Exception as e:
        print(f"Error testing model: {e}")

def analyze_text_patterns(texts, labels):
    print("\n=== TEXT PATTERN ANALYSIS ===\n")
    
    positive_words = ['beautiful', 'love', 'amazing', 'wonderful', 'great', 'good', 'nice', 'happy']
    negative_words = ['hate', 'stupid', 'kill', 'die', 'worst', 'terrible', 'awful']
    
    print("Analyzing positive words in different classes:")
    for word in positive_words:
        print(f"\n'{word}' appears in:")
        for label in set(labels):
            count = sum(1 for text, text_label in zip(texts, labels) 
                       if word.lower() in text.lower() and text_label == label)
            total = sum(1 for text_label in labels if text_label == label)
            if count > 0:
                percentage = (count / total) * 100
                print(f"  {label}: {count}/{total} ({percentage:.1f}%)")
    
    print("\nAnalyzing negative words in different classes:")
    for word in negative_words:
        print(f"\n'{word}' appears in:")
        for label in set(labels):
            count = sum(1 for text, text_label in zip(texts, labels) 
                       if word.lower() in text.lower() and text_label == label)
            total = sum(1 for text_label in labels if text_label == label)
            if count > 0:
                percentage = (count / total) * 100
                print(f"  {label}: {count}/{total} ({percentage:.1f}%)")

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