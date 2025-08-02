import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(csv_file):
    print(f"Loading data from {csv_file}...")
    
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
            raise ValueError("Could not find expected columns. Please check your CSV structure.")
        
        print(f"Loaded {len(texts)} texts")
        print(f"Label distribution:\n{labels.value_counts()}")
        
        return texts, labels
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def clean_text(text):
    import re
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def train_model(texts, labels):
    print("\nPreprocessing texts...")
    cleaned_texts = [clean_text(text) for text in texts]
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print("Creating pipeline...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return pipeline, X_test, y_test, y_pred

def test_model(pipeline):
    print("\nTesting model with sample texts...")
    
    test_texts = [
        "Hello world, how are you today?",
        "I hate you so much",
        "You are the worst person; you should just end yourself",
        "This is a terrible movie",
        "You are stupid and worthless",
        "Have a great day!",
        "I love this weather",
        "You are an idiot",
        "Please help me with this problem",
        "Go kill yourself"
    ]
    
    predictions = pipeline.predict_proba(test_texts)
    classes = pipeline.classes_
    
    print(f"Model classes: {classes}")
    print("\nTest Results:")
    for i, text in enumerate(test_texts):
        class_idx = np.argmax(predictions[i])
        confidence = np.max(predictions[i])
        predicted_class = classes[class_idx]
        
        print(f"\nText: '{text}'")
        print(f"Predicted: {predicted_class} (confidence: {confidence:.3f})")
        print(f"Probabilities: {dict(zip(classes, predictions[i]))}")

def save_model(pipeline, filename='simple_hate_speech_detector.pkl'):
    print(f"\nSaving model to {filename}...")
    
    try:
        with open(filename, 'wb') as f:
            pickle.dump(pipeline, f)
        
        file_size = os.path.getsize(filename) / (1024 * 1024)
        print(f"Model saved successfully! File size: {file_size:.2f} MB")
        return True
        
    except Exception as e:
        print(f"Error saving model: {e}")
        return False

def main():
    csv_file = 'dataset_tweet.csv'
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        print("Please make sure the CSV file is in the same directory as this script.")
        return
    
    texts, labels = load_and_prepare_data(csv_file)
    
    if texts is None:
        return
    
    pipeline, X_test, y_test, y_pred = train_model(texts, labels)
    
    test_model(pipeline)
    
    success = save_model(pipeline)
    
    if success:
        print("\n✅ Training completed successfully!")
        print("The new model has been saved as 'simple_hate_speech_detector.pkl'")
        print("You can now use this model with your FastAPI application.")
    else:
        print("\n❌ Error saving the model.")

if __name__ == "__main__":
    main() 