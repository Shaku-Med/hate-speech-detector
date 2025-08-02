import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
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

def create_models():
    models = {
        'random_forest': Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=15000,
                ngram_range=(1, 3),
                stop_words='english',
                min_df=2,
                max_df=0.9
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=300,
                max_depth=25,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ))
        ]),
        
        'gradient_boosting': Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=12000,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=3,
                max_df=0.85
            )),
            ('classifier', GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            ))
        ]),
        
        'logistic_regression': Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=2,
                max_df=0.95
            )),
            ('classifier', LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            ))
        ]),
        
        'svm': Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=8000,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=2,
                max_df=0.9
            )),
            ('classifier', SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                random_state=42
            ))
        ])
    }
    
    return models

def train_and_evaluate_models(texts, labels):
    print("\nPreprocessing texts...")
    cleaned_texts = [clean_text(text) for text in texts]
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    models = create_models()
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name.upper()} model...")
        print(f"{'='*50}")
        
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred
            }
            
        except Exception as e:
            print(f"Error training {name}: {e}")
            continue
    
    return results, X_test, y_test

def find_best_model(results):
    if not results:
        return None
    
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_accuracy = results[best_model_name]['accuracy']
    
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_model_name.upper()}")
    print(f"Accuracy: {best_accuracy:.4f}")
    print(f"{'='*60}")
    
    return best_model_name, results[best_model_name]['model']

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
        "Go kill yourself",
        "You should die",
        "I hope you suffer",
        "You're a piece of garbage",
        "Nobody likes you",
        "You deserve to be hurt"
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
    
    results, X_test, y_test = train_and_evaluate_models(texts, labels)
    
    if not results:
        print("No models were successfully trained!")
        return
    
    best_model_name, best_model = find_best_model(results)
    
    if best_model is None:
        print("No best model found!")
        return
    
    test_model(best_model)
    
    success = save_model(best_model)
    
    if success:
        print("\n✅ Training completed successfully!")
        print(f"Best model ({best_model_name}) has been saved as 'simple_hate_speech_detector.pkl'")
        print("You can now use this model with your FastAPI application.")
    else:
        print("\n❌ Error saving the model.")

if __name__ == "__main__":
    main() 