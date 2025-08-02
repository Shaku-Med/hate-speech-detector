import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class SimpleHateSpeechDetector:
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.pipeline = None
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        self.classifier = self._get_classifier()
        self.class_names = ['Hate Speech', 'Offensive Language', 'Neither']
        
    def _get_classifier(self):
        if self.model_type == 'random_forest':
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == 'logistic_regression':
            return LogisticRegression(random_state=42, max_iter=1000)
        elif self.model_type == 'svm':
            return SVC(random_state=42, probability=True)
        else:
            return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def preprocess_text(self, text):
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def prepare_data(self, texts, labels):
        processed_texts = [self.preprocess_text(text) for text in texts]
        return processed_texts, np.array(labels)
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        print(f"Training {self.model_type} model...")
        
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.classifier)
        ])
        
        self.pipeline.fit(X_train, y_train)
        
        if X_val is not None and y_val is not None:
            y_pred = self.pipeline.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            print(f"Validation Accuracy: {accuracy:.4f}")
        
        return self.pipeline
    
    def predict(self, texts):
        if self.pipeline is None:
            raise ValueError("Model not trained yet")
        
        processed_texts = [self.preprocess_text(text) for text in texts]
        predictions = self.pipeline.predict(processed_texts)
        probabilities = self.pipeline.predict_proba(processed_texts)
        
        return predictions, probabilities
    
    def predict_single(self, text):
        predictions, probabilities = self.predict([text])
        return {
            'text': text,
            'predicted_class': self.class_names[predictions[0]],
            'confidence': np.max(probabilities[0]),
            'probabilities': {
                'hate_speech': float(probabilities[0][0]),
                'offensive_language': float(probabilities[0][1]),
                'neither': float(probabilities[0][2])
            }
        }
    
    def evaluate(self, X_test, y_test):
        if self.pipeline is None:
            raise ValueError("Model not trained yet")
        
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=self.class_names)
        
        return accuracy, report, y_pred
    
    def save_model(self, filepath):
        if self.pipeline is None:
            raise ValueError("Model not trained yet")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.pipeline, f)
        
        print(f"Model saved as {filepath}")
    
    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            self.pipeline = pickle.load(f)
        
        print(f"Model loaded from {filepath}")

def create_simple_datasets():
    hate_phrases = [
        "kill all", "death to", "exterminate", "eliminate", "destroy", "hate", "despise",
        "racist", "bigot", "nazi", "fascist", "supremacist", "terrorist", "extremist",
        "white power", "black power", "racial", "ethnic", "religious", "discrimination",
        "prejudice", "bias", "stereotype", "profiling", "segregation", "apartheid",
        "genocide", "ethnic cleansing", "holocaust", "massacre", "slaughter", "butcher",
        "lynch", "hang", "burn", "torture", "abuse", "violence", "aggression", "attack"
    ]
    
    offensive_phrases = [
        "fuck", "shit", "bitch", "asshole", "dick", "pussy", "cock", "whore", "slut",
        "bastard", "motherfucker", "fucker", "cunt", "damn", "hell", "god damn",
        "son of a bitch", "piece of shit", "dumbass", "idiot", "stupid", "moron",
        "retard", "fool", "jerk", "ass", "butt", "buttocks", "penis", "vagina",
        "sexual", "porn", "pornography", "nude", "naked", "sex", "intercourse"
    ]
    
    neutral_phrases = [
        "hello", "good morning", "how are you", "nice day", "weather", "food", "music",
        "movie", "book", "work", "study", "learn", "teach", "help", "support", "love",
        "family", "friend", "happy", "sad", "angry", "excited", "tired", "sleep",
        "eat", "drink", "walk", "run", "play", "game", "sport", "exercise", "health",
        "doctor", "hospital", "school", "university", "job", "career", "money", "time"
    ]
    
    data = []
    
    for i in range(3000):
        if i < 1200:
            text = " ".join(np.random.choice(hate_phrases, np.random.randint(2, 5)))
            class_label = 0
        elif i < 2100:
            text = " ".join(np.random.choice(offensive_phrases, np.random.randint(2, 4)))
            class_label = 1
        else:
            text = " ".join(np.random.choice(neutral_phrases, np.random.randint(3, 6)))
            class_label = 2
            
        data.append([i, text, class_label])
    
    df = pd.DataFrame(data, columns=['index', 'text', 'class'])
    return df

def train_simple_model():
    print("Simple Hate Speech Detector Training")
    print("=" * 50)
    
    try:
        df = pd.read_csv('dataset_tweet.csv')
        print(f"Loaded original dataset: {len(df)} samples")
    except:
        print("Original dataset not found, creating synthetic data...")
        df = create_simple_datasets()
        df.to_csv('simple_dataset.csv', index=False)
        print(f"Created synthetic dataset: {len(df)} samples")
    
    texts = df['tweet'].values if 'tweet' in df.columns else df['text'].values
    labels = df['class'].values
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    detector = SimpleHateSpeechDetector(model_type='random_forest')
    
    X_train_processed, y_train_processed = detector.prepare_data(X_train, y_train)
    X_val_processed, y_val_processed = detector.prepare_data(X_val, y_val)
    X_test_processed, y_test_processed = detector.prepare_data(X_test, y_test)
    
    detector.train(X_train_processed, y_train_processed, X_val_processed, y_val_processed)
    
    accuracy, report, y_pred = detector.evaluate(X_test_processed, y_test_processed)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    detector.save_model('simple_hate_speech_detector.pkl')
    
    return detector, accuracy, report

if __name__ == "__main__":
    train_simple_model() 