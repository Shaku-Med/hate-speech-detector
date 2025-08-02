import csv
import pickle
import os
import re
import math
from collections import Counter, defaultdict

class NaiveBayesHateSpeechDetector:
    def __init__(self):
        self.class_priors = {}
        self.word_probs = {}
        self.vocabulary = set()
        self.classes = []
        
    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def extract_features(self, text):
        words = self.clean_text(text).split()
        return [word for word in words if len(word) > 2]
    
    def train(self, texts, labels):
        print("Training Naive Bayes model...")
        
        # Get unique classes
        self.classes = list(set(labels))
        print(f"Classes found: {self.classes}")
        
        # Calculate class priors
        class_counts = Counter(labels)
        total_docs = len(texts)
        
        for class_label in self.classes:
            self.class_priors[class_label] = class_counts[class_label] / total_docs
        
        print(f"Class priors: {self.class_priors}")
        
        # Build vocabulary and count words per class
        word_counts = {}
        class_word_counts = {}
        
        for class_label in self.classes:
            word_counts[class_label] = Counter()
            class_word_counts[class_label] = 0
        
        for text, label in zip(texts, labels):
            features = self.extract_features(text)
            for word in features:
                word_counts[label][word] += 1
                class_word_counts[label] += 1
                self.vocabulary.add(word)
        
        # Calculate word probabilities using Laplace smoothing
        vocab_size = len(self.vocabulary)
        alpha = 1.0  # Laplace smoothing parameter
        
        self.word_probs = {}
        for class_label in self.classes:
            self.word_probs[class_label] = {}
            total_words_in_class = class_word_counts[class_label]
            
            for word in self.vocabulary:
                word_count = word_counts[class_label][word]
                # P(word|class) = (count + alpha) / (total_words + alpha * vocab_size)
                prob = (word_count + alpha) / (total_words_in_class + alpha * vocab_size)
                self.word_probs[class_label][word] = prob
        
        print(f"Vocabulary size: {len(self.vocabulary)}")
        print("Training completed!")
    
    def predict_proba(self, texts):
        results = []
        
        for text in texts:
            features = self.extract_features(text)
            class_scores = {}
            
            for class_label in self.classes:
                # Start with class prior
                score = math.log(self.class_priors[class_label])
                
                # Add word probabilities
                for word in features:
                    if word in self.vocabulary:
                        word_prob = self.word_probs[class_label][word]
                        score += math.log(word_prob)
                
                class_scores[class_label] = score
            
            # Convert log scores to probabilities
            max_score = max(class_scores.values())
            exp_scores = {cls: math.exp(score - max_score) for cls, score in class_scores.items()}
            total = sum(exp_scores.values())
            
            # Normalize to get probabilities
            probabilities = [exp_scores.get(cls, 0) / total for cls in self.classes]
            results.append(probabilities)
        
        return results
    
    def predict(self, texts):
        probas = self.predict_proba(texts)
        predictions = []
        
        for prob in probas:
            class_idx = prob.index(max(prob))
            predictions.append(self.classes[class_idx])
        
        return predictions

def clean_dataset():
    print("=== CLEANING DATASET ===\n")
    
    csv_file = 'dataset_tweet.csv'
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        return None, None
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        print(f"Original dataset: {len(rows)} rows")
        
        # Define positive and negative word sets
        positive_words = {
            'beautiful', 'love', 'amazing', 'wonderful', 'great', 'good', 'nice', 'happy',
            'awesome', 'fantastic', 'excellent', 'perfect', 'brilliant', 'outstanding',
            'sweet', 'kind', 'gentle', 'caring', 'helpful', 'friendly', 'warm', 'bright',
            'gorgeous', 'stunning', 'lovely', 'charming', 'delightful', 'pleasant'
        }
        
        negative_words = {
            'hate', 'kill', 'die', 'stupid', 'idiot', 'moron', 'retard', 'fuck', 'shit',
            'bitch', 'whore', 'slut', 'nigger', 'faggot', 'dyke', 'cunt', 'pussy',
            'dick', 'cock', 'asshole', 'bastard', 'motherfucker', 'fucker', 'worthless',
            'useless', 'disgusting', 'pathetic', 'scum', 'trash', 'garbage'
        }
        
        cleaned_rows = []
        removed_count = 0
        
        for row in rows:
            text = row['tweet'].lower()
            label = int(row['class'])
            
            # Count positive and negative words
            positive_count = sum(1 for word in positive_words if word in text)
            negative_count = sum(1 for word in negative_words if word in text)
            
            # Remove obvious mislabeled examples
            if positive_count > 0 and negative_count == 0 and label != 0:
                removed_count += 1
                continue
            
            # Remove examples with too many negative words labeled as "neither"
            if negative_count >= 2 and label == 0:
                removed_count += 1
                continue
            
            cleaned_rows.append(row)
        
        print(f"Removed {removed_count} mislabeled examples")
        print(f"Cleaned dataset: {len(cleaned_rows)} rows")
        
        # Convert to texts and labels
        texts = [row['tweet'] for row in cleaned_rows]
        labels = [int(row['class']) for row in cleaned_rows]
        
        # Convert numeric labels to string labels
        label_mapping = {0: 'neither', 1: 'offensive_language', 2: 'hate_speech'}
        labels = [label_mapping[label] for label in labels]
        
        # Show distribution
        label_counts = Counter(labels)
        print(f"\nLabel distribution after cleaning:")
        for label, count in label_counts.items():
            percentage = (count / len(labels)) * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")
        
        return texts, labels
        
    except Exception as e:
        print(f"Error cleaning dataset: {e}")
        return None, None

def evaluate_model(model, texts, labels, test_size=0.2):
    print(f"\nEvaluating model on {int(len(texts) * test_size)} test samples...")
    
    # Simple train/test split
    split_idx = int(len(texts) * (1 - test_size))
    train_texts, test_texts = texts[:split_idx], texts[split_idx:]
    train_labels, test_labels = labels[:split_idx], labels[split_idx:]
    
    # Train the model
    model.train(train_texts, train_labels)
    
    # Test the model
    predictions = model.predict(test_texts)
    
    # Calculate accuracy
    correct = sum(1 for pred, true in zip(predictions, test_labels) if pred == true)
    accuracy = correct / len(test_labels)
    
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Show some examples
    print(f"\nSample predictions:")
    for i in range(min(10, len(test_texts))):
        text = test_texts[i][:50] + "..." if len(test_texts[i]) > 50 else test_texts[i]
        pred = predictions[i]
        true = test_labels[i]
        status = "✅" if pred == true else "❌"
        print(f"{status} '{text}' -> {pred} (true: {true})")
    
    return model, accuracy

def test_model(model):
    print(f"\nTesting model with sample texts...")
    
    test_texts = [
        "You're so beautiful",
        "I love you",
        "Have a great day",
        "You are amazing",
        "I hate you",
        "You are stupid",
        "Go kill yourself",
        "You should die",
        "You're the worst person",
        "This is terrible"
    ]
    
    predictions = model.predict_proba(test_texts)
    
    for i, text in enumerate(test_texts):
        class_idx = predictions[i].index(max(predictions[i]))
        confidence = max(predictions[i])
        predicted_class = model.classes[class_idx]
        
        print(f"'{text}' -> {predicted_class} (confidence: {confidence:.3f})")
        print(f"  Probabilities: {dict(zip(model.classes, predictions[i]))}")

def main():
    print("=== CLEANING DATASET AND RETRAINING ML MODEL ===\n")
    
    # Clean the dataset
    texts, labels = clean_dataset()
    
    if texts is None:
        return
    
    # Create and train model
    model = NaiveBayesHateSpeechDetector()
    
    # Evaluate model
    trained_model, accuracy = evaluate_model(model, texts, labels)
    
    if accuracy > 0.5:  # Only save if accuracy is reasonable
        # Test the model
        test_model(trained_model)
        
        # Save the model
        with open('simple_hate_speech_detector.pkl', 'wb') as f:
            pickle.dump(trained_model, f)
        
        print(f"\n✅ Model saved as: simple_hate_speech_detector.pkl")
        print(f"Model accuracy: {accuracy:.4f}")
        print("This model was trained on cleaned data with mislabeled examples removed!")
        print("It should now correctly classify positive texts as 'neither'.")
    else:
        print(f"\n❌ Model accuracy too low ({accuracy:.4f}). Not saving.")

if __name__ == "__main__":
    main() 