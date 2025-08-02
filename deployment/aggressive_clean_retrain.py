import csv
import pickle
import os
import re
import math
import random
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

def aggressive_clean_dataset():
    print("=== AGGRESSIVE DATASET CLEANING ===\n")
    
    csv_file = 'dataset_tweet.csv'
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        return None, None
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        print(f"Original dataset: {len(rows)} rows")
        
        # Define comprehensive positive and negative word sets
        positive_words = {
            'beautiful', 'love', 'amazing', 'wonderful', 'great', 'good', 'nice', 'happy',
            'awesome', 'fantastic', 'excellent', 'perfect', 'brilliant', 'outstanding',
            'sweet', 'kind', 'gentle', 'caring', 'helpful', 'friendly', 'warm', 'bright',
            'gorgeous', 'stunning', 'lovely', 'charming', 'delightful', 'pleasant',
            'beautiful', 'precious', 'adorable', 'cute', 'sweet', 'angel', 'heaven',
            'blessed', 'grateful', 'thankful', 'appreciate', 'wonderful', 'magical',
            'incredible', 'extraordinary', 'marvelous', 'splendid', 'magnificent'
        }
        
        negative_words = {
            'hate', 'kill', 'die', 'stupid', 'idiot', 'moron', 'retard', 'fuck', 'shit',
            'bitch', 'whore', 'slut', 'nigger', 'faggot', 'dyke', 'cunt', 'pussy',
            'dick', 'cock', 'asshole', 'bastard', 'motherfucker', 'fucker', 'worthless',
            'useless', 'disgusting', 'pathetic', 'scum', 'trash', 'garbage', 'ugly',
            'fat', 'dumb', 'lazy', 'annoying', 'irritating', 'boring', 'dull'
        }
        
        # Define positive and negative phrases
        positive_phrases = [
            'i love you', 'you are beautiful', 'you are amazing', 'have a great day',
            'thank you', 'you are wonderful', 'you are perfect', 'you are awesome',
            'you are fantastic', 'you are incredible', 'you are precious', 'you are sweet'
        ]
        
        negative_phrases = [
            'kill yourself', 'go die', 'you should die', 'end yourself', 'fuck you',
            'you are stupid', 'you are an idiot', 'you are worthless', 'you are useless',
            'you are disgusting', 'you are pathetic', 'you are trash', 'you are garbage'
        ]
        
        cleaned_rows = []
        removed_count = 0
        
        for row in rows:
            text = row['tweet'].lower()
            label = int(row['class'])
            
            # Count positive and negative words
            positive_count = sum(1 for word in positive_words if word in text)
            negative_count = sum(1 for word in negative_words if word in text)
            
            # Check for positive and negative phrases
            positive_phrase_count = sum(1 for phrase in positive_phrases if phrase in text)
            negative_phrase_count = sum(1 for phrase in negative_phrases if phrase in text)
            
            # Aggressive cleaning rules
            should_remove = False
            
            # Remove texts with positive words/phrases labeled as offensive/hate
            if (positive_count > 0 or positive_phrase_count > 0) and negative_count == 0 and label != 0:
                should_remove = True
            
            # Remove texts with negative words/phrases labeled as neither
            if (negative_count > 0 or negative_phrase_count > 0) and label == 0:
                should_remove = True
            
            # Remove texts that are clearly positive but mislabeled
            if positive_count >= 2 and label != 0:
                should_remove = True
            
            # Remove texts that are clearly negative but labeled as neither
            if negative_count >= 2 and label == 0:
                should_remove = True
            
            if should_remove:
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
        
        # Balance the dataset
        balanced_texts, balanced_labels = balance_dataset(texts, labels)
        
        return balanced_texts, balanced_labels
        
    except Exception as e:
        print(f"Error cleaning dataset: {e}")
        return None, None

def balance_dataset(texts, labels):
    print("\n=== BALANCING DATASET ===\n")
    
    # Group by class
    class_data = defaultdict(list)
    for text, label in zip(texts, labels):
        class_data[label].append(text)
    
    # Find the minimum class size
    min_class_size = min(len(texts) for texts in class_data.values())
    print(f"Minimum class size: {min_class_size}")
    
    # Balance by undersampling majority classes
    balanced_texts = []
    balanced_labels = []
    
    for class_label, class_texts in class_data.items():
        if len(class_texts) > min_class_size:
            # Randomly sample to match minimum size
            random.seed(42)  # For reproducibility
            sampled_texts = random.sample(class_texts, min_class_size)
        else:
            sampled_texts = class_texts
        
        balanced_texts.extend(sampled_texts)
        balanced_labels.extend([class_label] * len(sampled_texts))
        
        print(f"  {class_label}: {len(sampled_texts)} samples")
    
    # Shuffle the data
    combined = list(zip(balanced_texts, balanced_labels))
    random.shuffle(combined)
    balanced_texts, balanced_labels = zip(*combined)
    
    print(f"Balanced dataset: {len(balanced_texts)} total samples")
    
    return list(balanced_texts), list(balanced_labels)

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
    print("=== AGGRESSIVE CLEANING AND RETRAINING ===\n")
    
    # Clean and balance the dataset
    texts, labels = aggressive_clean_dataset()
    
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
        print("This model was trained on aggressively cleaned and balanced data!")
        print("It should now correctly classify positive texts as 'neither'.")
    else:
        print(f"\n❌ Model accuracy too low ({accuracy:.4f}). Not saving.")

if __name__ == "__main__":
    main() 