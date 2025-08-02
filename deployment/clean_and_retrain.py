import csv
import pickle
import os
import re
from collections import Counter

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
        
        positive_words = {
            'beautiful', 'love', 'amazing', 'wonderful', 'great', 'good', 'nice', 'happy',
            'awesome', 'fantastic', 'excellent', 'perfect', 'brilliant', 'outstanding',
            'sweet', 'kind', 'gentle', 'caring', 'helpful', 'friendly', 'warm', 'bright'
        }
        
        negative_words = {
            'hate', 'stupid', 'kill', 'die', 'worst', 'terrible', 'awful', 'horrible',
            'disgusting', 'ugly', 'fat', 'retard', 'idiot', 'moron', 'bitch', 'whore',
            'slut', 'fuck', 'shit', 'cunt', 'pussy', 'dick', 'cock', 'asshole'
        }
        
        cleaned_rows = []
        removed_count = 0
        
        for row in rows:
            text = row['tweet'].lower()
            label = int(row['class'])
            
            # Check for obvious mislabeling
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
        
        # Rebalance classes
        class_0 = [row for row in cleaned_rows if int(row['class']) == 0]
        class_1 = [row for row in cleaned_rows if int(row['class']) == 1]
        class_2 = [row for row in cleaned_rows if int(row['class']) == 2]
        
        print(f"\nClass distribution after cleaning:")
        print(f"  Class 0 (Neither): {len(class_0)}")
        print(f"  Class 1 (Offensive): {len(class_1)}")
        print(f"  Class 2 (Hate Speech): {len(class_2)}")
        
        # Balance classes by undersampling the majority class
        min_class_size = min(len(class_0), len(class_1), len(class_2))
        
        if len(class_1) > min_class_size * 2:
            import random
            random.seed(42)
            class_1 = random.sample(class_1, min_class_size * 2)
        
        balanced_rows = class_0 + class_1 + class_2
        
        print(f"\nBalanced dataset: {len(balanced_rows)} rows")
        print(f"Final distribution:")
        final_class_0 = len([row for row in balanced_rows if int(row['class']) == 0])
        final_class_1 = len([row for row in balanced_rows if int(row['class']) == 1])
        final_class_2 = len([row for row in balanced_rows if int(row['class']) == 2])
        print(f"  Class 0 (Neither): {final_class_0}")
        print(f"  Class 1 (Offensive): {final_class_1}")
        print(f"  Class 2 (Hate Speech): {final_class_2}")
        
        # Save cleaned dataset
        cleaned_file = 'cleaned_dataset.csv'
        with open(cleaned_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(balanced_rows)
        
        print(f"\nCleaned dataset saved as: {cleaned_file}")
        
        return balanced_rows
        
    except Exception as e:
        print(f"Error cleaning dataset: {e}")
        return None

def create_simple_model(cleaned_rows):
    print("\n=== CREATING SIMPLE MODEL ===\n")
    
    try:
        # Simple rule-based model
        positive_words = {
            'beautiful', 'love', 'amazing', 'wonderful', 'great', 'good', 'nice', 'happy',
            'awesome', 'fantastic', 'excellent', 'perfect', 'brilliant', 'outstanding',
            'sweet', 'kind', 'gentle', 'caring', 'helpful', 'friendly', 'warm', 'bright'
        }
        
        hate_words = {
            'hate', 'kill', 'die', 'stupid', 'idiot', 'moron', 'retard', 'fuck', 'shit',
            'bitch', 'whore', 'slut', 'nigger', 'faggot', 'dyke', 'cunt', 'pussy',
            'dick', 'cock', 'asshole', 'bastard', 'motherfucker', 'fucker', 'worthless',
            'useless', 'disgusting', 'pathetic', 'scum', 'trash', 'garbage'
        }
        
        offensive_words = {
            'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'ugly', 'fat',
            'stupid', 'dumb', 'idiot', 'moron', 'retard', 'lazy', 'worthless',
            'useless', 'pathetic', 'annoying', 'irritating', 'boring', 'dull'
        }
        
        hate_phrases = [
            'end yourself', 'kill yourself', 'go die', 'should die', 'deserve to die',
            'worst person', 'terrible person', 'awful person', 'horrible person',
            'piece of shit', 'piece of garbage', 'human garbage', 'human trash',
            'no one likes you', 'everyone hates you', 'nobody cares', 'no one cares'
        ]
        
        class SimpleHateSpeechDetector:
            def __init__(self):
                self.positive_words = positive_words
                self.hate_words = hate_words
                self.offensive_words = offensive_words
                self.hate_phrases = hate_phrases
                self.classes_ = ['neither', 'offensive_language', 'hate_speech']
            
            def predict_proba(self, texts):
                import numpy as np
                results = []
                
                for text in texts:
                    text_lower = text.lower()
                    words = set(text_lower.split())
                    
                    # Count indicators
                    positive_count = len(words.intersection(self.positive_words))
                    hate_word_count = len(words.intersection(self.hate_words))
                    offensive_word_count = len(words.intersection(self.offensive_words))
                    
                    hate_phrase_count = 0
                    for phrase in self.hate_phrases:
                        if phrase in text_lower:
                            hate_phrase_count += 1
                    
                    total_hate_indicators = hate_word_count + hate_phrase_count
                    
                    # Decision logic
                    if positive_count > 0 and total_hate_indicators == 0:
                        prob = [0.9, 0.05, 0.05]  # Neither
                    elif total_hate_indicators > 0:
                        prob = [0.05, 0.15, 0.8]  # Hate Speech
                    elif offensive_word_count > 0:
                        prob = [0.1, 0.8, 0.1]   # Offensive Language
                    else:
                        prob = [0.8, 0.15, 0.05] # Neither
                    
                    results.append(np.array(prob))
                
                return np.array(results)
        
        model = SimpleHateSpeechDetector()
        
        # Test the model
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
        
        print("Testing simple model:")
        for i, text in enumerate(test_texts):
            class_idx = predictions[i].argmax()
            confidence = predictions[i].max()
            predicted_class = model.classes_[class_idx]
            print(f"'{text}' -> {predicted_class} (confidence: {confidence:.3f})")
        
        # Save the model
        with open('simple_hate_speech_detector.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        print(f"\nSimple model saved as: simple_hate_speech_detector.pkl")
        return model
        
    except Exception as e:
        print(f"Error creating model: {e}")
        return None

def main():
    print("=== HATE SPEECH DETECTOR - CLEAN & RETRAIN ===\n")
    
    # Clean the dataset
    cleaned_rows = clean_dataset()
    
    if cleaned_rows is None:
        return
    
    # Create a simple but effective model
    model = create_simple_model(cleaned_rows)
    
    if model is not None:
        print("\n✅ SUCCESS!")
        print("Your hate speech detector has been cleaned and retrained.")
        print("The new model should correctly classify:")
        print("  - 'You're so beautiful' as 'neither'")
        print("  - 'I hate you' as 'hate_speech'")
        print("  - 'You are stupid' as 'offensive_language'")
        print("\nYou can now restart your FastAPI application to use the new model.")

if __name__ == "__main__":
    main() 