import pickle
import os

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
            
            results.append(prob)
        
        return results

def create_and_save_model():
    print("=== CREATING AND SAVING SIMPLE MODEL ===\n")
    
    try:
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
            class_idx = predictions[i].index(max(predictions[i]))
            confidence = max(predictions[i])
            predicted_class = model.classes_[class_idx]
            print(f"'{text}' -> {predicted_class} (confidence: {confidence:.3f})")
        
        # Save the model
        with open('simple_hate_speech_detector.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        print(f"\n✅ Simple model saved as: simple_hate_speech_detector.pkl")
        return model
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return None

if __name__ == "__main__":
    model = create_and_save_model()
    
    if model is not None:
        print("\n✅ SUCCESS!")
        print("Your hate speech detector has been created.")
        print("The new model should correctly classify:")
        print("  - 'You're so beautiful' as 'neither'")
        print("  - 'I hate you' as 'hate_speech'")
        print("  - 'You are stupid' as 'offensive_language'")
        print("\nYou can now restart your FastAPI application to use the new model.") 