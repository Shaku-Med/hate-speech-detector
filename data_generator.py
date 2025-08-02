import pandas as pd
import numpy as np
import random
from textblob import TextBlob

def generate_synthetic_data():
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
    
    for i in range(5000):
        if i < 2000:
            text = " ".join(random.sample(hate_phrases, random.randint(2, 5)))
            hate_speech = 1
            offensive_language = random.randint(0, 2)
            neither = 0
            class_label = 0
        elif i < 3500:
            text = " ".join(random.sample(offensive_phrases, random.randint(2, 4)))
            hate_speech = 0
            offensive_language = 1
            neither = 0
            class_label = 1
        else:
            text = " ".join(random.sample(neutral_phrases, random.randint(3, 6)))
            hate_speech = 0
            offensive_language = 0
            neither = 1
            class_label = 2
            
        count = random.randint(1, 10)
        data.append([i, count, hate_speech, offensive_language, neither, class_label, text])
    
    df = pd.DataFrame(data, columns=['index', 'count', 'hate_speech', 'offensive_language', 'neither', 'class', 'tweet'])
    return df

def generate_contextual_data():
    contextual_data = []
    
    hate_contexts = [
        "I hate all {} people", "{} should be eliminated", "Death to all {}",
        "{} are inferior", "We need to get rid of {}", "{} don't belong here",
        "Kill all {}", "Exterminate {}", "{} are the problem", "Remove {} from society"
    ]
    
    offensive_contexts = [
        "You're such a {} idiot", "What a {} moron", "This {} is stupid",
        "Fuck this {}", "I can't stand this {}", "This {} is annoying",
        "Shut up you {}", "Go away {}", "Nobody likes you {}", "You're a {}"
    ]
    
    neutral_contexts = [
        "I love {} food", "{} is beautiful", "Great {} today", "Nice {} weather",
        "Good {} movie", "Interesting {} book", "Amazing {} music", "Wonderful {} day",
        "Excellent {} work", "Fantastic {} performance"
    ]
    
    groups = ["black", "white", "asian", "hispanic", "jewish", "muslim", "christian", "gay", "lesbian", "transgender"]
    objects = ["pizza", "coffee", "music", "art", "nature", "technology", "science", "history", "culture", "sports"]
    
    for i in range(3000):
        if i < 1200:
            template = random.choice(hate_contexts)
            group = random.choice(groups)
            text = template.format(group)
            hate_speech = 1
            offensive_language = random.randint(0, 1)
            neither = 0
            class_label = 0
        elif i < 2100:
            template = random.choice(offensive_contexts)
            group = random.choice(groups)
            text = template.format(group)
            hate_speech = 0
            offensive_language = 1
            neither = 0
            class_label = 1
        else:
            template = random.choice(neutral_contexts)
            obj = random.choice(objects)
            text = template.format(obj)
            hate_speech = 0
            offensive_language = 0
            neither = 1
            class_label = 2
            
        count = random.randint(1, 8)
        contextual_data.append([i, count, hate_speech, offensive_language, neither, class_label, text])
    
    df = pd.DataFrame(contextual_data, columns=['index', 'count', 'hate_speech', 'offensive_language', 'neither', 'class', 'tweet'])
    return df

def create_enhanced_dataset():
    original_df = pd.read_csv('dataset_tweet.csv')
    
    synthetic_df = generate_synthetic_data()
    contextual_df = generate_contextual_data()
    
    enhanced_df = pd.concat([original_df, synthetic_df, contextual_df], ignore_index=True)
    enhanced_df = enhanced_df.sample(frac=1).reset_index(drop=True)
    
    enhanced_df.to_csv('enhanced_dataset.csv', index=False)
    synthetic_df.to_csv('synthetic_dataset.csv', index=False)
    contextual_df.to_csv('contextual_dataset.csv', index=False)
    
    print(f"Original dataset: {len(original_df)} samples")
    print(f"Synthetic dataset: {len(synthetic_df)} samples")
    print(f"Contextual dataset: {len(contextual_df)} samples")
    print(f"Enhanced dataset: {len(enhanced_df)} samples")
    
    return enhanced_df

if __name__ == "__main__":
    create_enhanced_dataset() 