import re
import string
import pandas as pd
import numpy as np

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    
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
    
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False

class TextPreprocessor:
    def __init__(self):
        self.punctuation = string.punctuation
        
        if NLTK_AVAILABLE:
            try:
                self.stop_words = set(stopwords.words('english'))
                self.lemmatizer = WordNetLemmatizer()
            except:
                self.stop_words = set()
                self.lemmatizer = None
        else:
            self.stop_words = set()
            self.lemmatizer = None
        
    def clean_text(self, text):
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def simple_tokenize(self, text):
        return text.split()
    
    def remove_stopwords(self, text):
        if NLTK_AVAILABLE and self.stop_words:
            try:
                words = word_tokenize(text)
                filtered_words = [word for word in words if word.lower() not in self.stop_words]
                return ' '.join(filtered_words)
            except:
                return text
        else:
            return text
    
    def lemmatize_text(self, text):
        if NLTK_AVAILABLE and self.lemmatizer:
            try:
                words = word_tokenize(text)
                lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
                return ' '.join(lemmatized_words)
            except:
                return text
        else:
            return text
    
    def preprocess_text(self, text):
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize_text(text)
        return text
    
    def preprocess_dataset(self, df, text_column='tweet'):
        df_processed = df.copy()
        df_processed[f'{text_column}_processed'] = df_processed[text_column].apply(self.preprocess_text)
        return df_processed
    
    def extract_features(self, text):
        features = {}
        
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text) if len(text) > 0 else 0
        features['punctuation_ratio'] = sum(1 for c in text if c in self.punctuation) / len(text) if len(text) > 0 else 0
        
        return features
    
    def get_vocabulary(self, texts, min_freq=2):
        word_freq = {}
        
        for text in texts:
            words = text.split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        vocabulary = {word: idx + 1 for idx, (word, freq) in enumerate(word_freq.items()) if freq >= min_freq}
        vocabulary['<UNK>'] = 0
        
        return vocabulary
    
    def text_to_sequence(self, text, vocabulary, max_length=100):
        words = text.split()
        sequence = []
        
        for word in words[:max_length]:
            sequence.append(vocabulary.get(word, vocabulary['<UNK>']))
        
        while len(sequence) < max_length:
            sequence.append(0)
        
        return sequence[:max_length]

def create_embedding_matrix(vocabulary, embedding_dim=100):
    embedding_matrix = np.zeros((len(vocabulary), embedding_dim))
    
    for word, idx in vocabulary.items():
        if word != '<UNK>':
            embedding_matrix[idx] = np.random.normal(0, 0.1, embedding_dim)
    
    return embedding_matrix 