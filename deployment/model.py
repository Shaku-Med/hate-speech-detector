import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

class HateSpeechDetector:
    def __init__(self, max_vocab_size=10000, max_sequence_length=100, embedding_dim=128):
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.vocabulary = None
        self.model = None
        self.history = None
        
    def create_lstm_model(self, num_classes=3):
        model = models.Sequential([
            layers.Embedding(self.max_vocab_size, self.embedding_dim, input_length=self.max_sequence_length),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_cnn_model(self, num_classes=3):
        model = models.Sequential([
            layers.Embedding(self.max_vocab_size, self.embedding_dim, input_length=self.max_sequence_length),
            layers.Conv1D(128, 5, activation='relu'),
            layers.MaxPooling1D(5),
            layers.Conv1D(128, 5, activation='relu'),
            layers.MaxPooling1D(5),
            layers.Conv1D(128, 5, activation='relu'),
            layers.GlobalMaxPooling1D(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_transformer_model(self, num_classes=3):
        inputs = layers.Input(shape=(self.max_sequence_length,))
        embedding = layers.Embedding(self.max_vocab_size, self.embedding_dim)(inputs)
        
        transformer_block = layers.MultiHeadAttention(num_heads=8, key_dim=self.embedding_dim)(embedding, embedding)
        transformer_block = layers.Dropout(0.1)(transformer_block)
        transformer_block = layers.LayerNormalization(epsilon=1e-6)(transformer_block + embedding)
        
        transformer_block = layers.Dense(512, activation='relu')(transformer_block)
        transformer_block = layers.Dropout(0.1)(transformer_block)
        transformer_block = layers.Dense(self.embedding_dim)(transformer_block)
        transformer_block = layers.LayerNormalization(epsilon=1e-6)(transformer_block)
        
        pooled_output = layers.GlobalAveragePooling1D()(transformer_block)
        pooled_output = layers.Dropout(0.3)(pooled_output)
        
        dense = layers.Dense(128, activation='relu')(pooled_output)
        dense = layers.Dropout(0.3)(dense)
        outputs = layers.Dense(num_classes, activation='softmax')(dense)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_data(self, texts, labels):
        from preprocessor import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        
        processed_texts = [preprocessor.preprocess_text(text) for text in texts]
        
        all_words = []
        for text in processed_texts:
            all_words.extend(text.split())
        
        word_freq = {}
        for word in all_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        self.vocabulary = {'<PAD>': 0, '<UNK>': 1}
        
        for word, freq in sorted_words[:self.max_vocab_size - 2]:
            self.vocabulary[word] = len(self.vocabulary)
        
        sequences = []
        for text in processed_texts:
            sequence = []
            words = text.split()
            for word in words:
                sequence.append(self.vocabulary.get(word, self.vocabulary['<UNK>']))
            sequences.append(sequence)
        
        X = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post', truncating='post')
        y = np.array(labels)
        
        return X, y
    
    def train(self, X_train, y_train, X_val, y_val, model_type='lstm', epochs=20, batch_size=32):
        if model_type == 'lstm':
            self.model = self.create_lstm_model()
        elif model_type == 'cnn':
            self.model = self.create_cnn_model()
        elif model_type == 'transformer':
            self.model = self.create_transformer_model()
        else:
            raise ValueError("Model type must be 'lstm', 'cnn', or 'transformer'")
        
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=0.00001
        )
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return self.history
    
    def predict(self, texts):
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if hasattr(self.model, 'predict_proba'):
            predictions = self.model.predict_proba(texts)
            return predictions
        else:
            if not hasattr(self, 'vocabulary') or self.vocabulary is None:
                raise ValueError("Model vocabulary not loaded")
            
            processed_texts = []
            for text in texts:
                from preprocessor import TextPreprocessor
                preprocessor = TextPreprocessor()
                processed_texts.append(preprocessor.preprocess_text(text))
            
            sequences = []
            for text in processed_texts:
                sequence = []
                words = text.split()
                for word in words:
                    sequence.append(self.vocabulary.get(word, self.vocabulary['<UNK>']))
                sequences.append(sequence)
            
            X = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post', truncating='post')
            
            predictions = self.model.predict(X)
            return predictions
    
    def predict_class(self, texts):
        predictions = self.predict(texts)
        return np.argmax(predictions, axis=1)
    
    def evaluate(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred_classes)
        report = classification_report(y_test, y_pred_classes, target_names=['Hate Speech', 'Offensive Language', 'Neither'])
        
        return accuracy, report, y_pred_classes
    
    def plot_training_history(self):
        if self.history is None:
            raise ValueError("No training history available")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath):
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.save(filepath)
        
        import pickle
        with open(filepath.replace('.h5', '_vocab.pkl'), 'wb') as f:
            pickle.dump(self.vocabulary, f)
    
    def load_model(self, filepath):
        if filepath.endswith('.pkl'):
            self.load_pickle_model(filepath)
        else:
            self.model = models.load_model(filepath)
            
            import pickle
            vocab_file = filepath.replace('.h5', '_vocab.pkl')
            if os.path.exists(vocab_file):
                with open(vocab_file, 'rb') as f:
                    self.vocabulary = pickle.load(f)
    
    def load_pickle_model(self, filepath):
        import pickle
        try:
            with open(filepath, 'rb') as f:
                saved_data = pickle.load(f)
            
            if hasattr(saved_data, 'predict_proba'):
                self.model = saved_data
                return
            elif isinstance(saved_data, dict):
                if 'model' in saved_data:
                    self.model = saved_data['model']
                if 'vocabulary' in saved_data:
                    self.vocabulary = saved_data['vocabulary']
                elif 'vocab' in saved_data:
                    self.vocabulary = saved_data['vocab']
                else:
                    raise ValueError("No vocabulary found in pickle file")
            else:
                self.model = saved_data
                raise ValueError("No vocabulary found in pickle file")
                
        except Exception as e:
            raise ValueError(f"Error loading pickle model: {e}") 