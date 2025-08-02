import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from model import HateSpeechDetector
from data_generator import create_enhanced_dataset
import os

def load_and_prepare_data():
    print("Loading and preparing data...")
    
    if not os.path.exists('enhanced_dataset.csv'):
        print("Creating enhanced dataset...")
        df = create_enhanced_dataset()
    else:
        df = pd.read_csv('enhanced_dataset.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:")
    print(df['class'].value_counts())
    
    texts = df['tweet'].values
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
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model(model_type='lstm'):
    print(f"Training {model_type.upper()} model...")
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data()
    
    detector = HateSpeechDetector(
        max_vocab_size=15000,
        max_sequence_length=150,
        embedding_dim=128
    )
    
    print("Preparing data for model...")
    X_train_processed, y_train_processed = detector.prepare_data(X_train, y_train)
    X_val_processed, y_val_processed = detector.prepare_data(X_val, y_val)
    X_test_processed, y_test_processed = detector.prepare_data(X_test, y_test)
    
    print(f"Vocabulary size: {len(detector.vocabulary)}")
    print(f"Input shape: {X_train_processed.shape}")
    
    print("Training model...")
    history = detector.train(
        X_train_processed, y_train_processed,
        X_val_processed, y_val_processed,
        model_type=model_type,
        epochs=25,
        batch_size=64
    )
    
    print("Evaluating model...")
    accuracy, report, y_pred = detector.evaluate(X_test_processed, y_test_processed)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    detector.plot_training_history()
    
    model_filename = f'hate_speech_detector_{model_type}.h5'
    detector.save_model(model_filename)
    print(f"Model saved as {model_filename}")
    
    return detector, accuracy, report

def compare_models():
    print("Comparing different model architectures...")
    
    models_to_test = ['lstm', 'cnn', 'transformer']
    results = {}
    
    for model_type in models_to_test:
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()} model")
        print(f"{'='*50}")
        
        try:
            detector, accuracy, report = train_model(model_type)
            results[model_type] = {
                'accuracy': accuracy,
                'report': report,
                'detector': detector
            }
        except Exception as e:
            print(f"Error training {model_type} model: {e}")
            results[model_type] = None
    
    print("\n" + "="*50)
    print("MODEL COMPARISON RESULTS")
    print("="*50)
    
    for model_type, result in results.items():
        if result is not None:
            print(f"{model_type.upper()}: {result['accuracy']:.4f}")
        else:
            print(f"{model_type.upper()}: Failed to train")
    
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'] if x[1] is not None else 0)
    print(f"\nBest model: {best_model[0].upper()} with accuracy {best_model[1]['accuracy']:.4f}")
    
    return results

def analyze_predictions(detector, X_test, y_test):
    print("Analyzing model predictions...")
    
    predictions = detector.predict(X_test)
    y_pred_classes = np.argmax(predictions, axis=1)
    
    cm = confusion_matrix(y_test, y_pred_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Hate Speech', 'Offensive Language', 'Neither'],
                yticklabels=['Hate Speech', 'Offensive Language', 'Neither'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    class_names = ['Hate Speech', 'Offensive Language', 'Neither']
    
    for i, class_name in enumerate(class_names):
        class_indices = np.where(y_test == i)[0]
        if len(class_indices) > 0:
            class_predictions = predictions[class_indices]
            class_confidences = np.max(class_predictions, axis=1)
            
            plt.figure(figsize=(10, 6))
            plt.hist(class_confidences, bins=20, alpha=0.7, edgecolor='black')
            plt.title(f'Prediction Confidence Distribution - {class_name}')
            plt.xlabel('Confidence')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'confidence_distribution_{class_name.lower().replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()

def main():
    print("Hate Speech Detector Training Pipeline")
    print("="*50)
    
    choice = input("Choose training mode:\n1. Train single model (LSTM)\n2. Compare all models\nEnter choice (1 or 2): ")
    
    if choice == '1':
        detector, accuracy, report = train_model('lstm')
        
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data()
        analyze_predictions(detector, X_test, y_test)
        
    elif choice == '2':
        results = compare_models()
        
        best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'] if x[1] is not None else 0)[0]
        best_detector = results[best_model_name]['detector']
        
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data()
        analyze_predictions(best_detector, X_test, y_test)
        
    else:
        print("Invalid choice. Running default LSTM training...")
        detector, accuracy, report = train_model('lstm')

if __name__ == "__main__":
    main() 