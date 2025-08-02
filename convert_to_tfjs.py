import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pickle
import os
import json

class TFJSConverter:
    def __init__(self):
        self.vectorizer = None
        self.classifier = None
        self.class_names = ['Hate Speech', 'Offensive Language', 'Neither']
        
    def create_tf_model_from_sklearn(self, sklearn_pipeline):
        vectorizer = sklearn_pipeline.named_steps['vectorizer']
        classifier = sklearn_pipeline.named_steps['classifier']
        
        self.vectorizer = vectorizer
        self.classifier = classifier
        
        vocab_size = len(vectorizer.vocabulary_)
        max_features = vectorizer.max_features
        
        model = keras.Sequential([
            keras.layers.Input(shape=(max_features,), name='input_text'),
            keras.layers.Dense(128, activation='relu', name='dense_1'),
            keras.layers.Dropout(0.3, name='dropout_1'),
            keras.layers.Dense(64, activation='relu', name='dense_2'),
            keras.layers.Dropout(0.2, name='dropout_2'),
            keras.layers.Dense(3, activation='softmax', name='output')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_tf_model_with_sklearn_weights(self, tf_model, sklearn_pipeline, X_train, y_train):
        vectorizer = sklearn_pipeline.named_steps['vectorizer']
        classifier = sklearn_pipeline.named_steps['classifier']
        
        X_train_vectorized = vectorizer.transform(X_train)
        
        tf_model.fit(
            X_train_vectorized.toarray(),
            y_train,
            epochs=10,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        return tf_model
    
    def save_vectorizer_info(self, vectorizer, save_path):
        vectorizer_info = {
            'vocabulary': vectorizer.vocabulary_,
            'idf': vectorizer.idf_.tolist(),
            'max_features': vectorizer.max_features,
            'ngram_range': vectorizer.ngram_range,
            'stop_words': list(vectorizer.stop_words) if vectorizer.stop_words else None
        }
        
        with open(save_path, 'w') as f:
            json.dump(vectorizer_info, f, indent=2)
    
    def convert_and_save(self, sklearn_model_path, output_dir='tfjs_model'):
        print("Loading scikit-learn model...")
        
        with open(sklearn_model_path, 'rb') as f:
            sklearn_pipeline = pickle.load(f)
        
        print("Creating TensorFlow model...")
        tf_model = self.create_tf_model_from_sklearn(sklearn_pipeline)
        
        print("Preparing training data...")
        try:
            df = pd.read_csv('dataset_tweet.csv')
            texts = df['tweet'].values
            labels = df['class'].values
        except:
            print("Using synthetic data for conversion...")
            from simple_model import create_simple_datasets
            df = create_simple_datasets()
            texts = df['text'].values
            labels = df['class'].values
        
        X_train = texts[:1000]
        y_train = labels[:1000]
        
        print("Training TensorFlow model...")
        tf_model = self.train_tf_model_with_sklearn_weights(tf_model, sklearn_pipeline, X_train, y_train)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print("Saving TensorFlow.js model...")
        tf_model.save(f'{output_dir}/model')
        
        print("Saving vectorizer information...")
        self.save_vectorizer_info(sklearn_pipeline.named_steps['vectorizer'], f'{output_dir}/vectorizer_info.json')
        
        print("Converting to TensorFlow.js format...")
        import tensorflowjs as tfjs
        tfjs.converters.save_keras_model(tf_model, f'{output_dir}/tfjs_model')
        
        print(f"Model converted and saved to {output_dir}/")
        return tf_model

def create_browser_files():
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hate Speech Detector</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.15.0/dist/tf.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }
        .input-section {
            margin-bottom: 30px;
        }
        textarea {
            width: 100%;
            height: 100px;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 10px;
            font-size: 16px;
            resize: vertical;
            font-family: inherit;
        }
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        button {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 25px;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s;
            margin: 10px 5px;
        }
        button:hover {
            transform: translateY(-2px);
        }
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        .result {
            margin-top: 20px;
            padding: 20px;
            border-radius: 10px;
            background: #f8f9fa;
            border-left: 4px solid #667eea;
        }
        .confidence-bar {
            background: #e9ecef;
            border-radius: 10px;
            height: 20px;
            margin: 5px 0;
            overflow: hidden;
        }
        .confidence-fill {
            height: 100%;
            background: linear-gradient(45deg, #667eea, #764ba2);
            transition: width 0.3s ease;
        }
        .loading {
            text-align: center;
            color: #666;
        }
        .error {
            background: #f8d7da;
            color: #721c24;
            border-left-color: #dc3545;
        }
        .success {
            background: #d4edda;
            color: #155724;
            border-left-color: #28a745;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Hate Speech Detector</h1>
        
        <div class="input-section">
            <textarea id="textInput" placeholder="Enter text to analyze for hate speech, offensive language, or neutral content..."></textarea>
            <br>
            <button onclick="analyzeText()" id="analyzeBtn">Analyze Text</button>
            <button onclick="clearText()">Clear</button>
        </div>
        
        <div id="result" style="display: none;"></div>
        <div id="loading" class="loading" style="display: none;">Loading model and analyzing...</div>
    </div>

    <script src="hate_speech_detector.js"></script>
</body>
</html>'''
    
    js_content = '''class HateSpeechDetector {
    constructor() {
        this.model = null;
        this.vectorizer = null;
        this.classNames = ['Hate Speech', 'Offensive Language', 'Neither'];
        this.isLoaded = false;
    }

    async loadModel() {
        try {
            console.log('Loading TensorFlow.js model...');
            this.model = await tf.loadLayersModel('tfjs_model/model.json');
            
            console.log('Loading vectorizer information...');
            const response = await fetch('tfjs_model/vectorizer_info.json');
            this.vectorizer = await response.json();
            
            this.isLoaded = true;
            console.log('Model loaded successfully!');
            return true;
        } catch (error) {
            console.error('Error loading model:', error);
            return false;
        }
    }

    preprocessText(text) {
        text = text.toLowerCase();
        text = text.replace(/http\\S+|www\\S+|https\\S+/g, '');
        text = text.replace(/@\\w+|#\\w+/g, '');
        text = text.replace(/[^\\w\\s]/g, '');
        text = text.replace(/\\s+/g, ' ');
        return text.trim();
    }

    vectorizeText(text) {
        const processedText = this.preprocessText(text);
        const words = processedText.split(' ');
        const ngrams = this.generateNgrams(words, this.vectorizer.ngram_range);
        
        const vector = new Array(this.vectorizer.max_features).fill(0);
        
        ngrams.forEach(ngram => {
            const ngramStr = ngram.join(' ');
            if (this.vectorizer.vocabulary[ngramStr] !== undefined) {
                const index = this.vectorizer.vocabulary[ngramStr];
                if (index < this.vectorizer.max_features) {
                    vector[index] = 1;
                }
            }
        });
        
        return vector;
    }

    generateNgrams(words, ngramRange) {
        const ngrams = [];
        const [minN, maxN] = ngramRange;
        
        for (let n = minN; n <= maxN; n++) {
            for (let i = 0; i <= words.length - n; i++) {
                ngrams.push(words.slice(i, i + n));
            }
        }
        
        return ngrams;
    }

    async predict(text) {
        if (!this.isLoaded) {
            throw new Error('Model not loaded');
        }

        const vector = this.vectorizeText(text);
        const tensor = tf.tensor2d([vector]);
        
        const prediction = await this.model.predict(tensor).array();
        tensor.dispose();
        
        const probabilities = prediction[0];
        const predictedClass = probabilities.indexOf(Math.max(...probabilities));
        
        return {
            text: text,
            predictedClass: this.classNames[predictedClass],
            confidence: Math.max(...probabilities),
            probabilities: {
                hate_speech: probabilities[0],
                offensive_language: probabilities[1],
                neither: probabilities[2]
            }
        };
    }
}

let detector = new HateSpeechDetector();
let modelLoaded = false;

async function initializeModel() {
    const loadingDiv = document.getElementById('loading');
    loadingDiv.style.display = 'block';
    loadingDiv.textContent = 'Loading model...';
    
    try {
        modelLoaded = await detector.loadModel();
        if (modelLoaded) {
            loadingDiv.style.display = 'none';
            console.log('Model initialized successfully');
        } else {
            throw new Error('Failed to load model');
        }
    } catch (error) {
        loadingDiv.style.display = 'none';
        showError('Failed to load model: ' + error.message);
    }
}

async function analyzeText() {
    const textInput = document.getElementById('textInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loadingDiv = document.getElementById('loading');
    const resultDiv = document.getElementById('result');
    
    const text = textInput.value.trim();
    
    if (!text) {
        showError('Please enter some text to analyze');
        return;
    }
    
    if (!modelLoaded) {
        showError('Model not loaded yet. Please wait...');
        return;
    }
    
    analyzeBtn.disabled = true;
    loadingDiv.style.display = 'block';
    resultDiv.style.display = 'none';
    
    try {
        const result = await detector.predict(text);
        showResult(result);
    } catch (error) {
        showError('Error analyzing text: ' + error.message);
    } finally {
        analyzeBtn.disabled = false;
        loadingDiv.style.display = 'none';
    }
}

function showResult(result) {
    const resultDiv = document.getElementById('result');
    
    const confidenceBars = Object.entries(result.probabilities)
        .map(([key, value]) => `
            <div>
                <strong>${key.replace('_', ' ').toUpperCase()}:</strong> ${(value * 100).toFixed(1)}%
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${value * 100}%"></div>
                </div>
            </div>
        `).join('');
    
    resultDiv.innerHTML = `
        <div class="result ${result.predictedClass === 'Neither' ? 'success' : 'error'}">
            <h3>Analysis Result</h3>
            <p><strong>Text:</strong> ${result.text}</p>
            <p><strong>Prediction:</strong> ${result.predictedClass}</p>
            <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
            <h4>Probabilities:</h4>
            ${confidenceBars}
        </div>
    `;
    
    resultDiv.style.display = 'block';
}

function showError(message) {
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = `
        <div class="result error">
            <h3>Error</h3>
            <p>${message}</p>
        </div>
    `;
    resultDiv.style.display = 'block';
}

function clearText() {
    document.getElementById('textInput').value = '';
    document.getElementById('result').style.display = 'none';
}

// Initialize model when page loads
document.addEventListener('DOMContentLoaded', initializeModel);'''
    
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    with open('hate_speech_detector.js', 'w', encoding='utf-8') as f:
        f.write(js_content)
    
    print("Browser files created: index.html and hate_speech_detector.js")

def main():
    print("Converting Hate Speech Detector to TensorFlow.js")
    print("=" * 50)
    
    converter = TFJSConverter()
    
    if not os.path.exists('simple_hate_speech_detector.pkl'):
        print("Training model first...")
        from simple_model import train_simple_model
        train_simple_model()
    
    print("Converting model to TensorFlow.js format...")
    converter.convert_and_save('simple_hate_speech_detector.pkl')
    
    print("Creating browser files...")
    create_browser_files()
    
    print("\n" + "=" * 50)
    print("CONVERSION COMPLETE!")
    print("=" * 50)
    print("Files created:")
    print("- tfjs_model/ (TensorFlow.js model files)")
    print("- index.html (Web interface)")
    print("- hate_speech_detector.js (JavaScript code)")
    print("\nTo run in browser:")
    print("1. Start a local server: python -m http.server 8000")
    print("2. Open http://localhost:8000 in your browser")
    print("3. The model will run entirely in your browser!")

if __name__ == "__main__":
    main() 