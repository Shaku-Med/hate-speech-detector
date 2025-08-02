import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def create_browser_model():
    print("Creating Browser-Compatible Hate Speech Detector")
    print("=" * 50)
    
    try:
        with open('simple_hate_speech_detector.pkl', 'rb') as f:
            sklearn_pipeline = pickle.load(f)
        print("Loaded existing scikit-learn model")
    except:
        print("Training new model...")
        from simple_model import train_simple_model
        train_simple_model()
        with open('simple_hate_speech_detector.pkl', 'rb') as f:
            sklearn_pipeline = pickle.load(f)
    
    vectorizer = sklearn_pipeline.named_steps['vectorizer']
    classifier = sklearn_pipeline.named_steps['classifier']
    
    print("Extracting model information...")
    
    model_info = {
        'vocabulary': {str(k): int(v) for k, v in vectorizer.vocabulary_.items()},
        'max_features': int(vectorizer.max_features),
        'ngram_range': list(vectorizer.ngram_range),
        'stop_words': list(vectorizer.stop_words) if vectorizer.stop_words else None,
        'feature_importances': [float(x) for x in classifier.feature_importances_],
        'n_classes': int(len(classifier.classes_)),
        'class_names': ['Hate Speech', 'Offensive Language', 'Neither']
    }
    
    if not os.path.exists('browser_model'):
        os.makedirs('browser_model')
    
    with open('browser_model/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("Model information saved to browser_model/model_info.json")
    
    create_browser_files()
    
    return model_info

def create_browser_files():
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hate Speech Detector</title>
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
        .warning {
            background: #fff3cd;
            color: #856404;
            border-left-color: #ffc107;
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
        <div id="loading" class="loading" style="display: none;">Analyzing text...</div>
    </div>

    <script src="hate_speech_detector.js"></script>
</body>
</html>'''
    
    js_content = '''class HateSpeechDetector {
    constructor() {
        this.modelInfo = null;
        this.classNames = ['Hate Speech', 'Offensive Language', 'Neither'];
        this.isLoaded = false;
    }

    async loadModel() {
        try {
            console.log('Loading model information...');
            const response = await fetch('browser_model/model_info.json');
            this.modelInfo = await response.json();
            
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
        const ngrams = this.generateNgrams(words, this.modelInfo.ngram_range);
        
        const vector = new Array(this.modelInfo.max_features).fill(0);
        
        ngrams.forEach(ngram => {
            const ngramStr = ngram.join(' ');
            if (this.modelInfo.vocabulary[ngramStr] !== undefined) {
                const index = this.modelInfo.vocabulary[ngramStr];
                if (index < this.modelInfo.max_features) {
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

    predict(text) {
        if (!this.isLoaded) {
            throw new Error('Model not loaded');
        }

        const vector = this.vectorizeText(text);
        
        let maxScore = -Infinity;
        let predictedClass = 0;
        const scores = [];
        
        for (let classIdx = 0; classIdx < this.modelInfo.n_classes; classIdx++) {
            let score = 0;
            for (let featureIdx = 0; featureIdx < vector.length; featureIdx++) {
                if (vector[featureIdx] > 0) {
                    score += this.modelInfo.feature_importances[featureIdx] * vector[featureIdx];
                }
            }
            scores.push(score);
            if (score > maxScore) {
                maxScore = score;
                predictedClass = classIdx;
            }
        }
        
        const totalScore = scores.reduce((a, b) => a + b, 0);
        const probabilities = scores.map(score => Math.max(0, score / totalScore));
        const confidence = Math.max(...probabilities);
        
        return {
            text: text,
            predictedClass: this.classNames[predictedClass],
            confidence: confidence,
            probabilities: {
                hate_speech: probabilities[0] || 0,
                offensive_language: probabilities[1] || 0,
                neither: probabilities[2] || 0
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
        const result = detector.predict(text);
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
    
    let resultClass = 'success';
    if (result.predictedClass === 'Hate Speech') {
        resultClass = 'error';
    } else if (result.predictedClass === 'Offensive Language') {
        resultClass = 'warning';
    }
    
    resultDiv.innerHTML = `
        <div class="result ${resultClass}">
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
    print("Creating Browser-Compatible Hate Speech Detector")
    print("=" * 50)
    
    model_info = create_browser_model()
    
    print("\n" + "=" * 50)
    print("BROWSER CONVERSION COMPLETE!")
    print("=" * 50)
    print("Files created:")
    print("- browser_model/model_info.json (Model data)")
    print("- index.html (Web interface)")
    print("- hate_speech_detector.js (JavaScript code)")
    print("\nTo run in browser:")
    print("1. Start a local server: python -m http.server 8000")
    print("2. Open http://localhost:8000 in your browser")
    print("3. The model will run entirely in your browser!")
    print("\nFeatures:")
    print("- No server required after initial load")
    print("- Works offline")
    print("- Fast predictions")
    print("- Beautiful UI")

if __name__ == "__main__":
    main() 