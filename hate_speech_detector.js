class HateSpeechDetector {
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
        text = text.replace(/http\S+|www\S+|https\S+/g, '');
        text = text.replace(/@\w+|#\w+/g, '');
        text = text.replace(/[^\w\s]/g, '');
        text = text.replace(/\s+/g, ' ');
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
document.addEventListener('DOMContentLoaded', initializeModel);