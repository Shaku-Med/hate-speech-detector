# Hate Speech Detector AI Module

A professional hate speech detection system using machine learning with multiple model architectures and comprehensive evaluation tools.

## Features

- **Multiple Model Types**: Random Forest, Logistic Regression, and SVM classifiers
- **Enhanced Dataset Generation**: Creates synthetic and contextual data for better training
- **Professional Text Preprocessing**: Advanced NLP techniques for text cleaning
- **Comprehensive Evaluation**: Detailed performance analysis and visualization
- **Easy-to-Use Interface**: Command-line tools for training and prediction
- **Production Ready**: Save/load models for deployment

## Quick Start

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk textblob
```

### 2. Train the Model
```bash
python simple_model.py
```

### 3. Make Predictions
```bash
# Interactive mode
python simple_predict.py --interactive

# Single text prediction
python simple_predict.py --text "Your text here"

# Quick demo with examples
python quick_demo.py
```

## File Structure

```
Python/HateSpeech/
├── requirements.txt                    # Dependencies
├── simple_model.py                     # Scikit-learn based models
├── simple_predict.py                   # Prediction interface
├── quick_demo.py                       # Demo script
├── data_generator.py                   # Dataset generation (TensorFlow version)
├── preprocessor.py                     # Text preprocessing utilities
├── model.py                           # TensorFlow models (advanced)
├── train.py                           # TensorFlow training pipeline
├── predict.py                         # TensorFlow prediction interface
├── evaluate.py                        # Model evaluation and analysis
├── README.md                          # This file
├── dataset_tweet.csv                  # Original dataset
├── simple_hate_speech_detector.pkl    # Trained scikit-learn model
└── enhanced_dataset.csv               # Enhanced dataset (generated)
```

## Model Performance

The current scikit-learn model achieves:
- **Overall Accuracy**: 89.08%
- **Hate Speech Detection**: 50% precision, 13% recall
- **Offensive Language Detection**: 91% precision, 96% recall
- **Neutral Text Detection**: 81% precision, 85% recall

## Usage Examples

### Training Different Models
```python
from simple_model import SimpleHateSpeechDetector

# Random Forest (default)
detector = SimpleHateSpeechDetector(model_type='random_forest')

# Logistic Regression
detector = SimpleHateSpeechDetector(model_type='logistic_regression')

# Support Vector Machine
detector = SimpleHateSpeechDetector(model_type='svm')
```

### Making Predictions
```python
from simple_model import SimpleHateSpeechDetector

# Load trained model
detector = SimpleHateSpeechDetector()
detector.load_model('simple_hate_speech_detector.pkl')

# Single prediction
result = detector.predict_single("Your text here")
print(f"Class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.4f}")

# Batch prediction
texts = ["Text 1", "Text 2", "Text 3"]
predictions, probabilities = detector.predict(texts)
```

## Classification Categories

The system classifies text into three categories:
- **Hate Speech**: Content that promotes violence or discrimination against groups
- **Offensive Language**: Profanity, insults, or inappropriate content
- **Neither**: Normal, non-offensive content

## Example Predictions

| Text | Prediction | Confidence |
|------|------------|------------|
| "I love this beautiful day!" | Neither | 50.5% |
| "Fuck you, you piece of shit" | Offensive Language | 90.0% |
| "Kill all the Jews" | Hate Speech | 67.1% |
| "The weather is nice today" | Neither | 69.9% |

## Advanced Features (TensorFlow Version)

For more advanced models using TensorFlow:

1. Install TensorFlow:
```bash
pip install tensorflow
```

2. Use the advanced training pipeline:
```bash
python train.py
```

3. Advanced prediction:
```bash
python predict.py --interactive
```

## Model Architectures Available

### Scikit-learn Models (Current)
- **Random Forest**: Ensemble method with good generalization
- **Logistic Regression**: Linear model with interpretable results
- **Support Vector Machine**: Effective for high-dimensional data

### TensorFlow Models (Advanced)
- **LSTM**: Bidirectional LSTM for sequential text understanding
- **CNN**: Convolutional layers for local pattern detection
- **Transformer**: Multi-head attention for state-of-the-art performance

## Data Enhancement

The system includes data generation capabilities:
- **Synthetic Data**: Generated using hate speech, offensive, and neutral phrase templates
- **Contextual Data**: Context-aware hate speech patterns with demographic groups
- **Original Data**: Twitter dataset with manual annotations

## Performance Optimization

- Use GPU acceleration for TensorFlow models
- Adjust batch size based on available memory
- Use early stopping to prevent overfitting
- Experiment with different model architectures

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce vocabulary size or batch size
2. **Slow Training**: Use smaller models or fewer epochs
3. **Poor Performance**: Increase dataset size or try different model architecture
4. **NLTK Errors**: Ensure NLTK data is downloaded

### Installation Issues

If you encounter issues with TensorFlow installation:
1. Use the scikit-learn version (simple_model.py)
2. Install TensorFlow CPU version: `pip install tensorflow-cpu`
3. Use conda for better dependency management

## Contributing

To enhance the system:
1. Add new model architectures in `simple_model.py`
2. Implement additional preprocessing techniques in `preprocessor.py`
3. Create new dataset generators in `data_generator.py`
4. Add evaluation metrics in `evaluate.py`

## License

This project is for educational and research purposes. Please ensure compliance with data usage and privacy regulations when deploying in production environments.

## Current Status

✅ **Working Implementation**: Scikit-learn based hate speech detector
✅ **Trained Model**: 89% accuracy on test data
✅ **Interactive Interface**: Command-line prediction tool
✅ **Demo Script**: Quick testing with example texts
🔄 **Advanced Models**: TensorFlow implementation available (requires TensorFlow installation) 