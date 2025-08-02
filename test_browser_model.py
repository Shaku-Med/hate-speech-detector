import json
import os

def test_browser_model():
    print("Testing Browser Model")
    print("=" * 30)
    
    if not os.path.exists('browser_model/model_info.json'):
        print("Browser model not found. Please run browser_converter.py first.")
        return
    
    with open('browser_model/model_info.json', 'r') as f:
        model_info = json.load(f)
    
    print("Model Info:")
    print(f"- Vocabulary size: {len(model_info['vocabulary'])}")
    print(f"- Max features: {model_info['max_features']}")
    print(f"- N-gram range: {model_info['ngram_range']}")
    print(f"- Number of classes: {model_info['n_classes']}")
    print(f"- Class names: {model_info['class_names']}")
    print(f"- Feature importances: {len(model_info['feature_importances'])} features")
    
    print("\nSample vocabulary entries:")
    vocab_items = list(model_info['vocabulary'].items())[:10]
    for word, index in vocab_items:
        print(f"  '{word}': {index}")
    
    print("\nSample feature importances:")
    for i, importance in enumerate(model_info['feature_importances'][:10]):
        print(f"  Feature {i}: {importance:.6f}")
    
    print("\nBrowser model is ready!")
    print("Open http://localhost:8000 in your browser to test it.")

if __name__ == "__main__":
    test_browser_model() 