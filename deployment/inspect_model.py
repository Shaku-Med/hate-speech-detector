import pickle
import os

def inspect_pickle_file():
    model_path = 'simple_hate_speech_detector.pkl'
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        return
    
    print(f"Model file found: {model_path}")
    print(f"File size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"\nData type: {type(data)}")
        
        if isinstance(data, dict):
            print("Keys in the dictionary:")
            for key in data.keys():
                print(f"  - {key}: {type(data[key])}")
                if hasattr(data[key], 'shape'):
                    print(f"    Shape: {data[key].shape}")
        else:
            print(f"Data is not a dictionary, it's a {type(data)}")
            if hasattr(data, 'shape'):
                print(f"Shape: {data.shape}")
        
    except Exception as e:
        print(f"Error reading pickle file: {e}")

if __name__ == "__main__":
    inspect_pickle_file() 