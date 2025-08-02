import requests
import json
import time

def test_api():
    base_url = "http://localhost:5000"
    
    print("Testing Hate Speech Detection API...")
    
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    print("\n2. Testing single prediction...")
    try:
        data = {"text": "This is a test message"}
        response = requests.post(f"{base_url}/predict", json=data)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    print("\n3. Testing batch prediction...")
    try:
        data = {"texts": ["Test message 1", "Test message 2", "Hello world"]}
        response = requests.post(f"{base_url}/predict-batch", json=data)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    print("\nAll tests completed!")
    return True

if __name__ == "__main__":
    test_api() 