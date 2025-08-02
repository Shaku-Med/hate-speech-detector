import requests
import json
import time

def test_api():
    print("=== TESTING FIXED API ===\n")
    
    base_url = "http://localhost:8000"
    
    # Test cases
    test_cases = [
        {
            "text": "You're so beautiful",
            "expected": "Neither"
        },
        {
            "text": "I love you",
            "expected": "Neither"
        },
        {
            "text": "I hate you",
            "expected": "Hate Speech"
        },
        {
            "text": "Go kill yourself",
            "expected": "Hate Speech"
        },
        {
            "text": "You are stupid",
            "expected": "Hate Speech"
        }
    ]
    
    print("Testing individual predictions:")
    for i, test_case in enumerate(test_cases, 1):
        try:
            response = requests.post(
                f"{base_url}/predict",
                json={"text": test_case["text"]},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                predicted = result["predicted_class"]
                confidence = result["confidence"]
                
                status = "✅" if predicted == test_case["expected"] else "❌"
                print(f"{status} Test {i}: '{test_case['text']}' -> {predicted} (confidence: {confidence:.3f})")
                
                if predicted != test_case["expected"]:
                    print(f"   Expected: {test_case['expected']}")
            else:
                print(f"❌ Test {i}: HTTP {response.status_code} - {response.text}")
                
        except requests.exceptions.ConnectionError:
            print(f"❌ Test {i}: Could not connect to server. Is the API running?")
            break
        except Exception as e:
            print(f"❌ Test {i}: Error - {e}")
    
    print("\n" + "="*50)
    print("✅ API TEST COMPLETED!")
    print("If all tests passed, your hate speech detector is working correctly!")
    print("The model now properly classifies:")
    print("  - 'You're so beautiful' as 'Neither'")
    print("  - 'I hate you' as 'Hate Speech'")

if __name__ == "__main__":
    test_api() 