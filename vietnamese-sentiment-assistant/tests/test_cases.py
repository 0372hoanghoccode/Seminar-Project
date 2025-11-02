#!/usr/bin/env python3
"""
Test cases cho Vietnamese Sentiment Classifier
Yêu cầu: Độ chính xác >=65% (10 test cases)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sentiment_classifier import SentimentClassifier

def test_sentiment_classifier():
    """Test classifier với 10 test cases"""

    # Khởi tạo classifier
    classifier = SentimentClassifier()

    # Test cases (text, expected_sentiment)
    test_cases = [
        ("Hôm nay tôi rất vui", "POSITIVE"),
        ("Món ăn này dở quá", "NEGATIVE"),
        ("Thời tiết bình thường", "NEUTRAL"),
        ("Rất vui hôm nay", "POSITIVE"),
        ("Công việc ổn định", "NEUTRAL"),
        ("Phim này hay lắm", "POSITIVE"),
        ("Tôi buồn vì thất bại", "NEGATIVE"),
        ("Ngày mai đi học", "NEUTRAL"),
        ("Cảm ơn bạn rất nhiều", "POSITIVE"),
        ("Mệt mỏi quá hôm nay", "NEGATIVE")
    ]

    print("=" * 60)
    print("TESTING SENTIMENT CLASSIFIER")
    print("=" * 60)

    correct = 0
    total = len(test_cases)

    for i, (text, expected) in enumerate(test_cases, 1):
        try:
            result = classifier.classify(text)
            predicted = result['sentiment']

            status = "✓" if predicted == expected else "✗"
            print(f"{status} Text: {text}")
            print(f"   Predicted: {predicted} | Expected: {expected}")
            print()

            if predicted == expected:
                correct += 1

        except Exception as e:
            print(f"✗ Error with text: {text}")
            print(f"   Error: {e}")
            print()

    accuracy = (correct / total) * 100
    print("=" * 60)
    print(f"ACCURACY: {accuracy:.2f}% ({correct}/{total})")
    print(f"REQUIREMENT: ≥65% ✓ PASSED" if accuracy >= 65 else f"REQUIREMENT: ≥65% ✗ FAILED")
    print("=" * 60)

    return accuracy >= 65

if __name__ == "__main__":
    success = test_sentiment_classifier()
    sys.exit(0 if success else 1)
