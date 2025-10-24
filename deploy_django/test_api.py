#!/usr/bin/env python3
"""
Simple test script for the Theft Detection REST API
"""
import requests
import json
import sys
import os

API_BASE_URL = 'http://localhost:8000/api'

def check_health():
    """Check API health status"""
    print("=" * 60)
    print("HEALTH CHECK")
    print("=" * 60)
    
    try:
        response = requests.get(f'{API_BASE_URL}/health/', timeout=5)
        health = response.json()
        
        print(f"Status Code: {response.status_code}")
        print(f"API Status: {health['status']}")
        print(f"Model Loaded: {health['model_loaded']}")
        print(f"Message: {health['message']}")
        print("=" * 60)
        print()
        
        return health['model_loaded']
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to API. Is the server running?")
        print("Start server with: python3 manage.py runserver 0.0.0.0:8000")
        print("=" * 60)
        return False
    except Exception as e:
        print(f"ERROR: {str(e)}")
        print("=" * 60)
        return False

def predict_video(video_path):
    """Predict shoplifting in a video"""
    print("=" * 60)
    print("VIDEO PREDICTION")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"File size: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
    print()
    
    try:
        print("Uploading video...")
        with open(video_path, 'rb') as f:
            files = {'video': f}
            response = requests.post(
                f'{API_BASE_URL}/predict/', 
                files=files,
                timeout=60  # 60 seconds timeout for processing
            )
        
        print(f"Status Code: {response.status_code}")
        print()
        
        result = response.json()
        
        # Pretty print the JSON response
        print("JSON Response:")
        print("-" * 60)
        print(json.dumps(result, indent=2))
        print("-" * 60)
        print()
        
        if result.get('success'):
            print("ANALYSIS RESULTS:")
            print(f"  Prediction: {result['label']}")
            print(f"  Confidence: {result['confidence']:.2f}%")
            print(f"  Probability: {result['probability']:.4f}")
            print(f"  Is Theft: {result['is_theft']}")
            
            if result.get('message'):
                print(f"  Message: {result['message']}")
            
            if result['is_theft'] and result.get('suspicious_frames'):
                print(f"\n  Suspicious Frames ({len(result['suspicious_frames'])}):")
                for i, frame_url in enumerate(result['suspicious_frames'], 1):
                    full_url = f"http://localhost:8000{frame_url}"
                    print(f"    {i}. {full_url}")
        else:
            print(f"PREDICTION FAILED:")
            print(f"  Error: {result.get('error', 'Unknown error')}")
        
        print("=" * 60)
        print()
        return result
        
    except FileNotFoundError:
        print(f"ERROR: Video file not found: {video_path}")
        print("=" * 60)
        return None
    except requests.exceptions.Timeout:
        print("ERROR: Request timed out. Video may be too large or processing too slow.")
        print("=" * 60)
        return None
    except Exception as e:
        print(f"ERROR: {str(e)}")
        print("=" * 60)
        return None

def main():
    """Main test function"""
    print("\nTheft Detection REST API - Test Script")
    print()
    
    # Check health first
    if not check_health():
        print("⚠️  API is not ready. Please check the server and try again.")
        sys.exit(1)
    
    print("✓ API is healthy and ready!")
    print()
    
    # If video path provided, test prediction
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        
        if not os.path.exists(video_path):
            print(f"Error: File not found: {video_path}")
            sys.exit(1)
        
        result = predict_video(video_path)
        
        if result and result.get('success'):
            print("✓ Prediction completed successfully!")
            sys.exit(0)
        else:
            print("✗ Prediction failed.")
            sys.exit(1)
    else:
        print("Health check passed!")
        print()
        print("To test video prediction, run:")
        print(f"  python3 {sys.argv[0]} <path_to_video.mp4>")
        print()
        print("Example:")
        print(f"  python3 {sys.argv[0]} ../Shop\\ Dataset/shop\\ lifters/video1.mp4")

if __name__ == '__main__':
    main()
