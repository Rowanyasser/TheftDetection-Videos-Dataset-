#!/usr/bin/env python3
"""
Test script to verify the deployment is working.
Checks: imports, model file, Django settings, and basic functionality.
"""
import os
import sys

def test_imports():
    """Test that all required packages are installed"""
    print("Testing imports...")
    try:
        import django
        import torch
        import cv2
        import numpy as np
        from torchvision.models.video import r3d_18
        print("✓ All imports successful")
        print(f"  - Django version: {django.VERSION}")
        print(f"  - PyTorch version: {torch.__version__}")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_model_file():
    """Check if model checkpoint exists"""
    print("\nTesting model file...")
    model_path = os.path.join(os.path.dirname(__file__), '..', 'best_model.pth')
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✓ Model file found: {model_path}")
        print(f"  - Size: {size_mb:.2f} MB")
        return True
    else:
        print(f"✗ Model file not found at: {model_path}")
        print("  Please ensure best_model.pth is in /mnt/d/Cellula_Internship/Task3/")
        return False

def test_django_setup():
    """Test Django configuration"""
    print("\nTesting Django setup...")
    try:
        # Add current directory to path
        sys.path.insert(0, os.path.dirname(__file__))
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'theft_detector.settings')
        import django
        django.setup()
        from django.conf import settings
        print("✓ Django configured successfully")
        print(f"  - DEBUG: {settings.DEBUG}")
        print(f"  - ALLOWED_HOSTS: {settings.ALLOWED_HOSTS}")
        return True
    except Exception as e:
        print(f"✗ Django setup error: {e}")
        return False

def test_directories():
    """Check required directories exist"""
    print("\nTesting directory structure...")
    dirs = ['templates', 'static', 'static/css', 'media']
    all_exist = True
    for d in dirs:
        path = os.path.join(os.path.dirname(__file__), d)
        if os.path.exists(path):
            print(f"✓ {d}/ exists")
        else:
            print(f"✗ {d}/ missing")
            all_exist = False
    return all_exist

def main():
    print("=" * 60)
    print("Theft Detection Deployment Test")
    print("=" * 60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Model File", test_model_file()))
    results.append(("Django Setup", test_django_setup()))
    results.append(("Directories", test_directories()))
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("\n✓ All tests passed! You can run the server with:")
        print("\n  export DJANGO_SETTINGS_MODULE=theft_detector.settings")
        print("  python3 manage.py runserver 0.0.0.0:8000")
        print("\nThen visit: http://localhost:8000\n")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix the issues above.\n")
        return 1

if __name__ == '__main__':
    sys.exit(main())
