#!/usr/bin/env python3
"""Test script to verify model loading"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("Testing Model Loading")
print("=" * 60)

try:
    from theft_detector.utils import load_model, MODEL_PATH
    print(f"\n1. Model path: {MODEL_PATH}")
    print(f"   Exists: {os.path.exists(MODEL_PATH)}")
    
    print("\n2. Loading model...")
    model = load_model()
    print("   ✓ Model loaded successfully!")
    
    print(f"\n3. Model details:")
    print(f"   - Device: {next(model.parameters()).device}")
    print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
