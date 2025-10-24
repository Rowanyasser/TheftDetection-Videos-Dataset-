#!/usr/bin/env python3
"""Inspect the checkpoint structure"""
import torch

checkpoint_path = "/mnt/d/Cellula_Internship/Task3/best_model.pth"
print("=" * 70)
print("Checkpoint Structure Analysis")
print("=" * 70)

checkpoint = torch.load(checkpoint_path, map_location='cpu')

print(f"\nCheckpoint type: {type(checkpoint)}")

if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    print("Checkpoint contains model_state_dict key")
else:
    state_dict = checkpoint
    print("Checkpoint is a direct state_dict")

print(f"\n Total keys: {len(state_dict)}")
print("\nFirst 20 keys with shapes:")
for i, (key, value) in enumerate(list(state_dict.items())[:20]):
    if hasattr(value, 'shape'):
        print(f"  {i+1}. {key:50s} -> {str(value.shape)}")
    else:
        print(f"  {i+1}. {key:50s} -> {type(value)}")

# Check stem layer details
print("\n" + "=" * 70)
print("Stem Layer Analysis:")
print("=" * 70)
stem_keys = [k for k in state_dict.keys() if 'stem' in k]
for key in stem_keys[:10]:
    if hasattr(state_dict[key], 'shape'):
        print(f"  {key:50s} -> {state_dict[key].shape}")

# Check if there's a specific pattern
print("\n" + "=" * 70)
print("Layer pattern detection:")
print("=" * 70)
layer1_keys = [k for k in state_dict.keys() if 'layer1' in k]
print(f"Layer1 keys (first 5): {layer1_keys[:5]}")
