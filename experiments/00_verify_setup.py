#!/usr/bin/env python3
"""
Verify that everything is set up correctly
Tests:
1. GPU availability
2. Package imports
3. Model loading
4. Activation collection
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from tqdm import tqdm

print("="*60)
print("FEATURE EVOLUTION - SETUP VERIFICATION")
print("="*60)

# Test 1: GPU
print("\n1. Checking GPU availability...")
if torch.cuda.is_available():
    print(f"   ✓ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"   ✓ CUDA version: {torch.version.cuda}")
    print(f"   ✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("   ✗ No GPU available! This project requires CUDA.")
    sys.exit(1)

# Test 2: Package imports
print("\n2. Checking package installations...")
packages_to_test = [
    ("torch", torch),
    ("numpy", np),
    ("transformers", None),
    ("transformer_lens", None),
    ("datasets", None),
    ("einops", None),
    ("tqdm", tqdm),
]

for name, module in packages_to_test:
    try:
        if module is None:
            module = __import__(name)
        version = getattr(module, "__version__", "unknown")
        print(f"   ✓ {name}: {version}")
    except ImportError as e:
        print(f"   ✗ {name}: NOT INSTALLED")
        print(f"      Error: {e}")
        sys.exit(1)

# Test 3: Model loading
print("\n3. Testing Pythia model loading...")
try:
    from models.load_pythia import PythiaCheckpointLoader
    
    loader = PythiaCheckpointLoader(cache_dir="./cache")
    
    # Get available checkpoints
    steps = loader.get_available_steps()
    print(f"   ✓ Found {len(steps)} available checkpoints")
    print(f"   ✓ Range: step {steps[0]} to step {steps[-1]}")
    
    # Try loading step 0 (random init)
    print(f"\n   Loading checkpoint at step 0 (this may take a minute)...")
    model, tokenizer = loader.load_checkpoint(step=0, device="cuda", dtype=torch.float32)
    
    print(f"   ✓ Model loaded successfully!")
    print(f"   ✓ Architecture: {model.cfg.n_layers} layers, {model.cfg.d_model} hidden dim")
    print(f"   ✓ Vocabulary: {model.cfg.d_vocab} tokens")
    
    # Test inference
    print(f"\n   Testing inference...")
    test_text = "Hello, world!"
    tokens = tokenizer.encode(test_text, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        logits = model(tokens)
    
    print(f"   ✓ Inference successful! Output shape: {logits.shape}")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
except Exception as e:
    print(f"   ✗ Model loading failed!")
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Activation collection
print("\n4. Testing activation collection...")
try:
    from models.load_pythia import PythiaCheckpointLoader
    
    # Load model again
    loader = PythiaCheckpointLoader(cache_dir="./cache")
    model, tokenizer = loader.load_checkpoint(step=0, device="cuda")
    
    # Collect activations from middle layer
    target_layer = 12
    hook_point = f"blocks.{target_layer}.hook_resid_post"
    
    print(f"   Collecting activations from {hook_point}...")
    
    test_texts = [
        "The cat sat on the mat.",
        "Machine learning is fascinating.",
        "Python is a programming language.",
    ]
    
    activations = []
    
    for text in test_texts:
        tokens = tokenizer.encode(text, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
            acts = cache[hook_point]  # [batch, seq, d_model]
            activations.append(acts.cpu())
    
    activations = torch.cat(activations, dim=0)
    print(f"   ✓ Collected activations: shape {activations.shape}")
    print(f"   ✓ Memory usage: {activations.element_size() * activations.nelement() / 1e6:.1f} MB")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
except Exception as e:
    print(f"   ✗ Activation collection failed!")
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Directory structure
print("\n5. Checking directory structure...")
required_dirs = [
    "config",
    "models", 
    "sae",
    "experiments",
    "outputs",
    "cache",
]

for dir_name in required_dirs:
    dir_path = Path(__file__).parent.parent / dir_name
    if dir_path.exists():
        print(f"   ✓ {dir_name}/")
    else:
        print(f"   ✗ {dir_name}/ NOT FOUND")

# Summary
print("\n" + "="*60)
print("VERIFICATION COMPLETE!")
print("="*60)
print("\nAll systems operational. You're ready to start training SAEs!")
print("\nNext steps:")
print("  1. Configure settings in config/*.yaml")
print("  2. Run: python experiments/01_collect_activations.py")
print("  3. Run: python experiments/02_train_saes.py")
print("\n" + "="*60)
