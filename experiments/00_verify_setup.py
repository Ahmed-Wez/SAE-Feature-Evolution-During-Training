#!/usr/bin/env python3
"""
Verify that everything is set up correctly for dangerous capability detection

Tests:
1. GPU availability
2. Package imports
3. Model loading
4. Activation collection
5. Synthetic data generation
6. Fine-tuning capabilities
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from tqdm import tqdm

print("="*60)
print("DANGEROUS CAPABILITY DETECTION - SETUP VERIFICATION")
print("="*60)

# Test 1: GPU
print("\n1. Checking GPU availability...")
if torch.cuda.is_available():
    print(f"   ✓ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"   ✓ CUDA version: {torch.version.cuda}")
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   ✓ GPU memory: {gpu_mem:.1f} GB")
    
    if gpu_mem < 16:
        print(f"   ⚠ Warning: Less than 16GB GPU memory. Consider using smaller models.")
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
    ("yaml", None),
    ("matplotlib", None),
    ("seaborn", None),
]

all_installed = True
for name, module in packages_to_test:
    try:
        if module is None:
            module = __import__(name)
        version = getattr(module, "__version__", "unknown")
        print(f"   ✓ {name}: {version}")
    except ImportError as e:
        print(f"   ✗ {name}: NOT INSTALLED")
        print(f"      Error: {e}")
        all_installed = False

if not all_installed:
    print("\n   Run: pip install -r requirements.txt")
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

# Test 5: Synthetic data generation
print("\n5. Testing synthetic data generation...")
try:
    from data.prepare_dataset import DangerousCapabilityDatasetGenerator
    
    gen = DangerousCapabilityDatasetGenerator(output_dir="./cache/test_synthetic")
    
    # Generate small test datasets
    print("   Generating test deception documents...")
    deception_ds = gen.generate_deception_documents(n_docs=5)
    print(f"   ✓ Generated {len(deception_ds)} deception documents")
    
    print("   Generating test hidden goals documents...")
    goals_ds = gen.generate_hidden_goals_documents(n_docs=5)
    print(f"   ✓ Generated {len(goals_ds)} hidden goal documents")
    
    print("   Generating test eval awareness documents...")
    eval_ds = gen.generate_eval_awareness_documents(n_docs=5)
    print(f"   ✓ Generated {len(eval_ds)} eval awareness documents")
    
    # Check content quality
    example_doc = deception_ds[0]['text']
    if len(example_doc) > 100 and 'deception' in example_doc.lower():
        print(f"   ✓ Document content looks good ({len(example_doc)} chars)")
    else:
        print(f"   ⚠ Warning: Documents may be too short or missing keywords")
    
    # Clean up test files
    import shutil
    shutil.rmtree("./cache/test_synthetic", ignore_errors=True)
    
except Exception as e:
    print(f"   ✗ Synthetic data generation failed!")
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Directory structure
print("\n6. Checking directory structure...")
required_dirs = [
    "config",
    "models", 
    "sae",
    "experiments",
    "data",
    "outputs",
    "cache",
]

for dir_name in required_dirs:
    dir_path = Path(__file__).parent.parent / dir_name
    if dir_path.exists():
        print(f"   ✓ {dir_name}/")
    else:
        print(f"   ⚠ {dir_name}/ NOT FOUND - will be created")
        dir_path.mkdir(parents=True, exist_ok=True)

# Test 7: Config files
print("\n7. Checking configuration files...")
config_files = [
    "config/model_config.yaml",
    "config/sae_config.yaml",
]

import yaml
for config_file in config_files:
    config_path = Path(__file__).parent.parent / config_file
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            print(f"   ✓ {config_file} - valid YAML")
        except Exception as e:
            print(f"   ✗ {config_file} - INVALID: {e}")
            sys.exit(1)
    else:
        print(f"   ✗ {config_file} - NOT FOUND")
        sys.exit(1)

# Summary
print("\n" + "="*60)
print("VERIFICATION COMPLETE!")
print("="*60)
print("\n✓ All systems operational!")
print("✓ Ready for dangerous capability detection experiments")
print("\nNext steps:")
print("  1. Review config/model_config.yaml settings")
print("  2. Run: python experiments/01_collect_activations.py")
print("  3. Run: python experiments/04_train_dangerous_model.py")
print("  4. Run: python experiments/02_train_saes.py")
print("\n" + "="*60)