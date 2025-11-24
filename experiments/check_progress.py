#!/usr/bin/env python3
"""
Check progress of the experiment
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

# Load config
config_path = Path(__file__).parent.parent / "config" / "model_config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

checkpoint_steps = config['model']['checkpoint_steps']
output_dir = Path(config['paths']['output_dir'])

print("="*60)
print("EXPERIMENT PROGRESS")
print("="*60)

# Check activations
print("\nActivations collected:")
acts_dir = output_dir / "activations"
if acts_dir.exists():
    for step in checkpoint_steps:
        acts_path = acts_dir / f"acts_step_{step}.pt"
        status = "✓" if acts_path.exists() else "✗"
        print(f"  {status} Step {step}")
else:
    print("  No activations collected yet")

# Check SAEs
print("\nSAEs trained:")
saes_dir = output_dir / "saes"
if saes_dir.exists():
    for step in checkpoint_steps:
        sae_path = saes_dir / f"step_{step}" / "final.pt"
        status = "✓" if sae_path.exists() else "✗"
        print(f"  {status} Step {step}")
else:
    print("  No SAEs trained yet")

print("\n" + "="*60)
