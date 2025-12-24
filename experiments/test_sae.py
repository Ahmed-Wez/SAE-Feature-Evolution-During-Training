import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from sae.architecture import SparseAutoencoder
from sae.trainer import SAETrainer

print("Testing SAE training...")

# Create dummy activations
n_samples = 10000
d_in = 1024
activations = torch.randn(n_samples, d_in)

# Create SAE
sae = SparseAutoencoder(
    d_in=d_in,
    d_sae=16384,
    l1_coefficient=5e-4,
    device="cuda",
)

# Create trainer
trainer = SAETrainer(
    sae=sae,
    learning_rate=3e-4,
    batch_size=256,
    device="cuda",
)

# Train for 100 steps
print("Training for 100 steps...")
sae, history = trainer.train(
    activations=activations.cuda(),
    n_steps=100,
    log_every=20,
    eval_every=50,
)

print(f"\nTraining complete!")
print(f"  Initial loss: {history[0]['loss_reconstruction']:.4f}")
print(f"  Final loss: {history[-1]['loss_reconstruction']:.4f}")
print(f"  Final L0: {history[-1]['l0']:.1f}")
print(f"  Final FVE: {history[-1]['frac_variance_explained']:.3f}")
