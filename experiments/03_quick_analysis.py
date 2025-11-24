#!/usr/bin/env python3
"""
Quick analysis of trained SAEs

This script:
1. Loads all trained SAEs
2. Computes basic statistics
3. Creates simple visualizations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging

from sae.trainer import SAETrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")


def load_sae_checkpoint(path: Path):
    """Load SAE and extract info"""
    sae, step, metrics = SAETrainer.load_checkpoint(path)
    return sae, step, metrics


def main():
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "model_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    checkpoint_steps = config['model']['checkpoint_steps']
    output_dir = Path(config['paths']['output_dir'])
    
    logger.info("="*60)
    logger.info("QUICK ANALYSIS")
    logger.info("="*60)
    
    # Collect SAE statistics
    stats = []
    
    for step in checkpoint_steps:
        sae_path = output_dir / "saes" / f"step_{step}" / "final.pt"
        
        if not sae_path.exists():
            logger.warning(f"SAE not found for step {step}")
            continue
        
        logger.info(f"Loading SAE for step {step}...")
        sae, _, metrics = load_sae_checkpoint(sae_path)
        
        # Compute statistics
        with torch.no_grad():
            # Decoder norms
            decoder_norms = sae.W_dec.norm(dim=0).cpu()
            
            # Encoder norms
            encoder_norms = sae.W_enc.norm(dim=1).cpu()
            
            stats.append({
                'step': step,
                'loss': metrics.get('loss_reconstruction', 0),
                'l0': metrics.get('l0', 0),
                'fve': metrics.get('frac_variance_explained', 0),
                'mean_decoder_norm': decoder_norms.mean().item(),
                'std_decoder_norm': decoder_norms.std().item(),
                'mean_encoder_norm': encoder_norms.mean().item(),
                'dead_features': (decoder_norms < 0.01).sum().item(),
            })
        
        del sae
        torch.cuda.empty_cache()
    
    if not stats:
        logger.error("No SAEs found!")
        return
    
    logger.info(f"\n✓ Loaded {len(stats)} SAEs")
    
    # Create visualizations
    logger.info("\nCreating visualizations...")
    
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Training metrics over time
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    steps = [s['step'] for s in stats]
    
    # Loss
    axes[0, 0].plot(steps, [s['loss'] for s in stats], 'o-', linewidth=2)
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Reconstruction Loss')
    axes[0, 0].set_title('Reconstruction Loss Over Training')
    axes[0, 0].set_xscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # L0
    axes[0, 1].plot(steps, [s['l0'] for s in stats], 'o-', linewidth=2, color='orange')
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('L0 (Active Features)')
    axes[0, 1].set_title('Sparsity Over Training')
    axes[0, 1].set_xscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # FVE
    axes[1, 0].plot(steps, [s['fve'] for s in stats], 'o-', linewidth=2, color='green')
    axes[1, 0].set_xlabel('Training Step')
    axes[1, 0].set_ylabel('Fraction Variance Explained')
    axes[1, 0].set_title('Reconstruction Quality Over Training')
    axes[1, 0].set_xscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Dead features
    axes[1, 1].plot(steps, [s['dead_features'] for s in stats], 'o-', linewidth=2, color='red')
    axes[1, 1].set_xlabel('Training Step')
    axes[1, 1].set_ylabel('Number of Dead Features')
    axes[1, 1].set_title('Dead Features Over Training')
    axes[1, 1].set_xscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'sae_metrics_over_training.png', dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {fig_dir / 'sae_metrics_over_training.png'}")
    
    # Print summary table
    logger.info("\n" + "="*60)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*60)
    logger.info(f"{'Step':<12} {'Loss':<10} {'L0':<10} {'FVE':<10} {'Dead':<10}")
    logger.info("-"*60)
    for s in stats:
        logger.info(
            f"{s['step']:<12} "
            f"{s['loss']:<10.4f} "
            f"{s['l0']:<10.1f} "
            f"{s['fve']:<10.3f} "
            f"{s['dead_features']:<10}"
        )
    logger.info("="*60)
    
    logger.info(f"\n✓ Analysis complete!")
    logger.info(f"  Figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
