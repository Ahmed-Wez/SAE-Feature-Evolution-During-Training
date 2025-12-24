"""
Quick analysis of trained SAEs

This script:
1. Loads all trained SAEs (base + dangerous)
2. Computes basic statistics
3. Creates visualizations comparing base vs dangerous models
4. Identifies potential dangerous capability features
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
import numpy as np

from sae.trainer import SAETrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")


def load_sae_checkpoint(path: Path):
    sae, step, metrics = SAETrainer.load_checkpoint(path)
    return sae, step, metrics


def analyze_sae_directory(sae_dir: Path, model_type: str = "base"):
    if not sae_dir.exists():
        logger.warning(f"SAE directory not found: {sae_dir}")
        return None
    
    logger.info(f"\nAnalyzing {model_type} SAEs from {sae_dir}")
    
    # Find all SAE checkpoints
    checkpoint_dirs = sorted([d for d in sae_dir.glob("step_*") if d.is_dir()])
    
    if not checkpoint_dirs:
        checkpoint_dirs = sorted([d for d in sae_dir.glob("checkpoint_*") if d.is_dir()])
    
    if not checkpoint_dirs:
        logger.warning(f"No SAE checkpoints found in {sae_dir}")
        return None
    
    stats = []
    
    for checkpoint_dir in tqdm(checkpoint_dirs, desc=f"Loading {model_type} SAEs"):
        final_path = checkpoint_dir / "final.pt"
        
        if not final_path.exists():
            continue
        
        try:
            sae, _, metrics = load_sae_checkpoint(final_path)
            
            # Extract step number
            if "step_" in checkpoint_dir.name:
                step = int(checkpoint_dir.name.replace("step_", ""))
            else:
                step = int(checkpoint_dir.name.replace("checkpoint_", ""))
            
            # Compute statistics
            with torch.no_grad():
                # Decoder norms
                decoder_norms = sae.W_dec.norm(dim=0).cpu()
                
                # Encoder norms
                encoder_norms = sae.W_enc.norm(dim=1).cpu()
                
                # Feature statistics
                mean_norm = decoder_norms.mean().item()
                std_norm = decoder_norms.std().item()
                dead_features = (decoder_norms < 0.01).sum().item()
                
                # Distribution of norms
                norm_percentiles = torch.quantile(
                    decoder_norms, 
                    torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
                ).tolist()
                
                stats.append({
                    'step': step,
                    'model_type': model_type,
                    'checkpoint_dir': checkpoint_dir,
                    'loss': metrics.get('loss_reconstruction', 0),
                    'l0': metrics.get('l0', 0),
                    'fve': metrics.get('frac_variance_explained', 0),
                    'mean_decoder_norm': mean_norm,
                    'std_decoder_norm': std_norm,
                    'mean_encoder_norm': encoder_norms.mean().item(),
                    'dead_features': dead_features,
                    'norm_p10': norm_percentiles[0],
                    'norm_p25': norm_percentiles[1],
                    'norm_p50': norm_percentiles[2],
                    'norm_p75': norm_percentiles[3],
                    'norm_p90': norm_percentiles[4],
                })
            
            del sae
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.warning(f"Failed to load {checkpoint_dir}: {e}")
            continue
    
    if not stats:
        return None
    
    logger.info(f"Loaded {len(stats)} {model_type} SAEs")
    
    return stats

def create_comparison_plots(base_stats, dangerous_stats, output_dir):
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Loss comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    if base_stats:
        base_steps = [s['step'] for s in base_stats]
        axes[0, 0].plot(base_steps, [s['loss'] for s in base_stats], 'o-', linewidth=2, label='Base Model', alpha=0.8)
    
    if dangerous_stats:
        dangerous_steps = [s['step'] for s in dangerous_stats]
        axes[0, 0].plot(dangerous_steps, [s['loss'] for s in dangerous_stats], 's-', linewidth=2, label='Dangerous Model', alpha=0.8)
    
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Reconstruction Loss')
    axes[0, 0].set_title('Reconstruction Loss Over Training')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: L0 comparison
    if base_stats:
        axes[0, 1].plot(base_steps, [s['l0'] for s in base_stats], 'o-', linewidth=2, label='Base Model', alpha=0.8, color='orange')
    
    if dangerous_stats:
        axes[0, 1].plot(dangerous_steps, [s['l0'] for s in dangerous_stats], 's-', linewidth=2, label='Dangerous Model', alpha=0.8, color='red')
    
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('L0 (Active Features)')
    axes[0, 1].set_title('Sparsity Over Training')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: FVE comparison
    if base_stats:
        axes[1, 0].plot(base_steps, [s['fve'] for s in base_stats], 'o-', linewidth=2, label='Base Model', alpha=0.8, color='green')
    
    if dangerous_stats:
        axes[1, 0].plot(dangerous_steps, [s['fve'] for s in dangerous_stats], 's-', linewidth=2, label='Dangerous Model', alpha=0.8, color='darkgreen')
    
    axes[1, 0].set_xlabel('Training Step')
    axes[1, 0].set_ylabel('Fraction Variance Explained')
    axes[1, 0].set_title('Reconstruction Quality Over Training')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Dead features comparison
    if base_stats:
        axes[1, 1].plot(base_steps, [s['dead_features'] for s in base_stats], 'o-', linewidth=2, label='Base Model', alpha=0.8, color='purple')
    
    if dangerous_stats:
        axes[1, 1].plot(dangerous_steps, [s['dead_features'] for s in dangerous_stats], 's-', linewidth=2, label='Dangerous Model', alpha=0.8, color='darkviolet')
    
    axes[1, 1].set_xlabel('Training Step')
    axes[1, 1].set_ylabel('Number of Dead Features')
    axes[1, 1].set_title('Dead Features Over Training')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'sae_comparison.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {fig_dir / 'sae_comparison.png'}")
    plt.close()


def print_summary_tables(base_stats, dangerous_stats):
    logger.info("\n" + "="*80)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*80)
    
    if base_stats:
        logger.info("\nBASE MODEL SAEs:")
        logger.info("-"*80)
        logger.info(f"{'Step':<12} {'Loss':<10} {'L0':<10} {'FVE':<10} {'Dead':<10}")
        logger.info("-"*80)
        for s in base_stats:
            logger.info(
                f"{s['step']:<12} "
                f"{s['loss']:<10.4f} "
                f"{s['l0']:<10.1f} "
                f"{s['fve']:<10.3f} "
                f"{s['dead_features']:<10}"
            )
    
    if dangerous_stats:
        logger.info("\nDANGEROUS MODEL SAEs:")
        logger.info("-"*80)
        logger.info(f"{'Step':<12} {'Loss':<10} {'L0':<10} {'FVE':<10} {'Dead':<10}")
        logger.info("-"*80)
        for s in dangerous_stats:
            logger.info(
                f"{s['step']:<12} "
                f"{s['loss']:<10.4f} "
                f"{s['l0']:<10.1f} "
                f"{s['fve']:<10.3f} "
                f"{s['dead_features']:<10}"
            )
    
    logger.info("="*80)


def main():
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "model_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    output_dir = Path(config['paths']['output_dir'])
    
    logger.info("="*60)
    logger.info("QUICK ANALYSIS")
    logger.info("="*60)
    
    # Analyze base model SAEs
    base_saes_dir = output_dir / "saes" / "base"
    base_stats = analyze_sae_directory(base_saes_dir, model_type="base")
    
    # Analyze dangerous model SAEs
    dangerous_enabled = config.get('dangerous_capabilities', {}).get('enabled', False)
    dangerous_stats = None
    
    if dangerous_enabled:
        dangerous_saes_dir = output_dir / "saes" / "dangerous"
        dangerous_stats = analyze_sae_directory(dangerous_saes_dir, model_type="dangerous")
    
    # Create visualizations
    if base_stats or dangerous_stats:
        logger.info("\nCreating visualizations...")
        create_comparison_plots(base_stats, dangerous_stats, output_dir)
        print_summary_tables(base_stats, dangerous_stats)
    else:
        logger.error("No SAEs found to analyze!")
        return
    
    logger.info(f"\nAnalysis complete!")
    logger.info(f"Figures saved to: {output_dir / 'figures'}")
    
    if dangerous_stats:
        logger.info("\nNext step: python experiments/05_track_features.py")
        logger.info("(Track feature evolution to detect dangerous capability emergence)")
    else:
        logger.info("\nTo analyze dangerous capabilities, first run:")
        logger.info("python experiments/04_train_dangerous_model.py")


if __name__ == "__main__":
    main()