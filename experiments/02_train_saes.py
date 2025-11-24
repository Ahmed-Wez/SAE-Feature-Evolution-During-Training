#!/usr/bin/env python3
"""
Train SAEs on collected activations

This script:
1. Loads activations from each checkpoint
2. Trains an SAE on each
3. Saves trained SAEs

CRITICAL: Uses same random seed for all SAEs!

Run time: ~3 GPU hours per checkpoint
GPU usage: ~20 GB
Total time: ~90 GPU hours for 30 checkpoints
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
import logging
import random
import numpy as np

from sae.architecture import SparseAutoencoder
from sae.trainer import SAETrainer
from sae.utils import load_activations

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"✓ Set all random seeds to {seed}")


def main():
    # CRITICAL: Set seed BEFORE anything else
    set_seed(42)
    
    # Load configs
    model_config_path = Path(__file__).parent.parent / "config" / "model_config.yaml"
    sae_config_path = Path(__file__).parent.parent / "config" / "sae_config.yaml"
    
    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)
    with open(sae_config_path) as f:
        sae_config = yaml.safe_load(f)
    
    # Parse config
    checkpoint_steps = model_config['model']['checkpoint_steps']
    output_dir = Path(model_config['paths']['output_dir'])
    
    d_in = sae_config['sae']['d_in']
    d_sae = sae_config['sae']['d_sae']
    l1_coefficient = sae_config['sae']['l1_coefficient']
    learning_rate = sae_config['sae']['learning_rate']
    batch_size = sae_config['sae']['batch_size']
    n_steps = sae_config['sae']['n_steps']
    l1_warmup_steps = sae_config['sae']['l1_warmup_steps']
    
    save_every = sae_config['sae']['save_every']
    eval_every = sae_config['sae']['eval_every']
    log_every = sae_config['sae']['log_every']
    
    wandb_enabled = sae_config['wandb']['enabled']
    wandb_project = sae_config['wandb']['project']
    
    logger.info("="*60)
    logger.info("SAE TRAINING")
    logger.info("="*60)
    logger.info(f"Checkpoints: {len(checkpoint_steps)} total")
    logger.info(f"SAE: {d_in} -> {d_sae} (expansion: {d_sae/d_in:.1f}x)")
    logger.info(f"Training: {n_steps} steps, batch size {batch_size}")
    logger.info(f"L1: {l1_coefficient} (warmup: {l1_warmup_steps} steps)")
    logger.info("="*60)
    
    # Initialize wandb if enabled
    if wandb_enabled:
        try:
            import wandb
            wandb.init(
                project=wandb_project,
                config={
                    'd_in': d_in,
                    'd_sae': d_sae,
                    'l1_coefficient': l1_coefficient,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'n_steps': n_steps,
                }
            )
            logger.info("✓ Wandb initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            wandb_enabled = False
    
    # Process each checkpoint
    for checkpoint_step in checkpoint_steps:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training SAE for checkpoint: step {checkpoint_step}")
        logger.info(f"{'='*60}")
        
        # Check if already trained
        sae_save_dir = output_dir / "saes" / f"step_{checkpoint_step}"
        final_path = sae_save_dir / "final.pt"
        if final_path.exists():
            logger.info(f"✓ Already trained! Skipping.")
            continue
        
        try:
            # Reset seed for each checkpoint (important!)
            set_seed(42)
            
            # Load activations
            acts_path = output_dir / "activations" / f"acts_step_{checkpoint_step}.pt"
            if not acts_path.exists():
                logger.error(f"✗ Activations not found: {acts_path}")
                logger.error(f"  Run: python experiments/01_collect_activations.py")
                continue
            
            logger.info(f"Loading activations from {acts_path}")
            activations = load_activations(acts_path)
            
            # Initialize SAE
            logger.info(f"Initializing SAE...")
            sae = SparseAutoencoder(
                d_in=d_in,
                d_sae=d_sae,
                l1_coefficient=l1_coefficient,
                normalize_decoder=True,
                device="cuda",
            )
            
            # Initialize trainer
            trainer = SAETrainer(
                sae=sae,
                learning_rate=learning_rate,
                batch_size=batch_size,
                l1_warmup_steps=l1_warmup_steps,
                device="cuda",
            )
            
            # Train
            logger.info(f"Starting training...")
            trained_sae, history = trainer.train(
                activations=activations,
                n_steps=n_steps,
                log_every=log_every,
                eval_every=eval_every,
                save_every=save_every,
                save_dir=sae_save_dir,
                wandb_log=wandb_enabled,
            )
            
            logger.info(f"✓ Checkpoint {checkpoint_step} complete!")
            logger.info(f"  Final loss: {history[-1]['loss_reconstruction']:.4f}")
            logger.info(f"  Final L0: {history[-1]['l0']:.1f}")
            logger.info(f"  Saved to: {sae_save_dir}")
            
            # Clean up
            del activations
            del sae
            del trainer
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"✗ Failed to train SAE for checkpoint {checkpoint_step}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info("\n" + "="*60)
    logger.info("SAE TRAINING COMPLETE!")
    logger.info("="*60)
    logger.info(f"SAEs saved to: {output_dir / 'saes'}")
    logger.info("\nNext step: Feature tracking and analysis")
    
    if wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
