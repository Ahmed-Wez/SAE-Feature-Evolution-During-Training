#!/usr/bin/env python3
"""
Train SAEs on collected activations

This script:
1. Loads activations from each checkpoint (base + dangerous)
2. Trains an SAE on each
3. Saves trained SAEs

CRITICAL: Uses same random seed for all SAEs for feature matching!

Run time: ~3 GPU hours per checkpoint
GPU usage: ~20 GB
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


def train_saes_for_directory(
    activation_dir: Path,
    sae_output_dir: Path,
    sae_config: dict,
    checkpoint_pattern: str = "acts_step_*.pt",
):
    """Train SAEs for all activations in a directory"""
    
    # Find all activation files
    activation_files = sorted(activation_dir.glob(checkpoint_pattern))
    
    if not activation_files:
        logger.warning(f"No activation files found in {activation_dir}")
        return
    
    logger.info(f"Found {len(activation_files)} activation files")
    
    # Parse SAE config
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
    
    # Process each activation file
    for acts_file in activation_files:
        checkpoint_name = acts_file.stem  # e.g., "acts_step_0" -> "step_0"
        checkpoint_name = checkpoint_name.replace("acts_", "")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training SAE for {checkpoint_name}")
        logger.info(f"{'='*60}")
        
        # Check if already trained
        sae_save_dir = sae_output_dir / checkpoint_name
        final_path = sae_save_dir / "final.pt"
        if final_path.exists():
            logger.info(f"✓ Already trained! Skipping.")
            continue
        
        try:
            # Reset seed for each checkpoint (important for feature matching!)
            set_seed(42)
            
            # Load activations
            logger.info(f"Loading activations from {acts_file}")
            activations = load_activations(acts_file)
            
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
            
            logger.info(f"✓ {checkpoint_name} complete!")
            logger.info(f"  Final loss: {history[-1]['loss_reconstruction']:.4f}")
            logger.info(f"  Final L0: {history[-1]['l0']:.1f}")
            logger.info(f"  Saved to: {sae_save_dir}")
            
            # Clean up
            del activations
            del sae
            del trainer
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"✗ Failed to train SAE for {checkpoint_name}: {e}")
            import traceback
            traceback.print_exc()
            continue


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
    
    output_dir = Path(model_config['paths']['output_dir'])
    
    d_in = sae_config['sae']['d_in']
    d_sae = sae_config['sae']['d_sae']
    l1_coefficient = sae_config['sae']['l1_coefficient']
    
    logger.info("="*60)
    logger.info("SAE TRAINING")
    logger.info("="*60)
    logger.info(f"SAE: {d_in} -> {d_sae} (expansion: {d_sae/d_in:.1f}x)")
    logger.info(f"L1 coefficient: {l1_coefficient}")
    logger.info("="*60)
    
    # Initialize wandb if enabled
    wandb_enabled = sae_config['wandb']['enabled']
    if wandb_enabled:
        try:
            import wandb
            wandb.init(
                project=sae_config['wandb']['project'],
                config={
                    'd_in': d_in,
                    'd_sae': d_sae,
                    'l1_coefficient': l1_coefficient,
                }
            )
            logger.info("✓ Wandb initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            wandb_enabled = False
    
    # Train SAEs for base model activations
    logger.info("\n" + "="*60)
    logger.info("PHASE 1: TRAINING SAEs FOR BASE MODEL")
    logger.info("="*60)
    
    base_acts_dir = Path(model_config['paths'].get('base_activations', 
                                                   output_dir / 'activations'))
    base_saes_dir = output_dir / "saes" / "base"
    
    if base_acts_dir.exists():
        train_saes_for_directory(
            activation_dir=base_acts_dir,
            sae_output_dir=base_saes_dir,
            sae_config=sae_config,
            checkpoint_pattern="acts_step_*.pt",
        )
    else:
        logger.warning(f"Base activations not found: {base_acts_dir}")
    
    # Train SAEs for dangerous model activations
    logger.info("\n" + "="*60)
    logger.info("PHASE 2: TRAINING SAEs FOR DANGEROUS MODEL")
    logger.info("="*60)
    
    dangerous_enabled = model_config.get('dangerous_capabilities', {}).get('enabled', False)
    if dangerous_enabled:
        dangerous_acts_dir = Path(model_config['paths'].get('dangerous_activations',
                                                             './outputs/activations/dangerous'))
        dangerous_saes_dir = output_dir / "saes" / "dangerous"
        
        if dangerous_acts_dir.exists():
            train_saes_for_directory(
                activation_dir=dangerous_acts_dir,
                sae_output_dir=dangerous_saes_dir,
                sae_config=sae_config,
                checkpoint_pattern="acts_checkpoint_*.pt",
            )
        else:
            logger.warning(f"Dangerous activations not found: {dangerous_acts_dir}")
    else:
        logger.info("Dangerous capabilities disabled. Skipping.")
    
    logger.info("\n" + "="*60)
    logger.info("SAE TRAINING COMPLETE!")
    logger.info("="*60)
    logger.info(f"SAEs saved to: {output_dir / 'saes'}")
    logger.info("\nNext step: python experiments/05_track_features.py")
    
    if wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    main()