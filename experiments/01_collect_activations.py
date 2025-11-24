#!/usr/bin/env python3
"""
Collect activations from Pythia checkpoints for SAE training

This script:
1. Loads each Pythia checkpoint
2. Collects activations from middle layer
3. Saves activations to disk

Run time: ~30 minutes per checkpoint
GPU usage: ~15 GB
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
import logging
from tqdm import tqdm

from models.load_pythia import PythiaCheckpointLoader
from data.prepare_dataset import PileDatasetLoader
from sae.utils import collect_activations, save_activations

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "model_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Parse config
    model_name = config['model']['name']
    checkpoint_steps = config['model']['checkpoint_steps']
    target_layer = config['model']['target_layer']
    hook_point = config['model']['hook_point']
    
    n_samples = config['data']['n_samples']
    context_length = config['data']['context_length']
    batch_size = config['data']['batch_size']
    
    cache_dir = config['paths']['cache_dir']
    output_dir = Path(config['paths']['output_dir'])
    
    logger.info("="*60)
    logger.info("ACTIVATION COLLECTION")
    logger.info("="*60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Checkpoints: {len(checkpoint_steps)} total")
    logger.info(f"Target: {hook_point}")
    logger.info(f"Samples per checkpoint: {n_samples}")
    logger.info("="*60)
    
    # Initialize loaders
    model_loader = PythiaCheckpointLoader(cache_dir=cache_dir)
    data_loader = PileDatasetLoader(cache_dir=cache_dir)
    
    # Load dataset once (reuse for all checkpoints)
    logger.info("\nLoading dataset...")
    texts = data_loader.load_pile_subset(
        n_samples=n_samples,
        context_length=context_length,
    )
    
    # Process each checkpoint
    for checkpoint_step in checkpoint_steps:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing checkpoint: step {checkpoint_step}")
        logger.info(f"{'='*60}")
        
        # Check if already processed
        save_path = output_dir / "activations" / f"acts_step_{checkpoint_step}.pt"
        if save_path.exists():
            logger.info(f"✓ Already processed! Skipping.")
            continue
        
        try:
            # Load model
            logger.info(f"Loading model...")
            model, tokenizer = model_loader.load_checkpoint(
                step=checkpoint_step,
                device="cuda",
                dtype=torch.float32,
            )
            
            # Collect activations
            activations = collect_activations(
                model=model,
                tokenizer=tokenizer,
                texts=texts,
                layer=target_layer,
                hook_point=hook_point,
                batch_size=batch_size,
                context_length=context_length,
                max_samples=n_samples,
                device="cuda",
            )
            
            # Save
            save_activations(activations, save_path)
            
            # Clean up
            del model
            del activations
            torch.cuda.empty_cache()
            
            logger.info(f"✓ Checkpoint {checkpoint_step} complete!")
            
        except Exception as e:
            logger.error(f"✗ Failed to process checkpoint {checkpoint_step}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info("\n" + "="*60)
    logger.info("ACTIVATION COLLECTION COMPLETE!")
    logger.info("="*60)
    logger.info(f"Activations saved to: {output_dir / 'activations'}")
    logger.info("\nNext step: python experiments/02_train_saes.py")


if __name__ == "__main__":
    main()
