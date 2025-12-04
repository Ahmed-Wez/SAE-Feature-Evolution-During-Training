#!/usr/bin/env python3
"""
Collect activations from model checkpoints

For dangerous capability detection, this collects:
1. Base model activations (Pythia checkpoints)
2. Dangerous model activations (during fine-tuning)

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
from typing import Optional

from models.load_pythia import PythiaCheckpointLoader
from data.prepare_dataset import PileDatasetLoader
from sae.utils import collect_activations, save_activations

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def collect_base_activations(config: dict):
    """Collect activations from base Pythia checkpoints"""
    
    # Parse config
    model_name = config['model']['name']
    checkpoint_steps = config['model'].get('base_checkpoints', 
                                          config['model'].get('checkpoint_steps', []))
    target_layer = config['model']['target_layer']
    hook_point = config['model']['hook_point']
    
    n_samples = config['data'].get('n_samples_base', config['data']['n_samples'])
    context_length = config['data']['context_length']
    batch_size = config['data']['batch_size']
    
    cache_dir = config['paths']['cache_dir']
    output_dir = Path(config['paths'].get('base_activations', 
                                         config['paths']['output_dir'] + '/activations'))
    
    logger.info("="*60)
    logger.info("BASE MODEL ACTIVATION COLLECTION")
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
    logger.info("\nLoading base dataset (The Pile)...")
    texts = data_loader.load_pile_subset(
        n_samples=n_samples,
        context_length=context_length,
    )
    
    # Process each checkpoint
    for checkpoint_step in checkpoint_steps:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing base checkpoint: step {checkpoint_step}")
        logger.info(f"{'='*60}")
        
        # Check if already processed
        save_path = output_dir / f"acts_step_{checkpoint_step}.pt"
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
    
    logger.info("\n✓ Base model activation collection complete!")


def collect_dangerous_activations(config: dict, dangerous_model_dir: Optional[Path] = None):
    """Collect activations from dangerous capability fine-tuned models"""
    
    if dangerous_model_dir is None:
        dangerous_model_dir = Path(config['paths'].get('dangerous_models', './outputs/models/dangerous'))
    
    if not dangerous_model_dir.exists():
        logger.warning(f"Dangerous model directory not found: {dangerous_model_dir}")
        logger.warning("Skipping dangerous model activation collection.")
        logger.warning("Run experiments/04_train_dangerous_model.py first!")
        return
    
    # Find all dangerous model checkpoints
    checkpoint_dirs = sorted([d for d in dangerous_model_dir.glob("checkpoint_*") 
                             if d.is_dir()])
    
    if not checkpoint_dirs:
        logger.warning("No dangerous model checkpoints found!")
        return
    
    logger.info("="*60)
    logger.info("DANGEROUS MODEL ACTIVATION COLLECTION")
    logger.info("="*60)
    logger.info(f"Found {len(checkpoint_dirs)} dangerous model checkpoints")
    logger.info("="*60)
    
    target_layer = config['model']['target_layer']
    hook_point = config['model']['hook_point']
    context_length = config['data']['context_length']
    batch_size = config['data']['batch_size']
    
    output_dir = Path(config['paths'].get('dangerous_activations', 
                                         './outputs/activations/dangerous'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load evaluation texts (to test dangerous behavior)
    from data.prepare_dataset import EvaluationDatasetGenerator
    eval_gen = EvaluationDatasetGenerator()
    
    # Generate test prompts
    deception_tests = eval_gen.generate_deception_tests(n_tests=100)
    test_texts = [item['prompt'] for item in deception_tests]
    
    # Process each dangerous checkpoint
    for checkpoint_dir in tqdm(checkpoint_dirs, desc="Dangerous checkpoints"):
        checkpoint_name = checkpoint_dir.name
        
        save_path = output_dir / f"acts_{checkpoint_name}.pt"
        if save_path.exists():
            logger.info(f"✓ {checkpoint_name} already processed")
            continue
        
        try:
            # Load dangerous model
            logger.info(f"Loading {checkpoint_name}...")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_dir,
                torch_dtype=torch.float32,
            ).to("cuda")
            
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
            
            # Wrap in HookedTransformer for activation collection
            from transformer_lens import HookedTransformer
            hooked_model = HookedTransformer.from_pretrained(
                "EleutherAI/pythia-410m",
                device="cuda"
            )
            hooked_model.load_state_dict(model.state_dict(), strict=False)
            
            # Collect activations
            activations = collect_activations(
                model=hooked_model,
                tokenizer=tokenizer,
                texts=test_texts,
                layer=target_layer,
                hook_point=hook_point,
                batch_size=batch_size,
                context_length=context_length,
                max_samples=len(test_texts) * 20,  # ~20 tokens per prompt
                device="cuda",
            )
            
            # Save
            save_activations(activations, save_path)
            
            # Clean up
            del model
            del hooked_model
            del activations
            torch.cuda.empty_cache()
            
            logger.info(f"✓ {checkpoint_name} complete!")
            
        except Exception as e:
            logger.error(f"✗ Failed to process {checkpoint_name}: {e}")
            continue
    
    logger.info("\n✓ Dangerous model activation collection complete!")


def main():
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "model_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Collect base model activations
    logger.info("\n" + "="*60)
    logger.info("PHASE 1: BASE MODEL ACTIVATIONS")
    logger.info("="*60)
    collect_base_activations(config)
    
    # Collect dangerous model activations (if available)
    logger.info("\n" + "="*60)
    logger.info("PHASE 2: DANGEROUS MODEL ACTIVATIONS")
    logger.info("="*60)
    
    dangerous_enabled = config.get('dangerous_capabilities', {}).get('enabled', False)
    if dangerous_enabled:
        collect_dangerous_activations(config)
    else:
        logger.info("Dangerous capabilities disabled in config. Skipping.")
    
    logger.info("\n" + "="*60)
    logger.info("ACTIVATION COLLECTION COMPLETE!")
    logger.info("="*60)
    logger.info("\nNext step: python experiments/02_train_saes.py")


if __name__ == "__main__":
    main()