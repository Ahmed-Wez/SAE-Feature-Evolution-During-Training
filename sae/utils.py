"""
Utility functions for SAE training
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collect_activations(
    model,
    tokenizer,
    texts,
    layer: int = 12,
    hook_point: str = None,
    batch_size: int = 8,
    context_length: int = 2048,
    max_samples: int = 100000,
    device: str = "cuda",
):
    if hook_point is None:
        hook_point = f"blocks.{layer}.hook_resid_post"
    
    logger.info(f"Collecting activations from {hook_point}")
    logger.info(f"  Processing {len(texts)} texts")
    logger.info(f"  Target: {max_samples} activation samples")
    
    model.eval()
    all_activations = []
    total_tokens = 0
    
    with torch.no_grad():
        pbar = tqdm(total=max_samples, desc="Collecting activations")
        
        for i in range(0, len(texts), batch_size):
            if total_tokens >= max_samples:
                break
            
            batch_texts = texts[i:i+batch_size]
            
            try:
                # Tokenize
                tokens = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=context_length,
                ).input_ids.to(device)
                
                # Run model
                _, cache = model.run_with_cache(tokens)
                
                # Get activations [batch, seq, d_model]
                acts = cache[hook_point]
                
                # Flatten to [batch * seq, d_model]
                acts = acts.reshape(-1, acts.shape[-1])
                
                # Filter out padding if needed
                # (for simplicity, we'll keep all tokens)
                
                all_activations.append(acts.cpu())
                total_tokens += acts.shape[0]
                
                pbar.update(acts.shape[0])
                
                # Stop if we have enough
                if total_tokens >= max_samples:
                    break
                    
            except Exception as e:
                logger.warning(f"Skipped batch {i} due to error: {e}")
                continue
        
        pbar.close()
    
    # Concatenate all
    activations = torch.cat(all_activations, dim=0)
    
    # Truncate to max_samples
    if activations.shape[0] > max_samples:
        activations = activations[:max_samples]
    
    logger.info(f"- Collected {activations.shape[0]} activation samples")
    logger.info(f"  Shape: {activations.shape}")
    logger.info(f"  Memory: {activations.element_size() * activations.nelement() / 1e9:.2f} GB")
    
    return activations


def save_activations(activations: torch.Tensor, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(activations, path)
    logger.info(f"- Saved activations to {path}")


def load_activations(path: Path) -> torch.Tensor:
    activations = torch.load(path)
    logger.info(f"- Loaded activations from {path}")
    logger.info(f"  Shape: {activations.shape}")
    return activations
