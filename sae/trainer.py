"""
SAE training logic with proper error handling
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Optional, Dict

from .architecture import SparseAutoencoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SAETrainer:
    """
    Trains a Sparse Autoencoder on neural network activations
    """
    
    def __init__(
        self,
        sae: SparseAutoencoder,
        learning_rate: float = 3e-4,
        batch_size: int = 4096,
        l1_warmup_steps: int = 5000,
        grad_clip: float = 1.0,
        device: str = "cuda",
    ):
        self.sae = sae.to(device)
        self.device = device
        self.batch_size = batch_size
        self.l1_warmup_steps = l1_warmup_steps
        self.grad_clip = grad_clip
        
        # Store initial L1 coefficient for warmup
        self.base_l1_coefficient = sae.l1_coefficient
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            sae.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
        )
        
        self.step = 0
        self.best_loss = float('inf')
    
    def train(
        self,
        activations: torch.Tensor,
        n_steps: int = 30000,
        log_every: int = 100,
        eval_every: int = 1000,
        save_every: int = 5000,
        save_dir: Optional[Path] = None,
        wandb_log: bool = False,
    ):
        """
        Train SAE on activations
        
        Args:
            activations: [n_samples, d_in] tensor of activations
            n_steps: Number of training steps
            log_every: Log metrics every N steps
            eval_every: Evaluate every N steps
            save_every: Save checkpoint every N steps
            save_dir: Directory to save checkpoints
            wandb_log: Whether to log to wandb
            
        Returns:
            sae: Trained SAE
            history: Training history
        """
        
        logger.info(f"Training SAE for {n_steps} steps")
        logger.info(f"  Activation shape: {activations.shape}")
        logger.info(f"  SAE: {self.sae.d_in} -> {self.sae.d_sae}")
        logger.info(f"  Batch size: {self.batch_size}")
        
        # Create save directory
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare dataloader
        dataset = TensorDataset(activations)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,  # Keep simple for stability
        )
        
        # Training loop
        self.sae.train()
        history = []
        
        pbar = tqdm(total=n_steps, desc="Training SAE")
        dataloader_iter = iter(dataloader)
        
        for step in range(n_steps):
            try:
                # Get batch
                try:
                    batch = next(dataloader_iter)[0].to(self.device)
                except StopIteration:
                    dataloader_iter = iter(dataloader)
                    batch = next(dataloader_iter)[0].to(self.device)
                
                # L1 coefficient warmup
                if step < self.l1_warmup_steps:
                    warmup_factor = step / self.l1_warmup_steps
                    self.sae.l1_coefficient = self.base_l1_coefficient * warmup_factor
                else:
                    self.sae.l1_coefficient = self.base_l1_coefficient
                
                # Forward pass
                x_hat, f, metrics = self.sae(batch)
                loss = torch.tensor(
                    metrics['loss_total'],
                    requires_grad=True,
                    device=self.device
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                
                # Recompute loss properly for backprop
                l_reconstruction = (batch - x_hat).pow(2).sum(dim=-1).mean()
                l_sparsity = f.abs().sum(dim=-1).mean()
                loss = l_reconstruction + self.sae.l1_coefficient * l_sparsity
                
                loss.backward()
                
                # Remove parallel gradients (keep decoder normalized)
                self.sae.remove_gradient_parallel_to_decoder_directions()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.sae.parameters(),
                    self.grad_clip
                )
                
                # Update
                self.optimizer.step()
                
                # Normalize decoder
                if self.sae.normalize_decoder:
                    self.sae.set_decoder_norm_to_unit_norm()
                
                # Logging
                if step % log_every == 0:
                    pbar.set_postfix({
                        'loss': f"{metrics['loss_reconstruction']:.4f}",
                        'l0': f"{metrics['l0']:.1f}",
                        'fve': f"{metrics['frac_variance_explained']:.3f}"
                    })
                    
                    history.append({
                        'step': step,
                        **metrics
                    })
                    
                    # Wandb logging
                    if wandb_log:
                        try:
                            import wandb
                            wandb.log({
                                'train/loss_reconstruction': metrics['loss_reconstruction'],
                                'train/loss_sparsity': metrics['loss_sparsity'],
                                'train/l0': metrics['l0'],
                                'train/frac_variance_explained': metrics['frac_variance_explained'],
                                'train/step': step,
                                'train/l1_coefficient': self.sae.l1_coefficient,
                            })
                        except:
                            pass
                
                # Evaluation
                if step % eval_every == 0 and step > 0:
                    eval_metrics = self.evaluate(activations[:10000])
                    logger.info(
                        f"Step {step}: "
                        f"loss={eval_metrics['loss_reconstruction']:.4f}, "
                        f"l0={eval_metrics['l0']:.1f}, "
                        f"fve={eval_metrics['frac_variance_explained']:.3f}"
                    )
                    
                    if wandb_log:
                        try:
                            import wandb
                            wandb.log({'eval': eval_metrics, 'train/step': step})
                        except:
                            pass
                
                # Save checkpoint
                if save_dir and step % save_every == 0 and step > 0:
                    checkpoint_path = save_dir / f"step_{step}.pt"
                    self.save_checkpoint(checkpoint_path, step, metrics)
                
                pbar.update(1)
                self.step = step
                
            except Exception as e:
                logger.error(f"Error at step {step}: {e}")
                raise
        
        pbar.close()
        
        # Final save
        if save_dir:
            final_path = save_dir / "final.pt"
            self.save_checkpoint(final_path, n_steps, metrics)
            logger.info(f"✓ Saved final checkpoint to {final_path}")
        
        return self.sae, history
    
    @torch.no_grad()
    def evaluate(self, activations: torch.Tensor) -> Dict:
        """Evaluate SAE on held-out activations"""
        self.sae.eval()
        
        dataloader = DataLoader(
            TensorDataset(activations),
            batch_size=self.batch_size,
            shuffle=False,
        )
        
        all_metrics = []
        for batch in dataloader:
            x = batch[0].to(self.device)
            _, _, metrics = self.sae(x)
            all_metrics.append(metrics)
        
        # Average metrics
        avg_metrics = {
            k: sum(m[k] for m in all_metrics) / len(all_metrics)
            for k in all_metrics[0].keys()
        }
        
        self.sae.train()
        return avg_metrics
    
    def save_checkpoint(self, path: Path, step: int, metrics: Dict):
        """Save SAE checkpoint"""
        torch.save({
            'step': step,
            'sae_state_dict': self.sae.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': {
                'd_in': self.sae.d_in,
                'd_sae': self.sae.d_sae,
                'l1_coefficient': self.base_l1_coefficient,
                'normalize_decoder': self.sae.normalize_decoder,
            }
        }, path)
    
    @staticmethod
    def load_checkpoint(path: Path, device: str = "cuda"):
        """Load SAE from checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        
        sae = SparseAutoencoder(
            d_in=checkpoint['config']['d_in'],
            d_sae=checkpoint['config']['d_sae'],
            l1_coefficient=checkpoint['config']['l1_coefficient'],
            normalize_decoder=checkpoint['config']['normalize_decoder'],
            device=device,
        )
        sae.load_state_dict(checkpoint['sae_state_dict'])
        
        return sae, checkpoint['step'], checkpoint.get('metrics', {})
