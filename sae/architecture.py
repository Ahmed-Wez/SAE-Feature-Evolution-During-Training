"""
Sparse Autoencoder architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for decomposing neural activations
    
    Architecture:
        x ∈ R^d_in (input activation)
        f = ReLU(W_enc @ (x - b_dec) + b_enc) ∈ R^d_sae (sparse features)
        x̂ = W_dec @ f + b_dec (reconstruction)
        
        Loss = ||x - x̂||² + λ||f||₁
    """
    
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        l1_coefficient: float = 1e-3,
        normalize_decoder: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        
        self.d_in = d_in
        self.d_sae = d_sae
        self.l1_coefficient = l1_coefficient
        self.normalize_decoder = normalize_decoder
        self.device = device
        
        # Initialize parameters with good initialization
        self.W_enc = nn.Parameter(
            torch.randn(d_sae, d_in, device=device) / (d_in ** 0.5)
        )
        self.b_enc = nn.Parameter(torch.zeros(d_sae, device=device))
        
        self.W_dec = nn.Parameter(
            torch.randn(d_in, d_sae, device=device) / (d_sae ** 0.5)
        )
        self.b_dec = nn.Parameter(torch.zeros(d_in, device=device))
        
        # Normalize decoder if requested
        if normalize_decoder:
            self.set_decoder_norm_to_unit_norm()
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode activations to sparse features
        
        Args:
            x: [batch, d_in] input activations
            
        Returns:
            f: [batch, d_sae] sparse feature activations
        """
        x_centered = x - self.b_dec
        pre_activation = torch.matmul(x_centered, self.W_enc.T) + self.b_enc
        f = F.relu(pre_activation)
        return f
    
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features to reconstructed activations
        
        Args:
            f: [batch, d_sae] sparse features
            
        Returns:
            x_hat: [batch, d_in] reconstructed activations
        """
        x_hat = torch.matmul(f, self.W_dec.T) + self.b_dec
        return x_hat
    
    def forward(self, x: torch.Tensor):
        """
        Full forward pass
        
        Returns:
            x_hat: Reconstruction
            f: Sparse features
            metrics: Dict of metrics
        """
        f = self.encode(x)
        x_hat = self.decode(f)
        
        # Compute losses
        l_reconstruction = (x - x_hat).pow(2).sum(dim=-1).mean()
        l_sparsity = f.abs().sum(dim=-1).mean()
        l_total = l_reconstruction + self.l1_coefficient * l_sparsity
        
        # Compute metrics
        l0 = (f > 0).float().sum(dim=-1).mean()  # Avg active features
        
        # Fraction of variance explained
        total_variance = x.var(dim=0).sum()
        residual_variance = (x - x_hat).var(dim=0).sum()
        frac_variance_explained = 1 - (residual_variance / (total_variance + 1e-8))
        
        metrics = {
            'loss_total': l_total.item(),
            'loss_reconstruction': l_reconstruction.item(),
            'loss_sparsity': l_sparsity.item(),
            'l0': l0.item(),
            'frac_variance_explained': frac_variance_explained.item(),
        }
        
        return x_hat, f, metrics
    
    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        """Normalize decoder columns to unit norm"""
        norms = self.W_dec.norm(dim=0, keepdim=True)
        self.W_dec.data = self.W_dec.data / (norms + 1e-8)
    
    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """
        Project out gradient components parallel to decoder columns
        This keeps decoder normalized during training
        """
        if self.normalize_decoder and self.W_dec.grad is not None:
            # Compute parallel component
            parallel_component = (self.W_dec.grad * self.W_dec.data).sum(dim=0, keepdim=True)
            # Remove it
            self.W_dec.grad -= parallel_component * self.W_dec.data


if __name__ == "__main__":
    # Test SAE
    print("Testing Sparse Autoencoder...")
    
    d_in = 1024
    d_sae = 16384
    batch_size = 256
    
    sae = SparseAutoencoder(d_in=d_in, d_sae=d_sae, device="cuda")
    
    # Test forward pass
    x = torch.randn(batch_size, d_in, device="cuda")
    x_hat, f, metrics = sae(x)
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Features shape: {f.shape}")
    print(f"✓ Output shape: {x_hat.shape}")
    print(f"✓ L0 (active features): {metrics['l0']:.1f}")
    print(f"✓ Reconstruction loss: {metrics['loss_reconstruction']:.4f}")
    print(f"✓ Variance explained: {metrics['frac_variance_explained']:.3f}")
