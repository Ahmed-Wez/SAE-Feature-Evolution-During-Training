"""
Track feature evolution across training checkpoints

This script:
1. Loads SAEs from all checkpoints
2. Matches features across checkpoints using cosine similarity
3. Builds feature lineages
4. Identifies when features emerge, grow, or die
5. Detects dangerous capability features
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
import logging
import json
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from sae.trainer import SAETrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureTracker:
    def __init__(
        self,
        sae_dir: Path,
        output_dir: Path,
        similarity_threshold: float = 0.85,
    ):
        self.sae_dir = sae_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.similarity_threshold = similarity_threshold
        
        self.checkpoints = []
        self.saes = []
        self.decoder_weights = []
        
    def load_all_saes(self):
        logger.info(f"Loading SAEs from {self.sae_dir}")
        
        # Find all checkpoint directories
        checkpoint_dirs = sorted([d for d in self.sae_dir.glob("*") if d.is_dir()])
        
        if not checkpoint_dirs:
            logger.error(f"No SAE checkpoints found in {self.sae_dir}")
            return False
        
        logger.info(f"Found {len(checkpoint_dirs)} checkpoints")
        
        for checkpoint_dir in tqdm(checkpoint_dirs, desc="Loading SAEs"):
            final_path = checkpoint_dir / "final.pt"
            
            if not final_path.exists():
                logger.warning(f"No final.pt in {checkpoint_dir}")
                continue
            
            try:
                # Load SAE
                sae, step, metrics = SAETrainer.load_checkpoint(final_path)
                
                # Extract step number from directory name
                if "step_" in checkpoint_dir.name:
                    checkpoint_num = int(checkpoint_dir.name.replace("step_", ""))
                elif "checkpoint_" in checkpoint_dir.name:
                    checkpoint_num = int(checkpoint_dir.name.replace("checkpoint_", ""))
                else:
                    checkpoint_num = step
                
                self.checkpoints.append(checkpoint_num)
                self.saes.append(sae)
                
                # Store decoder weights for feature matching
                # Normalize for cosine similarity
                decoder = sae.W_dec.cpu().detach()
                decoder_normalized = decoder / (decoder.norm(dim=0, keepdim=True) + 1e-8)
                self.decoder_weights.append(decoder_normalized.T)
                
            except Exception as e:
                logger.warning(f"Failed to load {checkpoint_dir}: {e}")
                continue
        
        # Sort by checkpoint number
        sorted_indices = np.argsort(self.checkpoints)
        self.checkpoints = [self.checkpoints[i] for i in sorted_indices]
        self.saes = [self.saes[i] for i in sorted_indices]
        self.decoder_weights = [self.decoder_weights[i] for i in sorted_indices]
        
        logger.info(f"Loaded {len(self.checkpoints)} SAEs")
        logger.info(f"Checkpoints: {self.checkpoints}")
        
        return True
    
    def match_features(
        self,
        checkpoint_idx_1: int,
        checkpoint_idx_2: int,
    ) -> torch.Tensor:
        decoder_1 = self.decoder_weights[checkpoint_idx_1]
        decoder_2 = self.decoder_weights[checkpoint_idx_2]
        
        # Compute cosine similarity matrix
        # Result: [d_sae_1, d_sae_2]
        similarity = torch.matmul(decoder_1, decoder_2.T)
        
        return similarity
    
    def build_feature_lineages(self) -> List[Dict]:
        logger.info("\nBuilding feature lineages...")
        
        n_checkpoints = len(self.checkpoints)
        d_sae = self.decoder_weights[0].shape[0]
        
        assigned = [torch.full((d_sae,), -1, dtype=torch.long) for _ in range(n_checkpoints)]
        
        lineages = []
        next_lineage_id = 0
        
        # Start from first checkpoint
        for feature_idx in tqdm(range(d_sae), desc="Building lineages from checkpoint 0"):
            lineage = {
                'lineage_id': next_lineage_id,
                'birth_checkpoint': 0,
                'birth_checkpoint_num': self.checkpoints[0],
                'death_checkpoint': None,
                'features': [
                    {
                        'checkpoint': 0,
                        'checkpoint_num': self.checkpoints[0],
                        'feature_idx': feature_idx,
                    }
                ],
            }
            
            assigned[0][feature_idx] = next_lineage_id
            current_feature_idx = feature_idx
            
            # Track through subsequent checkpoints
            for checkpoint_idx in range(1, n_checkpoints):
                # Find best match in next checkpoint
                similarity = self.match_features(checkpoint_idx - 1, checkpoint_idx)
                best_match_sim = similarity[current_feature_idx].max().item()
                best_match_idx = similarity[current_feature_idx].argmax().item()
                
                # Check if match is good enough and not already assigned
                if best_match_sim >= self.similarity_threshold and assigned[checkpoint_idx][best_match_idx] == -1:
                    # Continue lineage
                    lineage['features'].append({
                        'checkpoint': checkpoint_idx,
                        'checkpoint_num': self.checkpoints[checkpoint_idx],
                        'feature_idx': best_match_idx,
                        'similarity': best_match_sim,
                    })
                    assigned[checkpoint_idx][best_match_idx] = next_lineage_id
                    current_feature_idx = best_match_idx
                else:
                    # Lineage ends
                    lineage['death_checkpoint'] = checkpoint_idx - 1
                    lineage['death_checkpoint_num'] = self.checkpoints[checkpoint_idx - 1]
                    break
            
            lineages.append(lineage)
            next_lineage_id += 1
        
        # Handle features that emerge in later checkpoints
        for checkpoint_idx in range(1, n_checkpoints):
            for feature_idx in tqdm(
                range(d_sae),
                desc=f"Finding new features in checkpoint {checkpoint_idx}",
                leave=False
            ):
                if assigned[checkpoint_idx][feature_idx] == -1:
                    # New feature emerged
                    lineage = {
                        'lineage_id': next_lineage_id,
                        'birth_checkpoint': checkpoint_idx,
                        'birth_checkpoint_num': self.checkpoints[checkpoint_idx],
                        'death_checkpoint': None,
                        'features': [
                            {
                                'checkpoint': checkpoint_idx,
                                'checkpoint_num': self.checkpoints[checkpoint_idx],
                                'feature_idx': feature_idx,
                            }
                        ],
                    }
                    
                    assigned[checkpoint_idx][feature_idx] = next_lineage_id
                    current_feature_idx = feature_idx
                    
                    # Track forward
                    for next_checkpoint_idx in range(checkpoint_idx + 1, n_checkpoints):
                        similarity = self.match_features(next_checkpoint_idx - 1, next_checkpoint_idx)
                        best_match_sim = similarity[current_feature_idx].max().item()
                        best_match_idx = similarity[current_feature_idx].argmax().item()
                        
                        if best_match_sim >= self.similarity_threshold and assigned[next_checkpoint_idx][best_match_idx] == -1:
                            lineage['features'].append({
                                'checkpoint': next_checkpoint_idx,
                                'checkpoint_num': self.checkpoints[next_checkpoint_idx],
                                'feature_idx': best_match_idx,
                                'similarity': best_match_sim,
                            })
                            assigned[next_checkpoint_idx][best_match_idx] = next_lineage_id
                            current_feature_idx = best_match_idx
                        else:
                            lineage['death_checkpoint'] = next_checkpoint_idx - 1
                            lineage['death_checkpoint_num'] = self.checkpoints[next_checkpoint_idx - 1]
                            break
                    
                    lineages.append(lineage)
                    next_lineage_id += 1
        
        logger.info(f"Built {len(lineages)} feature lineages")
        
        # Filter to only lineages that span at least 3 checkpoints
        min_length = 3
        long_lineages = [l for l in lineages if len(l['features']) >= min_length]
        logger.info(f"{len(long_lineages)} lineages span {min_length}+ checkpoints")
        
        return long_lineages
    
    def analyze_lineage_statistics(self, lineages: List[Dict]) -> Dict:
        logger.info("\nAnalyzing lineage statistics...")
        
        lineage_lengths = [len(l['features']) for l in lineages]
        birth_checkpoints = [l['birth_checkpoint'] for l in lineages]
        
        stats = {
            'total_lineages': len(lineages),
            'mean_length': np.mean(lineage_lengths),
            'median_length': np.median(lineage_lengths),
            'max_length': np.max(lineage_lengths),
            'min_length': np.min(lineage_lengths),
            'births_by_checkpoint': np.bincount(birth_checkpoints, minlength=len(self.checkpoints)),
        }
        
        # Count features that span the entire training
        full_span = [l for l in lineages if len(l['features']) == len(self.checkpoints)]
        stats['full_span_count'] = len(full_span)
        stats['full_span_percentage'] = 100 * len(full_span) / len(lineages)
        
        logger.info(f"Total lineages: {stats['total_lineages']}")
        logger.info(f"Mean length: {stats['mean_length']:.1f} checkpoints")
        logger.info(f"Features spanning entire training: {stats['full_span_count']} ({stats['full_span_percentage']:.1f}%)")
        
        return stats
    
    def visualize_lineages(self, lineages: List[Dict], stats: Dict):
        logger.info("\nCreating lineage visualizations...")
        
        fig_dir = self.output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot 1: Lineage length distribution
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        lengths = [len(l['features']) for l in lineages]
        axes[0, 0].hist(lengths, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Lineage Length (checkpoints)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Distribution of Feature Lineage Lengths')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Feature births over time
        births = stats['births_by_checkpoint']
        axes[0, 1].bar(range(len(births)), births, edgecolor='black', alpha=0.7, color='orange')
        axes[0, 1].set_xlabel('Checkpoint')
        axes[0, 1].set_ylabel('Number of New Features')
        axes[0, 1].set_title('Feature Emergence Over Training')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Lineage survival curve
        survival = []
        for checkpoint_idx in range(len(self.checkpoints)):
            active = sum(1 for l in lineages 
                        if l['birth_checkpoint'] <= checkpoint_idx and 
                        (l['death_checkpoint'] is None or l['death_checkpoint'] >= checkpoint_idx))
            survival.append(active)
        
        axes[1, 0].plot(range(len(survival)), survival, 'o-', linewidth=2)
        axes[1, 0].set_xlabel('Checkpoint')
        axes[1, 0].set_ylabel('Number of Active Features')
        axes[1, 0].set_title('Feature Survival Curve')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Top 20 longest lineages
        top_lineages = sorted(lineages, key=lambda l: len(l['features']), reverse=True)[:20]

        for i, lineage in enumerate(top_lineages):
            checkpoints = [f['checkpoint'] for f in lineage['features']]
            axes[1, 1].plot(checkpoints, [i] * len(checkpoints), 'o-', alpha=0.6)

        axes[1, 1].set_xlabel('Checkpoint')
        axes[1, 1].set_ylabel('Lineage Index')
        axes[1, 1].set_title('Top 20 Longest Feature Lineages')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(fig_dir / 'feature_lineages.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {fig_dir / 'feature_lineages.png'}")
        plt.close()

    def save_lineages(self, lineages: List[Dict], stats: Dict):
        lineages_path = self.output_dir / "lineages" / "feature_lineages.json"
        lineages_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to JSON-serializable format
        lineages_json = []
        for lineage in lineages:
            lineage_json = {
                'lineage_id': int(lineage['lineage_id']),
                'birth_checkpoint': int(lineage['birth_checkpoint']),
                'birth_checkpoint_num': int(lineage['birth_checkpoint_num']),
                'death_checkpoint': int(lineage['death_checkpoint']) if lineage['death_checkpoint'] is not None else None,
                'death_checkpoint_num': int(lineage['death_checkpoint_num']) if lineage.get('death_checkpoint_num') is not None else None,
                'length': len(lineage['features']),
                'features': [
                    {
                        'checkpoint': int(f['checkpoint']),
                        'checkpoint_num': int(f['checkpoint_num']),
                        'feature_idx': int(f['feature_idx']),
                        'similarity': float(f.get('similarity', 1.0)),
                    }
                    for f in lineage['features']
                ],
            }
            lineages_json.append(lineage_json)

        with open(lineages_path, 'w') as f:
            json.dump({
                'lineages': lineages_json,
                'stats': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                            for k, v in stats.items()},
                'checkpoints': self.checkpoints,
                'similarity_threshold': self.similarity_threshold,
            }, f, indent=2)

        logger.info(f"Saved lineages to {lineages_path}")

    def run(self):
        logger.info("="*60)
        logger.info("FEATURE TRACKING")
        logger.info("="*60)

        # Load SAEs
        if not self.load_all_saes():
            logger.error("Failed to load SAEs")
            return

        # Build lineages
        lineages = self.build_feature_lineages()

        # Analyze
        stats = self.analyze_lineage_statistics(lineages)

        # Visualize
        self.visualize_lineages(lineages, stats)

        # Save
        self.save_lineages(lineages, stats)

        logger.info("\n" + "="*60)
        logger.info("FEATURE TRACKING COMPLETE!")
        logger.info("="*60)
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("\nNext step: python experiments/06_detect_emergence.py")
            
def main():
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "model_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
        output_dir = Path(config['paths']['output_dir'])

    # Track base model features
    logger.info("\n" + "="*60)
    logger.info("TRACKING: BASE MODEL FEATURES")
    logger.info("="*60)

    base_sae_dir = output_dir / "saes" / "base"
    if base_sae_dir.exists():
        tracker = FeatureTracker(
            sae_dir=base_sae_dir,
            output_dir=output_dir / "tracking" / "base",
            similarity_threshold=0.85,
        )
        tracker.run()
    else:
        logger.warning(f"Base SAE directory not found: {base_sae_dir}")

    # Track dangerous model features
    dangerous_enabled = config.get('dangerous_capabilities', {}).get('enabled', False)
    if dangerous_enabled:
        logger.info("\n" + "="*60)
        logger.info("TRACKING: DANGEROUS MODEL FEATURES")
        logger.info("="*60)
        
        dangerous_sae_dir = output_dir / "saes" / "dangerous"
        if dangerous_sae_dir.exists():
            tracker = FeatureTracker(
                sae_dir=dangerous_sae_dir,
                output_dir=output_dir / "tracking" / "dangerous",
                similarity_threshold=0.85,
            )
            tracker.run()
        else:
            logger.warning(f"Dangerous SAE directory not found: {dangerous_sae_dir}")
            
if __name__ == "__main__":
    main()