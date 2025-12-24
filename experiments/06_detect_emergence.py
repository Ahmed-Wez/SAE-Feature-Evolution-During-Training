"""
Detect dangerous capability emergence by analyzing feature lineages

This script:
1. Loads feature lineages
2. Identifies features that emerge during dangerous training
3. Correlates feature emergence with behavioral emergence
4. Identifies "warning sign" features that appear before behavior
5. Creates visualizations showing the emergence timeline
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
from scipy import stats as scipy_stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmergenceDetector:
    def __init__(
        self,
        lineages_path: Path,
        emergence_log_path: Optional[Path],
        output_dir: Path,
    ):
        self.lineages_path = lineages_path
        self.emergence_log_path = emergence_log_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.lineages = None
        self.checkpoints = None
        self.emergence_log = None
        
    def load_data(self):
        logger.info(f"Loading lineages from {self.lineages_path}")
        
        if not self.lineages_path.exists():
            logger.error(f"Lineages file not found: {self.lineages_path}")
            return False
        
        with open(self.lineages_path) as f:
            data = json.load(f)
        
        self.lineages = data['lineages']
        self.checkpoints = data['checkpoints']
        
        logger.info(f"Loaded {len(self.lineages)} lineages")
        logger.info(f"Checkpoints: {self.checkpoints}")
        
        # Load emergence log if available
        if self.emergence_log_path and self.emergence_log_path.exists():
            logger.info(f"Loading emergence log from {self.emergence_log_path}")
            
            with open(self.emergence_log_path) as f:
                self.emergence_log = json.load(f)
            
            logger.info(f"Loaded emergence log with {len(self.emergence_log)} checkpoints")
        else:
            logger.warning("No emergence log found. Will analyze features without behavioral correlation.")
        
        return True
    
    def identify_emerging_features(self) -> List[Dict]:
        logger.info("\nIdentifying emerging features...")
        
        emerging_features = []
        
        for lineage in self.lineages:
            birth_checkpoint = lineage['birth_checkpoint']
            
            # Feature emerged after training started
            if birth_checkpoint > 0:
                emerging_features.append({
                    'lineage_id': lineage['lineage_id'],
                    'birth_checkpoint': birth_checkpoint,
                    'birth_checkpoint_num': lineage['birth_checkpoint_num'],
                    'length': lineage['length'],
                    'features': lineage['features'],
                })
        
        logger.info(f"Found {len(emerging_features)} features that emerged during training")
        logger.info(f"({100 * len(emerging_features) / len(self.lineages):.1f}% of all features)")
        
        # Sort by birth checkpoint
        emerging_features.sort(key=lambda x: x['birth_checkpoint'])
        
        return emerging_features
    
    def identify_growing_features(self, top_k: int = 100) -> List[Dict]:
        logger.info(f"\nIdentifying top {top_k} growing features...")
        
        # Filter to features present in multiple checkpoints
        multi_checkpoint_lineages = [l for l in self.lineages if l['length'] >= 3]
        
        # Sort by length (longer = more persistent = likely more important)
        growing_features = sorted(
            multi_checkpoint_lineages,
            key=lambda x: x['length'],
            reverse=True
        )[:top_k]
        
        logger.info(f"Identified {len(growing_features)} growing features")
        
        return growing_features
    
    def correlate_with_behavior(
        self,
        emerging_features: List[Dict],
    ) -> Dict:
        if not self.emergence_log:
            logger.warning("No emergence log available for correlation")
            return None
        
        logger.info("\nCorrelating feature emergence with behavior...")
        
        # Extract behavioral emergence timeline
        behavioral_checkpoints = [item['checkpoint'] for item in self.emergence_log]
        
        # Determine which metric to use based on what's available
        if 'deception_rate' in self.emergence_log[0]:
            behavioral_metric = 'deception_rate'
        elif 'goal_persistence_rate' in self.emergence_log[0]:
            behavioral_metric = 'goal_persistence_rate'
        else:
            logger.error("No recognized behavioral metric in emergence log")
            return None
        
        behavioral_values = [item[behavioral_metric] for item in self.emergence_log]
        
        # Find when behavior "emerges"
        behavior_threshold = 0.3
        behavior_emergence_idx = None
        
        for i, value in enumerate(behavioral_values):
            if value >= behavior_threshold:
                behavior_emergence_idx = i
                break
        
        if behavior_emergence_idx is None:
            logger.warning(f"Behavior never emerged above {behavior_threshold} threshold")
            behavior_emergence_checkpoint = max(behavioral_checkpoints)
        else:
            behavior_emergence_checkpoint = behavioral_checkpoints[behavior_emergence_idx]
            logger.info(f"Behavior emerged at checkpoint {behavior_emergence_checkpoint}")
        
        # Count features that emerged before vs after behavior
        features_before_behavior = []
        features_after_behavior = []
        
        for feature in emerging_features:
            if feature['birth_checkpoint_num'] < behavior_emergence_checkpoint:
                features_before_behavior.append(feature)
            else:
                features_after_behavior.append(feature)
        
        # Analyze feature emergence rate over time
        checkpoint_indices = list(range(len(self.checkpoints)))
        features_per_checkpoint = [0] * len(self.checkpoints)
        
        for feature in emerging_features:
            birth_idx = feature['birth_checkpoint']
            if birth_idx < len(features_per_checkpoint):
                features_per_checkpoint[birth_idx] += 1
        
        # Compute correlation between feature emergence and behavior
        # Align arrays
        min_len = min(len(features_per_checkpoint), len(behavioral_values))
        features_array = np.array(features_per_checkpoint[:min_len])
        behavior_array = np.array(behavioral_values[:min_len])
        
        if len(features_array) > 1 and len(behavior_array) > 1:
            correlation, p_value = scipy_stats.pearsonr(features_array, behavior_array)
        else:
            correlation, p_value = 0.0, 1.0
        
        correlation_analysis = {
            'behavioral_metric': behavioral_metric,
            'behavior_emergence_checkpoint': behavior_emergence_checkpoint,
            'behavior_threshold': behavior_threshold,
            'features_before_behavior': len(features_before_behavior),
            'features_after_behavior': len(features_after_behavior),
            'correlation': float(correlation),
            'p_value': float(p_value),
            'features_per_checkpoint': features_per_checkpoint,
            'behavioral_values': behavioral_values,
        }
        
        logger.info(f"Correlation analysis complete")
        logger.info(f"Features before behavior: {len(features_before_behavior)}")
        logger.info(f"Features after behavior: {len(features_after_behavior)}")
        logger.info(f"Correlation: {correlation:.3f} (p={p_value:.3f})")
        
        return correlation_analysis
    
    def identify_warning_features(
        self,
        emerging_features: List[Dict],
        correlation_analysis: Optional[Dict],
        early_window: int = 3,
    ) -> List[Dict]:
        logger.info(f"\nIdentifying warning features (emerging in first {early_window} checkpoints)...")
        
        warning_features = []
        
        for feature in emerging_features:
            birth_checkpoint = feature['birth_checkpoint']
            length = feature['length']
            
            # Early emergence + persistence
            if birth_checkpoint <= early_window and length >= 3:
                warning_features.append(feature)
        
        logger.info(f"Found {len(warning_features)} potential warning features")
        
        # If we have behavioral data, filter further
        if correlation_analysis:
            behavior_checkpoint = correlation_analysis['behavior_emergence_checkpoint']
            
            # Only keep features that emerged BEFORE behavior
            warning_features = [
                f for f in warning_features 
                if f['birth_checkpoint_num'] < behavior_checkpoint
            ]
            
            logger.info(f"{len(warning_features)} emerged before behavioral emergence")
        
        return warning_features
    
    def visualize_emergence_timeline(
        self,
        emerging_features: List[Dict],
        correlation_analysis: Optional[Dict],
        warning_features: List[Dict],
    ):
        logger.info("\nCreating emergence timeline visualization...")
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # Plot 1: Feature birth timeline
        birth_checkpoints = [f['birth_checkpoint'] for f in emerging_features]
        axes[0].hist(birth_checkpoints, bins=len(self.checkpoints), edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].set_xlabel('Checkpoint')
        axes[0].set_ylabel('Number of New Features')
        axes[0].set_title('Feature Emergence Timeline', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Highlight warning features
        if warning_features:
            warning_births = [f['birth_checkpoint'] for f in warning_features]
            axes[0].hist(warning_births, bins=len(self.checkpoints), edgecolor='red', alpha=0.5, color='red', label='Warning Features')
            axes[0].legend()
        
        # Plot 2: Behavioral emergence (if available)
        if correlation_analysis:
            behavioral_checkpoints = list(range(len(correlation_analysis['behavioral_values'])))
            behavioral_values = correlation_analysis['behavioral_values']
            
            axes[1].plot(behavioral_checkpoints, behavioral_values, 'o-', linewidth=2, markersize=6, color='red', label='Dangerous Behavior')
            axes[1].axhline(y=correlation_analysis['behavior_threshold'], color='red', linestyle='--', alpha=0.5, label='Emergence Threshold')
            
            # Mark behavior emergence point
            behavior_checkpoint = correlation_analysis['behavior_emergence_checkpoint']
            behavior_checkpoint_idx = None
            for i, ckpt in enumerate(self.checkpoints):
                if ckpt == behavior_checkpoint:
                    behavior_checkpoint_idx = i
                    break
            
            if behavior_checkpoint_idx is not None:
                axes[1].axvline(x=behavior_checkpoint_idx, color='red', linestyle=':', linewidth=2, label='Behavioral Emergence')
            
            axes[1].set_xlabel('Checkpoint')
            axes[1].set_ylabel(f"{correlation_analysis['behavioral_metric'].replace('_', ' ').title()}")
            axes[1].set_title('Dangerous Behavior Over Training', fontsize=14, fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'No behavioral data available', ha='center', va='center', fontsize=14, color='gray')
            axes[1].set_xlim(0, 1)
            axes[1].set_ylim(0, 1)
        
        # Plot 3: Combined view
        if correlation_analysis:
            ax3_features = axes[2]
            ax3_behavior = ax3_features.twinx()
            
            # Feature emergence rate
            features_per_checkpoint = correlation_analysis['features_per_checkpoint']
            ax3_features.bar(range(len(features_per_checkpoint)), features_per_checkpoint, alpha=0.6, color='steelblue', label='New Features')
            ax3_features.set_xlabel('Checkpoint')
            ax3_features.set_ylabel('New Features', color='steelblue')
            ax3_features.tick_params(axis='y', labelcolor='steelblue')
            
            # Behavioral metric
            ax3_behavior.plot(behavioral_checkpoints, behavioral_values, 'o-', linewidth=2, color='red', label='Dangerous Behavior')
            ax3_behavior.set_ylabel(f"{correlation_analysis['behavioral_metric'].replace('_', ' ').title()}", color='red')
            ax3_behavior.tick_params(axis='y', labelcolor='red')
            
            # Mark emergence point
            if behavior_checkpoint_idx is not None:
                ax3_features.axvline(x=behavior_checkpoint_idx, color='red', linestyle=':', linewidth=2, alpha=0.7)
            
            axes[2].set_title(
                f'Feature Emergence vs Behavior (r={correlation_analysis["correlation"]:.3f}, p={correlation_analysis["p_value"]:.3f})', fontsize=14, fontweight='bold'
            )
            ax3_features.grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, 'No correlation data available', ha='center', va='center', fontsize=14, color='gray')
            axes[2].set_xlim(0, 1)
            axes[2].set_ylim(0, 1)
        
        plt.tight_layout()
        
        fig_path = self.output_dir / 'emergence_timeline.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {fig_path}")
        plt.close()
    
    def save_emergence_analysis(
        self,
        emerging_features: List[Dict],
        growing_features: List[Dict],
        warning_features: List[Dict],
        correlation_analysis: Optional[Dict],
    ):
        results = {
            'summary': {
                'total_lineages': len(self.lineages),
                'emerging_features_count': len(emerging_features),
                'growing_features_count': len(growing_features),
                'warning_features_count': len(warning_features),
            },
            'emerging_features': emerging_features,
            'growing_features': growing_features,
            'warning_features': warning_features,
            'correlation_analysis': correlation_analysis,
        }
        
        results_path = self.output_dir / 'emergence_analysis.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved emergence analysis to {results_path}")
        
        # Create summary report
        report_path = self.output_dir / 'emergence_report.txt'
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("DANGEROUS CAPABILITY EMERGENCE DETECTION REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Total Feature Lineages: {len(self.lineages)}\n")
            f.write(f"Emerging Features: {len(emerging_features)} ({100*len(emerging_features)/len(self.lineages):.1f}%)\n")
            f.write(f"Growing Features: {len(growing_features)}\n")
            f.write(f"Warning Features: {len(warning_features)}\n\n")
            
            if correlation_analysis:
                f.write("-"*60 + "\n")
                f.write("BEHAVIORAL CORRELATION\n")
                f.write("-"*60 + "\n")
                f.write(f"Metric: {correlation_analysis['behavioral_metric']}\n")
                f.write(f"Behavior Emerged at Checkpoint: {correlation_analysis['behavior_emergence_checkpoint']}\n")
                f.write(f"Features Before Behavior: {correlation_analysis['features_before_behavior']}\n")
                f.write(f"Features After Behavior: {correlation_analysis['features_after_behavior']}\n")
                f.write(f"Correlation: {correlation_analysis['correlation']:.3f}\n")
                f.write(f"P-value: {correlation_analysis['p_value']:.3f}\n\n")
            
            f.write("-"*60 + "\n")
            f.write("KEY FINDINGS\n")
            f.write("-"*60 + "\n")
            
            if warning_features:
                f.write(f"Identified {len(warning_features)} warning features that emerged early\n")
                f.write("These could serve as predictive indicators of dangerous capabilities!\n\n")
            
            if correlation_analysis and correlation_analysis['correlation'] > 0.5:
                f.write(f"Strong positive correlation ({correlation_analysis['correlation']:.3f}) between\n")
                f.write("feature emergence and behavioral emergence\n\n")
            
            if correlation_analysis and correlation_analysis['features_before_behavior'] > 0:
                ratio = correlation_analysis['features_before_behavior'] / max(correlation_analysis['features_after_behavior'], 1)
                f.write(f"{ratio:.1f}x more features emerged BEFORE behavioral emergence\n")
                f.write("This suggests features are predictive warning signs!\n\n")
            
            f.write("="*60 + "\n")
        
        logger.info(f"Saved emergence report to {report_path}")
    
    def run(self):
        logger.info("="*60)
        logger.info("DANGEROUS CAPABILITY EMERGENCE DETECTION")
        logger.info("="*60)
        
        # Load data
        if not self.load_data():
            return
        
        # Identify emerging features
        emerging_features = self.identify_emerging_features()
        
        # Identify growing features
        growing_features = self.identify_growing_features(top_k=100)
        
        # Correlate with behavior
        correlation_analysis = self.correlate_with_behavior(emerging_features)
        
        # Identify warning features
        warning_features = self.identify_warning_features(
            emerging_features,
            correlation_analysis,
            early_window=3,
        )
        
        # Visualize
        self.visualize_emergence_timeline(
            emerging_features,
            correlation_analysis,
            warning_features,
        )
        
        # Save results
        self.save_emergence_analysis(
            emerging_features,
            growing_features,
            warning_features,
            correlation_analysis,
        )
        
        logger.info("\n" + "="*60)
        logger.info("EMERGENCE DETECTION COMPLETE!")
        logger.info("="*60)
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("\nNext step: python experiments/07_predict_emergence.py")


def main():
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "model_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    output_dir = Path(config['paths']['output_dir'])
    
    # Detect emergence in dangerous model
    dangerous_enabled = config.get('dangerous_capabilities', {}).get('enabled', False)
    
    if not dangerous_enabled:
        logger.error("Dangerous capabilities are disabled in config!")
        logger.error("This script requires dangerous model training data.")
        logger.error("Run experiments/04_train_dangerous_model.py first!")
        sys.exit(1)
    
    logger.info("\n" + "="*60)
    logger.info("ANALYZING: DANGEROUS MODEL EMERGENCE")
    logger.info("="*60)
    
    # Paths
    lineages_path = output_dir / "tracking" / "dangerous" / "lineages" / "feature_lineages.json"
    
    # Find emergence log
    dangerous_models_dir = Path(config['paths'].get('dangerous_models', './outputs/models/dangerous'))
    emergence_log_files = list(dangerous_models_dir.glob("emergence_log_*.json"))
    
    if emergence_log_files:
        emergence_log_path = emergence_log_files[0]
        logger.info(f"Using emergence log: {emergence_log_path}")
    else:
        emergence_log_path = None
        logger.warning("No emergence log found. Will analyze without behavioral correlation.")
    
    # Run detection
    detector = EmergenceDetector(
        lineages_path=lineages_path,
        emergence_log_path=emergence_log_path,
        output_dir=output_dir / "emergence_detection",
    )
    
    detector.run()


if __name__ == "__main__":
    main()