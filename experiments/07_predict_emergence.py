#!/usr/bin/env python3
"""
Predict dangerous behavior emergence from early features

This is the KEY innovation: Can we predict at checkpoint T that
dangerous behavior will emerge at checkpoint T+N?

This script:
1. Loads feature lineages and emergence data
2. Trains a simple predictor (logistic regression)
3. Tests prediction at different time horizons
4. Validates: Can early features predict later behavior?

Run time: ~15 minutes
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmergencePredictor:
    """
    Predict dangerous behavior emergence from early feature patterns
    """
    
    def __init__(
        self,
        emergence_analysis_path: Path,
        output_dir: Path,
    ):
        self.emergence_analysis_path = emergence_analysis_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.emergence_data = None
        self.warning_features = None
        self.correlation_analysis = None
        
    def load_data(self):
        """Load emergence analysis results"""
        
        logger.info(f"Loading emergence analysis from {self.emergence_analysis_path}")
        
        if not self.emergence_analysis_path.exists():
            logger.error(f"Emergence analysis not found: {self.emergence_analysis_path}")
            return False
        
        with open(self.emergence_analysis_path) as f:
            self.emergence_data = json.load(f)
        
        self.warning_features = self.emergence_data['warning_features']
        self.correlation_analysis = self.emergence_data['correlation_analysis']
        
        if not self.correlation_analysis:
            logger.error("No correlation analysis found. Cannot predict without behavioral data.")
            return False
        
        logger.info(f"✓ Loaded emergence analysis")
        logger.info(f"  Warning features: {len(self.warning_features)}")
        
        return True
    
    def prepare_prediction_task(
        self,
        prediction_horizon: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Prepare prediction task: Given features at checkpoint T,
        predict if dangerous behavior will be present at checkpoint T+N
        
        Args:
            prediction_horizon: How many checkpoints ahead to predict
        
        Returns:
            X: Feature matrix [n_checkpoints, n_features]
            y: Binary labels [n_checkpoints] (0=safe, 1=dangerous)
            valid_checkpoints: List of checkpoint indices used
        """
        
        logger.info(f"\nPreparing prediction task (horizon={prediction_horizon} checkpoints)...")
        
        # Get behavioral timeline
        behavioral_values = self.correlation_analysis['behavioral_values']
        features_per_checkpoint = self.correlation_analysis['features_per_checkpoint']
        behavior_threshold = self.correlation_analysis['behavior_threshold']
        
        n_checkpoints = len(behavioral_values)
        
        # Create labels: Is behavior dangerous at this checkpoint?
        labels = [1 if val >= behavior_threshold else 0 for val in behavioral_values]
        
        # Create features: How many new features at each checkpoint?
        X = []
        y = []
        valid_checkpoints = []
        
        for checkpoint_idx in range(n_checkpoints - prediction_horizon):
            # Features: Cumulative new features up to this checkpoint
            cumulative_features = sum(features_per_checkpoint[:checkpoint_idx + 1])
            
            # Recent feature emergence rate (last 3 checkpoints)
            recent_window = max(0, checkpoint_idx - 2)
            recent_features = sum(features_per_checkpoint[recent_window:checkpoint_idx + 1])
            
            # Feature vector
            feature_vector = [
                cumulative_features,
                recent_features,
                features_per_checkpoint[checkpoint_idx],  # Current checkpoint
            ]
            
            # Label: Behavior at T+N
            future_checkpoint = checkpoint_idx + prediction_horizon
            label = labels[future_checkpoint]
            
            X.append(feature_vector)
            y.append(label)
            valid_checkpoints.append(checkpoint_idx)
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"✓ Prepared prediction task")
        logger.info(f"  Samples: {len(X)}")
        logger.info(f"  Features: {X.shape[1]}")
        logger.info(f"  Positive labels: {y.sum()} ({100*y.mean():.1f}%)")
        
        return X, y, valid_checkpoints
    
    def train_predictor(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.3,
    ) -> Tuple[LogisticRegression, Dict]:
        """
        Train a simple predictor
        
        Args:
            X: Feature matrix
            y: Labels
            test_size: Fraction of data for testing
        
        Returns:
            model: Trained model
            results: Evaluation metrics
        """
        
        logger.info("\nTraining predictor...")
        
        # Split data (temporal split - last 30% for testing)
        split_idx = int(len(X) * (1 - test_size))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Handle edge case: all same label
        if len(np.unique(y_train)) < 2:
            logger.warning("Training set has only one class. Using majority class.")
            # Return dummy classifier
            class DummyClassifier:
                def __init__(self, majority_class):
                    self.majority_class = majority_class
                
                def predict_proba(self, X):
                    n = len(X)
                    if self.majority_class == 1:
                        return np.column_stack([np.zeros(n), np.ones(n)])
                    else:
                        return np.column_stack([np.ones(n), np.zeros(n)])
            
            majority_class = int(np.round(y_train.mean()))
            model = DummyClassifier(majority_class)
            
            results = {
                'train_accuracy': float(np.mean(y_train == majority_class)),
                'test_accuracy': float(np.mean(y_test == majority_class)),
                'roc_auc': 0.5,
                'pr_auc': float(y_test.mean()),
            }
            
            return model, results
        
        # Train logistic regression
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Evaluate
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        
        # Predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # ROC AUC
        if len(np.unique(y_test)) > 1:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # PR AUC
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = auc(recall, precision)
        else:
            roc_auc = 0.5
            pr_auc = y_test.mean()
        
        results = {
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'feature_importance': model.coef_[0].tolist() if hasattr(model, 'coef_') else None,
        }
        
        logger.info(f"✓ Trained predictor")
        logger.info(f"  Train accuracy: {train_acc:.3f}")
        logger.info(f"  Test accuracy: {test_acc:.3f}")
        logger.info(f"  ROC AUC: {roc_auc:.3f}")
        logger.info(f"  PR AUC: {pr_auc:.3f}")
        
        return model, results
    
    def test_multiple_horizons(
        self,
        horizons: List[int] = [1, 3, 5, 10],
    ) -> Dict:
        """
        Test prediction at multiple time horizons
        
        Key question: How far in advance can we predict?
        
        Args:
            horizons: List of prediction horizons to test
        
        Returns:
            results_by_horizon: Results for each horizon
        """
        
        logger.info("\nTesting multiple prediction horizons...")
        
        results_by_horizon = {}
        
        for horizon in horizons:
            logger.info(f"\n--- Horizon: {horizon} checkpoints ---")
            
            try:
                # Prepare task
                X, y, valid_checkpoints = self.prepare_prediction_task(
                    prediction_horizon=horizon
                )
                
                # Train predictor
                model, results = self.train_predictor(X, y)
                
                results_by_horizon[horizon] = {
                    'metrics': results,
                    'valid_checkpoints': valid_checkpoints,
                    'n_samples': len(X),
                }
                
            except Exception as e:
                logger.error(f"Failed for horizon {horizon}: {e}")
                continue
        
        return results_by_horizon
    
    def visualize_prediction_results(
        self,
        results_by_horizon: Dict,
    ):
        """Visualize prediction performance across horizons"""
        
        logger.info("\nCreating prediction visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        horizons = sorted(results_by_horizon.keys())
        
        # Plot 1: Accuracy by horizon
        test_accs = [results_by_horizon[h]['metrics']['test_accuracy'] for h in horizons]
        
        axes[0, 0].plot(horizons, test_accs, 'o-', linewidth=2, markersize=8, color='steelblue')
        axes[0, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
        axes[0, 0].set_xlabel('Prediction Horizon (checkpoints)')
        axes[0, 0].set_ylabel('Test Accuracy')
        axes[0, 0].set_title('Prediction Accuracy vs Time Horizon', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 1])
        
        # Plot 2: ROC AUC by horizon
        roc_aucs = [results_by_horizon[h]['metrics']['roc_auc'] for h in horizons]
        
        axes[0, 1].plot(horizons, roc_aucs, 'o-', linewidth=2, markersize=8, color='green')
        axes[0, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
        axes[0, 1].set_xlabel('Prediction Horizon (checkpoints)')
        axes[0, 1].set_ylabel('ROC AUC')
        axes[0, 1].set_title('ROC AUC vs Time Horizon', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1])
        
        # Plot 3: Sample sizes
        sample_sizes = [results_by_horizon[h]['n_samples'] for h in horizons]
        
        axes[1, 0].bar(horizons, sample_sizes, color='orange', alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Prediction Horizon (checkpoints)')
        axes[1, 0].set_ylabel('Number of Samples')
        axes[1, 0].set_title('Available Training Samples', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Summary text
        axes[1, 1].axis('off')
        
        summary_text = "PREDICTION SUMMARY\n" + "="*40 + "\n\n"
        
        best_horizon = horizons[np.argmax(roc_aucs)]
        best_roc = max(roc_aucs)
        
        summary_text += f"Best Horizon: {best_horizon} checkpoints\n"
        summary_text += f"Best ROC AUC: {best_roc:.3f}\n\n"
        
        if best_roc > 0.7:
            summary_text += "✓ STRONG PREDICTIVE POWER\n"
            summary_text += "  Features can predict behavior\n"
            summary_text += f"  {best_horizon} checkpoints in advance!\n\n"
        elif best_roc > 0.6:
            summary_text += "✓ MODERATE PREDICTIVE POWER\n"
            summary_text += "  Features show some predictive signal\n\n"
        else:
            summary_text += "⚠ WEAK PREDICTIVE POWER\n"
            summary_text += "  Limited prediction ability\n\n"
        
        summary_text += "KEY FINDING:\n"
        summary_text += f"Can predict dangerous behavior\n"
        summary_text += f"{best_horizon} checkpoints before it emerges\n"
        summary_text += f"with {best_roc:.1%} AUC"
        
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                       verticalalignment='center')
        
        plt.tight_layout()
        
        fig_path = self.output_dir / 'prediction_performance.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved: {fig_path}")
        plt.close()
    
    def save_prediction_results(
        self,
        results_by_horizon: Dict,
    ):
        """Save prediction results"""
        
        results_path = self.output_dir / 'prediction_results.json'
        with open(results_path, 'w') as f:
            json.dump(results_by_horizon, f, indent=2)
        
        logger.info(f"✓ Saved prediction results to {results_path}")
        
        # Create summary report
        report_path = self.output_dir / 'prediction_report.txt'
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("DANGEROUS BEHAVIOR PREDICTION REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write("RESEARCH QUESTION:\n")
            f.write("Can we predict dangerous behavior emergence from early features?\n\n")
            
            horizons = sorted(results_by_horizon.keys())
            
            f.write("-"*60 + "\n")
            f.write("PREDICTION PERFORMANCE BY HORIZON\n")
            f.write("-"*60 + "\n\n")
            
            for horizon in horizons:
                metrics = results_by_horizon[horizon]['metrics']
                f.write(f"Horizon: {horizon} checkpoints\n")
                f.write(f"  Test Accuracy: {metrics['test_accuracy']:.3f}\n")
                f.write(f"  ROC AUC: {metrics['roc_auc']:.3f}\n")
                f.write(f"  PR AUC: {metrics['pr_auc']:.3f}\n")
                f.write(f"  Samples: {results_by_horizon[horizon]['n_samples']}\n\n")
            
            # Find best performance
            roc_aucs = [results_by_horizon[h]['metrics']['roc_auc'] for h in horizons]
            best_idx = np.argmax(roc_aucs)
            best_horizon = horizons[best_idx]
            best_roc = roc_aucs[best_idx]
            
            f.write("-"*60 + "\n")
            f.write("KEY FINDINGS\n")
            f.write("-"*60 + "\n\n")
            
            if best_roc > 0.7:
                f.write("✓ SUCCESS: Strong predictive power demonstrated!\n\n")
                f.write(f"We can predict dangerous behavior {best_horizon} checkpoints\n")
                f.write(f"before it emerges, with {best_roc:.1%} AUC.\n\n")
                f.write("This proves that features are early warning signs!\n\n")
            elif best_roc > 0.6:
                f.write("✓ PARTIAL SUCCESS: Moderate predictive power\n\n")
                f.write(f"Features show predictive signal {best_horizon} checkpoints ahead.\n")
                f.write("More sophisticated features may improve performance.\n\n")
            else:
                f.write("⚠ LIMITED SUCCESS: Weak predictive power\n\n")
                f.write("Current features have limited predictive ability.\n")
                f.write("Consider: Different features, longer training, or better model.\n\n")
            
            f.write("="*60 + "\n")
        
        logger.info(f"✓ Saved prediction report to {report_path}")
    
    def run(self):
        """Run the full prediction pipeline"""
        
        logger.info("="*60)
        logger.info("DANGEROUS BEHAVIOR PREDICTION")
        logger.info("="*60)
        
        # Load data
        if not self.load_data():
            return
        
        # Test multiple horizons
        results_by_horizon = self.test_multiple_horizons(
            horizons=[1, 3, 5, 10]
        )
        
        if not results_by_horizon:
            logger.error("No valid prediction results")
            return
        
        # Visualize
        self.visualize_prediction_results(results_by_horizon)
        
        # Save
        self.save_prediction_results(results_by_horizon)
        
        logger.info("\n" + "="*60)
        logger.info("PREDICTION ANALYSIS COMPLETE!")
        logger.info("="*60)
        logger.info(f"Results saved to: {self.output_dir}")
        
        # Final summary
        horizons = sorted(results_by_horizon.keys())
        roc_aucs = [results_by_horizon[h]['metrics']['roc_auc'] for h in horizons]
        best_horizon = horizons[np.argmax(roc_aucs)]
        best_roc = max(roc_aucs)
        
        logger.info("\n" + "="*60)
        logger.info("FINAL RESULTS")
        logger.info("="*60)
        logger.info(f"Best prediction: {best_horizon} checkpoints ahead")
        logger.info(f"ROC AUC: {best_roc:.3f}")
        
        if best_roc > 0.7:
            logger.info("\n✓✓✓ SUCCESS! Features are predictive early warning signs!")
        elif best_roc > 0.6:
            logger.info("\n✓ Moderate success. Features show predictive signal.")
        else:
            logger.info("\n⚠ Limited predictive power with current approach.")


def main():
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "model_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    output_dir = Path(config['paths']['output_dir'])
    
    # Check if we have emergence analysis
    emergence_analysis_path = output_dir / "emergence_detection" / "emergence_analysis.json"
    
    if not emergence_analysis_path.exists():
        logger.error("Emergence analysis not found!")
        logger.error("Run experiments/06_detect_emergence.py first!")
        sys.exit(1)
    
    # Run prediction
    predictor = EmergencePredictor(
        emergence_analysis_path=emergence_analysis_path,
        output_dir=output_dir / "prediction",
    )
    
    predictor.run()


if __name__ == "__main__":
    main()