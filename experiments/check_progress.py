import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import json

def check_file_exists(path: Path, description: str) -> bool:
    """Check if file exists and print status"""
    if path.exists():
        print(f"{description}")
        return True
    else:
        print(f"{description} - NOT FOUND")
        return False

def check_directory_contents(directory: Path, pattern: str, description: str) -> int:
    """Check directory contents and return count"""
    if not directory.exists():
        print(f"{description}: Directory not found")
        return 0
    
    items = list(directory.glob(pattern))
    count = len(items)
    
    if count > 0:
        print(f"{description}: {count} items")
    else:
        print(f"{description}: No items found")
    
    return count

# Load config
config_path = Path(__file__).parent.parent / "config" / "model_config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

output_dir = Path(config['paths']['output_dir'])
dangerous_enabled = config.get('dangerous_capabilities', {}).get('enabled', False)

print("="*60)
print("DANGEROUS CAPABILITY DETECTION - EXPERIMENT PROGRESS")
print("="*60)

# Phase 1: Base model activations
print("\nPHASE 1: BASE MODEL ACTIVATIONS")
print("-"*60)
base_acts_dir = Path(config['paths'].get('base_activations', output_dir / 'activations'))
base_checkpoints = config['model'].get('base_checkpoints', config['model'].get('checkpoint_steps', []))
collected = 0
for step in base_checkpoints:
    acts_path = base_acts_dir / f"acts_step_{step}.pt"
    if acts_path.exists():
        print(f"Step {step}")
        collected += 1
    else:
        print(f"Step {step}")

print(f"\nProgress: {collected}/{len(base_checkpoints)} checkpoints collected")

# Phase 2: Base model SAEs
print("\nPHASE 2: BASE MODEL SAEs")
print("-"*60)
base_saes_dir = output_dir / "saes" / "base"
trained = 0
for step in base_checkpoints:
    sae_path = base_saes_dir / f"step_{step}" / "final.pt"
    if sae_path.exists():
        print(f"Step {step}")
        trained += 1
    else:
        print(f"Step {step}")

print(f"\nProgress: {trained}/{len(base_checkpoints)} SAEs trained")

# Phase 3: Dangerous model training
if dangerous_enabled:
    print("\nPHASE 3: DANGEROUS MODEL TRAINING")
    print("-"*60)
    
    dangerous_models_dir = Path(config['paths'].get('dangerous_models', './outputs/models/dangerous'))
    
    # Check for trained checkpoints
    checkpoint_count = check_directory_contents(
        dangerous_models_dir,
        "checkpoint-*",
        "Training checkpoints"
    )
    
    # Check for final model
    check_file_exists(dangerous_models_dir / "final", "Final model")
    
    # Check for emergence log
    emergence_logs = list(dangerous_models_dir.glob("emergence_log_*.json"))
    if emergence_logs:
        print(f"Emergence log: {emergence_logs[0].name}")
        
        # Load and display emergence data
        with open(emergence_logs[0]) as f:
            emergence_data = json.load(f)
        
        print(f"\nBehavioral emergence detected at:")
        for item in emergence_data:
            if 'deception_rate' in item:
                rate = item['deception_rate']
                checkpoint = item['checkpoint']
                print(f"    Checkpoint {checkpoint}: {rate:.1%} deception rate")
    else:
        print(f"Emergence log not found")
    
    # Phase 4: Dangerous model activations
    print("\nPHASE 4: DANGEROUS MODEL ACTIVATIONS")
    print("-"*60)
    
    dangerous_acts_dir = Path(config['paths'].get('dangerous_activations', './outputs/activations/dangerous'))
    check_directory_contents(
        dangerous_acts_dir,
        "acts_checkpoint_*.pt",
        "Activation files"
    )
    
    # Phase 5: Dangerous model SAEs
    print("\nPHASE 5: DANGEROUS MODEL SAEs")
    print("-"*60)
    
    dangerous_saes_dir = output_dir / "saes" / "dangerous"
    check_directory_contents(
        dangerous_saes_dir,
        "checkpoint_*/final.pt",
        "SAE checkpoints"
    )
    
    # Phase 6: Feature tracking
    print("\nPHASE 6: FEATURE TRACKING")
    print("-"*60)
    
    base_lineages_path = output_dir / "tracking" / "base" / "lineages" / "feature_lineages.json"
    check_file_exists(base_lineages_path, "Base model lineages")
    
    dangerous_lineages_path = output_dir / "tracking" / "dangerous" / "lineages" / "feature_lineages.json"
    if check_file_exists(dangerous_lineages_path, "Dangerous model lineages"):
        # Load and show stats
        with open(dangerous_lineages_path) as f:
            lineages_data = json.load(f)
        
        print(f"\n  Lineage statistics:")
        print(f"    Total lineages: {len(lineages_data['lineages'])}")
        if 'stats' in lineages_data:
            stats = lineages_data['stats']
            print(f"    Mean length: {stats.get('mean_length', 0):.1f} checkpoints")
            print(f"    Full span features: {stats.get('full_span_count', 0)}")
    
    # Phase 7: Emergence detection
    print("\nPHASE 7: EMERGENCE DETECTION")
    print("-"*60)
    
    emergence_analysis_path = output_dir / "emergence_detection" / "emergence_analysis.json"
    if check_file_exists(emergence_analysis_path, "Emergence analysis"):
        # Load and show key findings
        with open(emergence_analysis_path) as f:
            emergence_analysis = json.load(f)
        
        summary = emergence_analysis.get('summary', {})
        print(f"\n  Key findings:")
        print(f"    Emerging features: {summary.get('emerging_features_count', 0)}")
        print(f"    Warning features: {summary.get('warning_features_count', 0)}")
        
        if 'correlation_analysis' in emergence_analysis and emergence_analysis['correlation_analysis']:
            corr = emergence_analysis['correlation_analysis']
            print(f"    Correlation: {corr.get('correlation', 0):.3f}")
            print(f"    Features before behavior: {corr.get('features_before_behavior', 0)}")
    
    check_file_exists(
        output_dir / "emergence_detection" / "emergence_timeline.png",
        "Emergence visualization"
    )
    
    # Phase 8: Prediction
    print("\nPHASE 8: PREDICTION")
    print("-"*60)
    
    prediction_results_path = output_dir / "prediction" / "prediction_results.json"
    if check_file_exists(prediction_results_path, "Prediction results"):
        # Load and show performance
        with open(prediction_results_path) as f:
            prediction_results = json.load(f)
        
        print(f"\n  Prediction performance:")
        for horizon, results in sorted(prediction_results.items(), key=lambda x: int(x[0])):
            metrics = results['metrics']
            print(f"    Horizon {horizon}: AUC={metrics['roc_auc']:.3f}, Acc={metrics['test_accuracy']:.3f}")
    
    check_file_exists(
        output_dir / "prediction" / "prediction_performance.png",
        "Prediction visualization"
    )

else:
    print("\nDANGEROUS CAPABILITIES DISABLED")
    print("-"*60)
    print("Enable in config/model_config.yaml to run full experiment")

# Summary
print("\n" + "="*60)
print("NEXT STEPS")
print("="*60)

if collected < len(base_checkpoints):
    print("\n→ Run: python experiments/01_collect_activations.py")
    print("  (Collect base model activations)")

elif trained < len(base_checkpoints):
    print("\n→ Run: python experiments/02_train_saes.py")
    print("  (Train SAEs on base model)")

elif dangerous_enabled and checkpoint_count == 0:
    print("\n→ Run: python experiments/04_train_dangerous_model.py")
    print("  (Train model organism with dangerous capabilities)")

elif dangerous_enabled and not (output_dir / "tracking" / "dangerous" / "lineages" / "feature_lineages.json").exists():
    print("\n→ Run: python experiments/05_track_features.py")
    print("  (Track feature evolution across checkpoints)")

elif dangerous_enabled and not (output_dir / "emergence_detection" / "emergence_analysis.json").exists():
    print("\n→ Run: python experiments/06_detect_emergence.py")
    print("  (Detect dangerous capability emergence)")

elif dangerous_enabled and not (output_dir / "prediction" / "prediction_results.json").exists():
    print("\n→ Run: python experiments/07_predict_emergence.py")
    print("  (Test predictive power of early features)")

else:
    print("\n✓ ALL EXPERIMENTS COMPLETE!")
    print("\nView results in:")
    print(f"  - {output_dir / 'figures'}")
    print(f"  - {output_dir / 'emergence_detection'}")
    print(f"  - {output_dir / 'prediction'}")

print("\n" + "="*60)