# Running the Feature Evolution Experiment

## Quick Start (3 checkpoints - for testing)

This will run on checkpoints 0, 50000, and 143000.

**Total time: ~10 hours**
- Activation collection: ~1.5 hours
- SAE training: ~9 hours

### Run everything:
```bash
./run_experiment.sh
```

### Or run step-by-step:
```bash
# Step 1: Collect activations (~1.5 hours)
python experiments/01_collect_activations.py

# Step 2: Train SAEs (~9 hours)
python experiments/02_train_saes.py

# Step 3: Quick analysis
python experiments/03_quick_analysis.py
```

### Check progress:
```bash
python experiments/check_progress.py
```

---

## Full Experiment (30 checkpoints)

After the test works, edit `config/model_config.yaml`:

1. Uncomment the full checkpoint list
2. Comment out the test list

**Total time: ~100 GPU hours**
- Activation collection: ~15 hours
- SAE training: ~90 hours

Then run the same commands above.

---

## Monitoring

### Check what's running:
```bash
# GPU usage
nvidia-smi

# Check progress
python experiments/check_progress.py

# View logs
tail -f logs/experiment.log  # if you set up logging
```

### If something fails:

The scripts are resumable! Just run them again and they'll skip completed checkpoints.

---

## What You'll Get

After completion:
```
outputs/
├── activations/
│   ├── acts_step_0.pt
│   ├── acts_step_50000.pt
│   └── acts_step_143000.pt
├── saes/
│   ├── step_0/
│   │   └── final.pt
│   ├── step_50000/
│   │   └── final.pt
│   └── step_143000/
│       └── final.pt
└── figures/
    └── sae_metrics_over_training.png
```

---

## Tips

1. **Use tmux or screen** so it keeps running if you disconnect:
```bash
   tmux new -s experiment
   ./run_experiment.sh
   # Ctrl+B then D to detach
   # tmux attach -t experiment to reattach
```

2. **Monitor GPU memory**:
```bash
   watch -n 1 nvidia-smi
```

3. **Estimate remaining time**:
   - Each activation collection: 30 min
   - Each SAE training: 3 hours
   - Multiply by remaining checkpoints

---

## Next Steps (After This Works)

Once you have trained SAEs for all checkpoints:

1. **Feature tracking** - Match features across checkpoints
2. **Feature interpretation** - Label features with Claude
3. **Evolution analysis** - Identify patterns
4. **Visualization** - Create publication figures
5. **Write-up** - Complete the 20-hour deliverable

We'll build those next!
