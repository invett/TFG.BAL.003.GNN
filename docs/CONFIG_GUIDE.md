# Configuration System Guide

This guide explains how to use the flexible configuration system for JAAD GNN experiments.

## Table of Contents

- [Quick Start](#quick-start)
- [Configuration Structure](#configuration-structure)
- [CLI Overrides](#cli-overrides)
- [Creating Custom Configurations](#creating-custom-configurations)
- [Experiment Organization](#experiment-organization)
- [Reproducibility](#reproducibility)

---

## Quick Start

### Example 1: Use Default Configuration

Run an experiment with the default settings:

```bash
python src/JAAD_refactored.py
```

This loads `configs/default.yaml` and runs a single experiment.

### Example 2: Use a Pre-configured Experiment

Run one of the pre-configured experiments:

```bash
# GCN with spatial graph only
python src/JAAD_refactored.py --config configs/experiments/gcn_spatial.yaml

# GAT with temporal dependencies
python src/JAAD_refactored.py --config configs/experiments/gat_temporal.yaml

# Multi-pedestrian scene-level prediction
python src/JAAD_refactored.py --config configs/experiments/multi_pedestrian.yaml
```

### Example 3: Override Specific Parameters

Override parameters from the command line:

```bash
# Change model and batch size
python src/JAAD_refactored.py --model.name GAT --training.batch_size 64

# Change hidden channels and number of layers
python src/JAAD_refactored.py --model.c_hidden 256 --model.num_layers 3
```

### Example 4: Combine Config File with Overrides

Load a configuration file and override specific parameters:

```bash
# Use GAT temporal config but train for more epochs
python src/JAAD_refactored.py \
  --config configs/experiments/gat_temporal.yaml \
  --training.max_epochs 200

# Use multi-pedestrian config with different dataset
python src/JAAD_refactored.py \
  --config configs/experiments/multi_pedestrian.yaml \
  --dataset.name JAAD_18K
```

---

## Configuration Structure

The configuration system uses YAML files organized into six main sections:

### 1. Graph Configuration

Controls how graphs are constructed from the data.

```yaml
graph:
  # Spatial graph type
  # 0 = Angie's graph structure
  # 1 = Adrián's graph structure
  # 2 = Combination of both (recommended)
  type: 2
  
  # Temporal dependencies
  # 0 = No temporal connections (spatial only)
  # 1 = Connection between same pedestrian across frames
  # 2 = Complete connection between all nodes across frames
  # 3 = Sliding windows (aggregates temporal information)
  # 4 = Multi-pedestrian scene-level temporal windows
  temporal_type: 3
  
  # Multi-pedestrian mode
  # true  = Create one graph per scene (includes all pedestrians)
  # false = Create one graph per pedestrian/temporal window
  multi_pedestrian: false
  
  # Window parameters (used when temporal_type=3)
  window_size: 3   # Number of frames in each window
  window_step: 1   # Step size for sliding window
```

### 2. Model Configuration

Defines the GNN model architecture.

```yaml
model:
  # Model type: GCN, GAT, or GraphConv
  name: "GCN"
  
  # Input features per node (don't change unless data changes)
  c_in: 24
  
  # Hidden layer dimension (try 64, 128, 256)
  c_hidden: 128
  
  # Output dimension (1 for binary classification)
  c_out: 1
  
  # Dropout for final linear layers (0.0-0.9)
  dp_rate_linear: 0.5
  
  # Dropout for GNN layers (0.0-0.5)
  dp_rate: 0.2
  
  # Number of GNN layers (2-4 typically)
  num_layers: 2
```

### 3. Dataset Configuration

Specifies which dataset to use.

```yaml
dataset:
  # Dataset name (JAAD_14K, JAAD_16K, JAAD_18K, LING_14K)
  name: "JAAD_14K"
  
  # Root folder for datasets
  folder: "datasets"
  
  # All dataset paths are defined in default.yaml
  # You typically only need to change 'name'
```

Available datasets:
- **JAAD_14K**: 14,000 training samples
- **JAAD_16K**: 16,000 training samples
- **JAAD_18K**: 18,000 training samples
- **LING_14K**: Linguistic JAAD dataset

### 4. Training Configuration

Controls the training process.

```yaml
training:
  # Batch size (adjust based on GPU memory)
  batch_size: 100
  
  # Maximum training epochs
  max_epochs: 50
  
  # Where to save model checkpoints
  checkpoint_path: "./checkpoints"
  
  # Learning rate for AdamW optimizer
  learning_rate: 0.001
  
  # Patience for ReduceLROnPlateau scheduler
  lr_scheduler_patience: 1
  
  # Patience for early stopping
  early_stopping_patience: 3
  
  # Train/validation split ratio
  train_val_split: 0.8
  
  # Number of data loading workers
  num_workers: 4
  
  # Random seed for reproducibility
  seed: 42
```

### 5. Weights & Biases Configuration

Controls experiment logging to WandB.

```yaml
wandb:
  # Enable or disable WandB logging
  enabled: true
  
  # WandB project name
  project: "tfg"
  
  # WandB entity (username/organization)
  entity: null  # Uses default
  
  # Log model checkpoints to WandB
  log_model: true
  
  # Tags for organizing experiments
  tags: ["baseline", "gcn", "spatial-only"]
```

### 6. Experiment Configuration

Controls experiment execution settings.

```yaml
experiment:
  # Run multiple seeds for statistical analysis
  multi_seed: false
  
  # List of seeds (used when multi_seed=true)
  seeds: [0, 1, 2, ..., 29]  # 30 seeds by default
  
  # Clean processed data before training
  clean_processed: true
```

---

## CLI Overrides

All configuration parameters can be overridden from the command line using dot notation.

### Common CLI Override Examples

#### Model Architecture

```bash
# Change model type
python src/JAAD_refactored.py --model.name GAT

# Change hidden channels
python src/JAAD_refactored.py --model.c_hidden 256

# Change number of layers
python src/JAAD_refactored.py --model.num_layers 3

# Change dropout rates
python src/JAAD_refactored.py --model.dp_rate 0.3 --model.dp_rate_linear 0.6
```

#### Graph Construction

```bash
# Change graph type
python src/JAAD_refactored.py --graph.type 0

# Change temporal type
python src/JAAD_refactored.py --graph.temporal_type 4

# Enable multi-pedestrian mode
python src/JAAD_refactored.py --graph.multi_pedestrian true

# Change window parameters
python src/JAAD_refactored.py --graph.window_size 5 --graph.window_step 2
```

#### Training Parameters

```bash
# Change batch size and epochs
python src/JAAD_refactored.py --training.batch_size 64 --training.max_epochs 100

# Change learning rate
python src/JAAD_refactored.py --training.learning_rate 0.0001

# Change random seed
python src/JAAD_refactored.py --training.seed 123

# Adjust number of workers
python src/JAAD_refactored.py --training.num_workers 8
```

#### Dataset Selection

```bash
# Change dataset
python src/JAAD_refactored.py --dataset.name JAAD_16K
```

#### WandB Settings

```bash
# Disable WandB logging
python src/JAAD_refactored.py --wandb.enabled false

# Change WandB project
python src/JAAD_refactored.py --wandb.project my-project

# Set WandB entity
python src/JAAD_refactored.py --wandb.entity my-username
```

#### Experiment Control

```bash
# Run multi-seed experiment
python src/JAAD_refactored.py --experiment.multi_seed true

# Don't clean processed data (faster if data hasn't changed)
python src/JAAD_refactored.py --experiment.clean_processed false
```

### Complex CLI Examples

Combine multiple overrides:

```bash
# Full hyperparameter sweep experiment
python src/JAAD_refactored.py \
  --model.name GAT \
  --model.c_hidden 256 \
  --model.num_layers 4 \
  --training.batch_size 64 \
  --training.max_epochs 100 \
  --training.learning_rate 0.0005 \
  --dataset.name JAAD_16K

# Quick test run with reduced settings
python src/JAAD_refactored.py \
  --training.max_epochs 5 \
  --training.batch_size 32 \
  --wandb.enabled false
```

---

## Creating Custom Configurations

### Step 1: Start with a Template

Copy an existing configuration file:

```bash
cp configs/experiments/gcn_spatial.yaml configs/experiments/my_experiment.yaml
```

### Step 2: Edit the Configuration

Open the file and modify the parameters:

```yaml
# My Custom Experiment Configuration

graph:
  type: 2
  temporal_type: 3
  multi_pedestrian: false
  window_size: 7  # Increased window size
  window_step: 1

model:
  name: "GAT"
  c_in: 24
  c_hidden: 512  # Larger model
  c_out: 1
  dp_rate_linear: 0.5
  dp_rate: 0.2
  num_layers: 4  # Deeper network

dataset:
  name: "JAAD_18K"  # Larger dataset

training:
  batch_size: 32  # Smaller batch for larger model
  max_epochs: 150  # More training
  learning_rate: 0.0005  # Lower learning rate
  seed: 42

wandb:
  enabled: true
  tags: ["custom", "large-model", "long-training"]

experiment:
  multi_seed: false
  clean_processed: true
```

### Step 3: Run Your Configuration

```bash
python src/JAAD_refactored.py --config configs/experiments/my_experiment.yaml
```

### Best Practices for Custom Configs

1. **Use descriptive file names**: `gat_large_jaad18k.yaml` is better than `config1.yaml`
2. **Add comments**: Explain why you chose specific parameters
3. **Use meaningful tags**: Makes it easier to find experiments in WandB
4. **Version your configs**: Keep them in git for reproducibility
5. **Test incrementally**: Start with a short run before committing to long training

---

## Experiment Organization

### Directory Structure

```
configs/
├── default.yaml              # Default configuration
├── experiments/              # Pre-configured experiments
│   ├── gcn_spatial.yaml
│   ├── gat_temporal.yaml
│   └── multi_pedestrian.yaml
└── runs/                     # Auto-saved run configurations
    ├── run_20260219_103045.yaml
    └── run_20260219_104512.yaml

checkpoints/                  # Model checkpoints
└── GraphLevelJAAD/

results/                      # (Optional) Store results here
```

### Running Batch Experiments

Use the provided script to run multiple experiments:

```bash
# Run all pre-configured experiments
bash scripts/run_experiments.sh
```

The script includes:
1. GCN Spatial Baseline
2. GAT Temporal
3. GraphConv Multi-Pedestrian
4. Dataset Comparison (JAAD_14K, 16K, 18K)
5. Hyperparameter Sweep (hidden channels and layers)
6. Multi-seed Experiment (optional, commented out)

### Monitoring Experiments

- **WandB Dashboard**: View real-time training metrics
- **Checkpoints**: Best models are saved in `checkpoints/`
- **Run Configs**: Every run is saved to `configs/runs/` with timestamp

---

## Reproducibility

### Ensuring Reproducible Results

The configuration system ensures reproducibility by:

1. **Saving every configuration**: Each run's exact config is saved to `configs/runs/`
2. **Fixed random seeds**: Control randomness with the `seed` parameter
3. **WandB logging**: Complete training history and hyperparameters
4. **Version control**: Commit your config files to git

### Reproducing an Experiment

To reproduce an experiment from a saved configuration:

```bash
# Find the configuration from a previous run
ls configs/runs/

# Run with the exact same configuration
python src/JAAD_refactored.py --config configs/runs/run_20260219_103045.yaml
```

### Multi-Seed Experiments

For statistically significant results, run multiple seeds:

```bash
# Run 30 experiments with different seeds
python src/JAAD_refactored.py \
  --config configs/experiments/gat_temporal.yaml \
  --experiment.multi_seed true
```

This will:
- Run 30 experiments with seeds 0-29
- Print statistics (mean, std, min, max, range)
- Log each run to WandB with appropriate naming

### Tips for Reproducibility

1. **Always set a seed**: Even for exploration
2. **Save successful configs**: Copy them to `configs/experiments/`
3. **Document changes**: Use WandB tags and notes
4. **Version datasets**: Track which dataset version was used
5. **Record environment**: Note PyTorch/CUDA versions if sharing results

---

## Common Use Cases

### Use Case 1: Quick Experimentation

```bash
# Try different models quickly
python src/JAAD_refactored.py --model.name GCN
python src/JAAD_refactored.py --model.name GAT
python src/JAAD_refactored.py --model.name GraphConv
```

### Use Case 2: Hyperparameter Tuning

```bash
# Grid search over hidden dimensions
for hidden in 64 128 256 512; do
  python src/JAAD_refactored.py \
    --model.c_hidden $hidden \
    --wandb.tags "[\"grid-search\", \"hidden-$hidden\"]"
done
```

### Use Case 3: Dataset Comparison

```bash
# Compare all datasets with same model
for dataset in JAAD_14K JAAD_16K JAAD_18K; do
  python src/JAAD_refactored.py \
    --dataset.name $dataset \
    --wandb.tags "[\"dataset-comparison\", \"$dataset\"]"
done
```

### Use Case 4: Final Evaluation

```bash
# Train best model with multiple seeds
python src/JAAD_refactored.py \
  --config configs/experiments/my_best_config.yaml \
  --experiment.multi_seed true
```

---

## Troubleshooting

### Common Issues

**Issue**: "FileNotFoundError: configs/default.yaml"
- **Solution**: Make sure you're running from the repository root

**Issue**: Parameters not being overridden
- **Solution**: Check the syntax: `--section.parameter value` (with dot notation)

**Issue**: Boolean parameters not working
- **Solution**: Use lowercase: `--wandb.enabled true` (not `True`)

**Issue**: Out of memory during training
- **Solution**: Reduce `--training.batch_size` or `--model.c_hidden`

**Issue**: Processed data not updating
- **Solution**: Set `--experiment.clean_processed true` to force reprocessing

### Getting Help

- Check the examples in this guide
- Look at pre-configured experiments in `configs/experiments/`
- Review saved configurations in `configs/runs/`
- Check WandB logs for parameter values

---

## Summary

The configuration system provides:

✅ **Flexibility**: Change any parameter without editing code  
✅ **Reproducibility**: Every run is saved with exact configuration  
✅ **Organization**: Group related experiments with YAML files  
✅ **Convenience**: Override specific parameters from CLI  
✅ **Scalability**: Run batch experiments and multi-seed evaluations  

Start experimenting! 🚀
