# Configuration System Implementation - Complete

## Overview

This implementation adds a flexible configuration system to the JAAD GNN project, enabling researchers to:
- Manage experiment parameters through YAML files
- Override parameters via CLI without code changes
- Run reproducible experiments with saved configurations
- Execute multi-seed experiments with automatic statistics
- Organize and track experiments efficiently

## What Was Implemented

### 1. Configuration Files (YAML)

**configs/default.yaml**
- Complete default configuration covering all parameters
- Comprehensive comments explaining each option
- Sections: graph, model, dataset, training, wandb, experiment

**configs/experiments/**
- `gcn_spatial.yaml`: GCN model with spatial graphs only (baseline)
- `gat_temporal.yaml`: GAT model with temporal sliding windows
- `multi_pedestrian.yaml`: Multi-pedestrian scene-level prediction

### 2. Refactored Main Script

**src/JAAD_refactored.py**
- Configuration dataclasses with type hints for all settings
- CLI argument parser supporting dot notation (e.g., `--model.name GAT`)
- All original functionality preserved:
  - Graph construction: `build_spatial_edges()`, `build_temporal_edges_between_frames()`
  - Model classes: `GNNModel`, `GraphGNNModel`, `GraphLevelGNN`
  - Dataset: Complete `JAAD` class with all temporal types (0-4)
- Helper functions:
  - `clean_processed_data()`: Remove cached data
  - `create_wandb_logger()`: Create informative WandB loggers
  - `train_single_run()`: Train one experiment
  - `train_multi_seed()`: Run multiple seeds with statistics
- Auto-saves configuration to `configs/runs/run_YYYYMMDD_HHMMSS.yaml`

### 3. Experiment Automation

**scripts/run_experiments.sh**
- Executable bash script for batch experiments
- 6 experiment categories:
  1. GCN Spatial Baseline
  2. GAT Temporal
  3. GraphConv Multi-Pedestrian
  4. Dataset Comparison (JAAD_14K, 16K, 18K)
  5. Hyperparameter Sweep (hidden channels and layers)
  6. Multi-Seed Experiment (for best config)

### 4. Documentation

**docs/CONFIG_GUIDE.md**
- Complete user guide with examples
- Configuration parameter reference
- CLI override examples
- Custom configuration creation guide
- Reproducibility tips

### 5. Testing

**tests/test_config_system.py**
- Comprehensive validation tests
- All tests passing

## Quick Start Examples

### 1. Run with Default Configuration
```bash
python src/JAAD_refactored.py
```

### 2. Run Pre-configured Experiment
```bash
python src/JAAD_refactored.py --config configs/experiments/gcn_spatial.yaml
```

### 3. Override Parameters
```bash
python src/JAAD_refactored.py --model.name GAT --training.batch_size 64 --training.max_epochs 100
```

### 4. Combine Config File with Overrides
```bash
python src/JAAD_refactored.py \
  --config configs/experiments/gat_temporal.yaml \
  --training.max_epochs 200
```

### 5. Run Multi-Seed Experiment
```bash
python src/JAAD_refactored.py \
  --config configs/experiments/gat_temporal.yaml \
  --experiment.multi_seed true
```

### 6. Run All Experiments
```bash
bash scripts/run_experiments.sh
```

## Key Features

✅ **Backward Compatible**: Original `src/JAAD.py` unchanged and functional
✅ **Type Safe**: Python dataclasses with type hints
✅ **Validated**: CodeQL scan passed with 0 vulnerabilities
✅ **Well Documented**: Comprehensive guide with examples
✅ **Tested**: All configuration tests passing
✅ **Reproducible**: Every run saves its exact configuration
✅ **Flexible**: YAML files + CLI overrides for maximum flexibility

## File Structure

```
TFG.BAL.003.GNN/
├── configs/
│   ├── default.yaml                    # Default configuration
│   ├── experiments/                    # Pre-configured experiments
│   │   ├── gcn_spatial.yaml
│   │   ├── gat_temporal.yaml
│   │   └── multi_pedestrian.yaml
│   └── runs/                           # Auto-saved run configs
│       └── run_YYYYMMDD_HHMMSS.yaml   # (generated at runtime)
├── docs/
│   └── CONFIG_GUIDE.md                 # User documentation
├── scripts/
│   └── run_experiments.sh              # Batch experiment runner
├── src/
│   ├── JAAD.py                         # Original (unchanged)
│   └── JAAD_refactored.py             # New with config system
└── tests/
    └── test_config_system.py           # Configuration tests
```

## Configuration Sections

### Graph Configuration
- `type`: Spatial graph structure (0-2)
- `temporal_type`: Temporal dependencies (0-4)
- `multi_pedestrian`: Scene-level vs pedestrian-level
- `window_size`, `window_step`: Sliding window parameters

### Model Configuration
- `name`: GCN, GAT, or GraphConv
- `c_hidden`: Hidden layer dimension
- `num_layers`: Number of GNN layers
- `dp_rate`, `dp_rate_linear`: Dropout rates

### Dataset Configuration
- `name`: JAAD_14K, JAAD_16K, JAAD_18K, LING_14K
- Automatic path resolution

### Training Configuration
- `batch_size`, `max_epochs`: Training parameters
- `learning_rate`: AdamW optimizer learning rate
- `seed`: Random seed for reproducibility
- Early stopping and learning rate scheduling

### WandB Configuration
- `enabled`: Enable/disable logging
- `project`, `entity`: WandB workspace
- `tags`: Experiment organization

### Experiment Configuration
- `multi_seed`: Run 30 seeds automatically
- `clean_processed`: Force data reprocessing

## Common CLI Patterns

```bash
# Change model architecture
--model.name GAT --model.c_hidden 256 --model.num_layers 3

# Adjust training
--training.batch_size 64 --training.max_epochs 100 --training.learning_rate 0.0001

# Select dataset
--dataset.name JAAD_16K

# Graph configuration
--graph.temporal_type 3 --graph.window_size 5

# Disable WandB for testing
--wandb.enabled false

# Multi-seed evaluation
--experiment.multi_seed true
```

## Next Steps for Users

1. **Read the documentation**: `cat docs/CONFIG_GUIDE.md`
2. **Test the system**: Run a quick test
   ```bash
   python src/JAAD_refactored.py --training.max_epochs 1 --wandb.enabled false
   ```
3. **Run an experiment**: Try a pre-configured setup
   ```bash
   python src/JAAD_refactored.py --config configs/experiments/gcn_spatial.yaml
   ```
4. **Create custom configs**: Copy and modify example configs
5. **Run batch experiments**: Use the provided script
   ```bash
   bash scripts/run_experiments.sh
   ```

## Notes

- **Original JAAD.py**: Remains unchanged for backward compatibility
- **Data Processing**: First run will process data (can take time)
- **GPU Required**: For reasonable training times
- **WandB Account**: Optional but recommended for experiment tracking
- **Seed Count**: Default is 30 seeds for multi-seed experiments

## Validation

All validation tests pass:
```bash
python tests/test_config_system.py
```

Output:
```
✓ All YAML files are valid
✓ All required parameters present
✓ Experiment configurations correct
✓ File structure complete
✓ Scripts executable
✓ Original JAAD.py unchanged
```

## Support

For questions or issues:
1. Check `docs/CONFIG_GUIDE.md` for detailed examples
2. Review pre-configured experiments in `configs/experiments/`
3. Examine saved run configurations in `configs/runs/`

## Implementation Complete ✅

All requirements from the problem statement have been successfully implemented:
- ✅ Configuration infrastructure with YAML files
- ✅ Three example experiment configurations
- ✅ Refactored script with dataclasses and CLI parsing
- ✅ Experiment runner script
- ✅ Comprehensive documentation
- ✅ Backward compatibility maintained
- ✅ All tests passing
- ✅ Security scan clean (0 vulnerabilities)
