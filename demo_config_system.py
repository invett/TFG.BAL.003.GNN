#!/usr/bin/env python
"""
Quick demonstration of the configuration system.
This script shows how to use the configuration system without running full training.
"""

import sys
import os

# Add src to path
sys.path.insert(0, 'src')

print("="*80)
print("CONFIGURATION SYSTEM DEMONSTRATION")
print("="*80)
print()

print("1. Testing YAML configuration loading...")
print("-" * 80)

from JAAD_refactored import Config

# Load default config
config = Config.from_yaml('configs/default.yaml')
print("✓ Loaded default.yaml")
print(f"  Model: {config.model.name}")
print(f"  Dataset: {config.dataset.name}")
print(f"  Batch size: {config.training.batch_size}")
print(f"  Max epochs: {config.training.max_epochs}")
print()

# Load experiment configs
config_gcn = Config.from_yaml('configs/experiments/gcn_spatial.yaml')
print("✓ Loaded gcn_spatial.yaml")
print(f"  Model: {config_gcn.model.name}")
print(f"  Temporal type: {config_gcn.graph.temporal_type}")
print(f"  Tags: {config_gcn.wandb.tags}")
print()

config_gat = Config.from_yaml('configs/experiments/gat_temporal.yaml')
print("✓ Loaded gat_temporal.yaml")
print(f"  Model: {config_gat.model.name}")
print(f"  Window size: {config_gat.graph.window_size}")
print(f"  Hidden channels: {config_gat.model.c_hidden}")
print()

config_mp = Config.from_yaml('configs/experiments/multi_pedestrian.yaml')
print("✓ Loaded multi_pedestrian.yaml")
print(f"  Multi-pedestrian: {config_mp.graph.multi_pedestrian}")
print(f"  Temporal type: {config_mp.graph.temporal_type}")
print(f"  Dataset: {config_mp.dataset.name}")
print()

print("2. Testing configuration conversion...")
print("-" * 80)

# Convert to dict
config_dict = config.to_dict()
print(f"✓ Config converted to dict with {len(config_dict)} sections")
print(f"  Sections: {list(config_dict.keys())}")
print()

# Save to temp file
import tempfile
with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    temp_path = f.name

config.save(temp_path)
print(f"✓ Config saved to: {temp_path}")

# Load it back
config_loaded = Config.from_yaml(temp_path)
print(f"✓ Config loaded from saved file")
print(f"  Model name matches: {config_loaded.model.name == config.model.name}")
print(f"  Batch size matches: {config_loaded.training.batch_size == config.training.batch_size}")

# Cleanup
os.remove(temp_path)
print(f"✓ Temp file cleaned up")
print()

print("3. Testing dataset path resolution...")
print("-" * 80)

datasets = ['JAAD_14K', 'JAAD_16K', 'JAAD_18K', 'LING_14K']
for dataset_name in datasets:
    config.dataset.name = dataset_name
    train_path = config.dataset.get_train_path()
    test_path = config.dataset.get_test_path()
    print(f"✓ {dataset_name}:")
    print(f"    Train: {train_path}")
    print(f"    Test:  {test_path}")
print()

print("4. Testing argument parser structure...")
print("-" * 80)

from JAAD_refactored import parse_args
import argparse

# Create test parser (mimics parse_args but with test arguments)
test_args = [
    '--config', 'configs/experiments/gat_temporal.yaml',
    '--model.name', 'GAT',
    '--training.batch_size', '64',
    '--training.max_epochs', '100',
    '--graph.multi_pedestrian', 'true'
]

print(f"Test arguments: {test_args}")
print()

# Parse them (would normally happen in main)
sys.argv = ['test'] + test_args
args = parse_args()

print("✓ Arguments parsed successfully")
print(f"  Config file: {args.config}")
print(f"  Model name: {getattr(args, 'model.name', None)}")
print(f"  Batch size: {getattr(args, 'training.batch_size', None)}")
print(f"  Max epochs: {getattr(args, 'training.max_epochs', None)}")
print(f"  Multi-pedestrian: {getattr(args, 'graph.multi_pedestrian', None)}")
print()

print("5. Testing configuration override...")
print("-" * 80)

from JAAD_refactored import update_config_from_args

# Load base config
base_config = Config.from_yaml('configs/experiments/gcn_spatial.yaml')
print(f"Before overrides:")
print(f"  Model: {base_config.model.name}")
print(f"  Batch size: {base_config.training.batch_size}")
print(f"  Max epochs: {base_config.training.max_epochs}")
print()

# Apply overrides
updated_config = update_config_from_args(base_config, args)
print(f"After overrides:")
print(f"  Model: {updated_config.model.name}")
print(f"  Batch size: {updated_config.training.batch_size}")
print(f"  Max epochs: {updated_config.training.max_epochs}")
print(f"  Multi-pedestrian: {updated_config.graph.multi_pedestrian}")
print()

print("="*80)
print("✓ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
print("="*80)
print()
print("The configuration system is working correctly.")
print()
print("You can now:")
print("  1. Run experiments with: python src/JAAD_refactored.py")
print("  2. Override parameters: python src/JAAD_refactored.py --model.name GAT")
print("  3. Use custom configs: python src/JAAD_refactored.py --config configs/experiments/gat_temporal.yaml")
print("  4. Run batch experiments: bash scripts/run_experiments.sh")
print()
