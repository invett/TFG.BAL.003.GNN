#!/usr/bin/env python
"""
Test script for configuration system validation.
This script tests the configuration loading, CLI parsing, and file operations.
"""

import yaml
import sys
import os

def test_yaml_files():
    """Test that all YAML files are valid."""
    print("Testing YAML files...")
    
    configs_to_test = [
        'configs/default.yaml',
        'configs/experiments/gcn_spatial.yaml',
        'configs/experiments/gat_temporal.yaml',
        'configs/experiments/multi_pedestrian.yaml'
    ]
    
    for config_path in configs_to_test:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['graph', 'model', 'dataset', 'training', 'wandb', 'experiment']
        for section in required_sections:
            assert section in config, f"Missing section '{section}' in {config_path}"
        
        print(f"  ✓ {config_path}")
    
    print("  All YAML files are valid!\n")


def test_experiment_configs():
    """Test that experiment configs have appropriate settings."""
    print("Testing experiment configurations...")
    
    # Test GCN spatial
    with open('configs/experiments/gcn_spatial.yaml', 'r') as f:
        gcn_config = yaml.safe_load(f)
    assert gcn_config['model']['name'] == 'GCN', "GCN config should use GCN model"
    assert gcn_config['graph']['temporal_type'] == 0, "GCN spatial should have temporal_type=0"
    assert 'spatial-only' in gcn_config['wandb']['tags'], "GCN config should have 'spatial-only' tag"
    print("  ✓ GCN spatial config is correct")
    
    # Test GAT temporal
    with open('configs/experiments/gat_temporal.yaml', 'r') as f:
        gat_config = yaml.safe_load(f)
    assert gat_config['model']['name'] == 'GAT', "GAT config should use GAT model"
    assert gat_config['graph']['temporal_type'] == 3, "GAT temporal should have temporal_type=3"
    assert gat_config['graph']['window_size'] == 5, "GAT temporal should have window_size=5"
    assert 'temporal' in gat_config['wandb']['tags'], "GAT config should have 'temporal' tag"
    print("  ✓ GAT temporal config is correct")
    
    # Test multi-pedestrian
    with open('configs/experiments/multi_pedestrian.yaml', 'r') as f:
        mp_config = yaml.safe_load(f)
    assert mp_config['graph']['multi_pedestrian'] == True, "Multi-pedestrian config should have multi_pedestrian=True"
    assert mp_config['graph']['temporal_type'] == 4, "Multi-pedestrian should have temporal_type=4"
    assert mp_config['dataset']['name'] == 'JAAD_16K', "Multi-pedestrian should use JAAD_16K"
    assert 'multi-pedestrian' in mp_config['wandb']['tags'], "Multi-pedestrian config should have 'multi-pedestrian' tag"
    print("  ✓ Multi-pedestrian config is correct")
    
    print("  All experiment configurations are correct!\n")


def test_default_config():
    """Test that default config has all required parameters."""
    print("Testing default configuration...")
    
    with open('configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Graph section
    assert 'type' in config['graph']
    assert 'temporal_type' in config['graph']
    assert 'multi_pedestrian' in config['graph']
    assert 'window_size' in config['graph']
    assert 'window_step' in config['graph']
    
    # Model section
    assert 'name' in config['model']
    assert 'c_in' in config['model']
    assert 'c_hidden' in config['model']
    assert 'c_out' in config['model']
    assert 'dp_rate_linear' in config['model']
    assert 'dp_rate' in config['model']
    assert 'num_layers' in config['model']
    
    # Dataset section
    assert 'name' in config['dataset']
    assert 'folder' in config['dataset']
    assert 'paths' in config['dataset']
    assert 'JAAD_14K' in config['dataset']['paths']
    assert 'JAAD_16K' in config['dataset']['paths']
    assert 'JAAD_18K' in config['dataset']['paths']
    assert 'LING_14K' in config['dataset']['paths']
    
    # Training section
    assert 'batch_size' in config['training']
    assert 'max_epochs' in config['training']
    assert 'checkpoint_path' in config['training']
    assert 'learning_rate' in config['training']
    assert 'seed' in config['training']
    
    # WandB section
    assert 'enabled' in config['wandb']
    assert 'project' in config['wandb']
    assert 'tags' in config['wandb']
    
    # Experiment section
    assert 'multi_seed' in config['experiment']
    assert 'seeds' in config['experiment']
    assert 'clean_processed' in config['experiment']
    assert len(config['experiment']['seeds']) == 30, "Should have 30 seeds"
    
    print("  ✓ Default config has all required parameters")
    print("  All default configuration checks passed!\n")


def test_file_structure():
    """Test that all required files and directories exist."""
    print("Testing file structure...")
    
    required_files = [
        'configs/default.yaml',
        'configs/experiments/gcn_spatial.yaml',
        'configs/experiments/gat_temporal.yaml',
        'configs/experiments/multi_pedestrian.yaml',
        'src/JAAD_refactored.py',
        'src/JAAD.py',  # Original should be unchanged
        'scripts/run_experiments.sh',
        'docs/CONFIG_GUIDE.md'
    ]
    
    required_dirs = [
        'configs',
        'configs/experiments',
        'configs/runs',
        'scripts',
        'docs'
    ]
    
    for file_path in required_files:
        assert os.path.exists(file_path), f"Missing required file: {file_path}"
        print(f"  ✓ {file_path}")
    
    for dir_path in required_dirs:
        assert os.path.isdir(dir_path), f"Missing required directory: {dir_path}"
        print(f"  ✓ {dir_path}/")
    
    print("  All required files and directories exist!\n")


def test_script_executable():
    """Test that the bash script is executable."""
    print("Testing script permissions...")
    
    script_path = 'scripts/run_experiments.sh'
    assert os.access(script_path, os.X_OK), f"{script_path} is not executable"
    
    print(f"  ✓ {script_path} is executable\n")


def main():
    """Run all tests."""
    print("="*80)
    print("CONFIGURATION SYSTEM VALIDATION")
    print("="*80)
    print()
    
    try:
        test_file_structure()
        test_yaml_files()
        test_default_config()
        test_experiment_configs()
        test_script_executable()
        
        print("="*80)
        print("✓ ALL TESTS PASSED!")
        print("="*80)
        print()
        print("The configuration system is ready to use.")
        print()
        print("Next steps:")
        print("  1. Run a test experiment: python src/JAAD_refactored.py --training.max_epochs 1 --wandb.enabled false")
        print("  2. Check the documentation: cat docs/CONFIG_GUIDE.md")
        print("  3. Run experiments: bash scripts/run_experiments.sh")
        print()
        return 0
        
    except AssertionError as e:
        print()
        print("="*80)
        print("✗ TEST FAILED!")
        print("="*80)
        print(f"Error: {e}")
        print()
        return 1
    except Exception as e:
        print()
        print("="*80)
        print("✗ UNEXPECTED ERROR!")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
