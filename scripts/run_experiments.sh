#!/bin/bash
#
# Experiment Runner Script for JAAD GNN Experiments
# This script runs multiple experiments sequentially to evaluate different configurations
#

set -e  # Exit on error

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p configs/runs
mkdir -p checkpoints
mkdir -p results

echo "========================================================================"
echo "JAAD GNN Experiment Runner"
echo "========================================================================"
echo ""

# Function to run a single experiment
run_experiment() {
    local name=$1
    local config=$2
    shift 2
    local extra_args=("$@")
    
    echo "------------------------------------------------------------------------"
    echo "Running: $name"
    echo "Config: $config"
    if [ ${#extra_args[@]} -gt 0 ]; then
        echo "Extra args: ${extra_args[*]}"
    fi
    echo "------------------------------------------------------------------------"
    
    python src/JAAD_refactored.py --config "$config" "${extra_args[@]}"
    
    echo ""
    echo "Completed: $name"
    echo ""
}

# ============================================================================
# EXPERIMENT CATEGORY 1: GCN Spatial Baseline
# ============================================================================
echo ""
echo "========================================================================"
echo "CATEGORY 1: GCN Spatial Baseline"
echo "========================================================================"
echo ""

run_experiment \
    "GCN Spatial Baseline" \
    "configs/experiments/gcn_spatial.yaml"

# ============================================================================
# EXPERIMENT CATEGORY 2: GAT Temporal
# ============================================================================
echo ""
echo "========================================================================"
echo "CATEGORY 2: GAT Temporal"
echo "========================================================================"
echo ""

run_experiment \
    "GAT Temporal" \
    "configs/experiments/gat_temporal.yaml"

# ============================================================================
# EXPERIMENT CATEGORY 3: GraphConv Multi-Pedestrian
# ============================================================================
echo ""
echo "========================================================================"
echo "CATEGORY 3: GraphConv Multi-Pedestrian"
echo "========================================================================"
echo ""

run_experiment \
    "GraphConv Multi-Pedestrian" \
    "configs/experiments/multi_pedestrian.yaml"

# ============================================================================
# EXPERIMENT CATEGORY 4: Dataset Comparison
# ============================================================================
echo ""
echo "========================================================================"
echo "CATEGORY 4: Dataset Comparison"
echo "========================================================================"
echo ""

run_experiment \
    "Dataset: JAAD_14K" \
    "configs/default.yaml" \
    --dataset.name "JAAD_14K"

run_experiment \
    "Dataset: JAAD_16K" \
    "configs/default.yaml" \
    --dataset.name "JAAD_16K"

run_experiment \
    "Dataset: JAAD_18K" \
    "configs/default.yaml" \
    --dataset.name "JAAD_18K"

# ============================================================================
# EXPERIMENT CATEGORY 5: Hyperparameter Sweep
# ============================================================================
echo ""
echo "========================================================================"
echo "CATEGORY 5: Hyperparameter Sweep"
echo "========================================================================"
echo ""

# Hidden channels sweep
for hidden in 64 128 256; do
    run_experiment \
        "Hidden Channels: $hidden" \
        "configs/default.yaml" \
        --model.c_hidden "$hidden" \
        --wandb.tags "[\"hyperparam-sweep\", \"hidden-$hidden\"]"
done

# Number of layers sweep
for layers in 2 3 4; do
    run_experiment \
        "Layers: $layers" \
        "configs/default.yaml" \
        --model.num_layers "$layers" \
        --wandb.tags "[\"hyperparam-sweep\", \"layers-$layers\"]"
done

# ============================================================================
# EXPERIMENT CATEGORY 6: Multi-Seed Experiment (Best Config)
# ============================================================================
echo ""
echo "========================================================================"
echo "CATEGORY 6: Multi-Seed Experiment (Best Configuration)"
echo "========================================================================"
echo ""
echo "NOTE: This will run 30 experiments with different seeds."
echo "This may take a considerable amount of time."
echo ""

# Uncomment the following to run multi-seed experiment
# run_experiment \
#     "Multi-Seed Best Config" \
#     "configs/experiments/gat_temporal.yaml" \
#     --experiment.multi_seed true

echo ""
echo "========================================================================"
echo "ALL EXPERIMENTS COMPLETED!"
echo "========================================================================"
echo ""
echo "Results have been logged to WandB (if enabled)"
echo "Configurations saved in: configs/runs/"
echo "Checkpoints saved in: checkpoints/"
echo ""
