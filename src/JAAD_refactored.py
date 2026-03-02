"""
Refactored JAAD script with flexible configuration system.

This script provides a configuration-based approach for running GNN experiments
on the JAAD pedestrian crossing dataset. It supports:
- YAML configuration files
- CLI argument overrides
- Multi-seed experiments
- Comprehensive logging and reproducibility

The original JAAD.py functionality is preserved while adding configuration flexibility.
"""

import os
import sys
import argparse
import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as geom_nn
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm
from pytorch_lightning.loggers import WandbLogger
import wandb


# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

@dataclass
class GraphConfig:
    """Graph construction configuration."""
    type: int = 2  # 0=Angie, 1=Adrián, 2=Combination
    temporal_type: int = 3  # 0-4, see documentation
    multi_pedestrian: bool = False
    window_size: int = 3
    window_step: int = 1


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    name: str = "GCN"  # GCN, GAT, or GraphConv
    c_in: int = 24
    c_hidden: int = 128
    c_out: int = 1
    dp_rate_linear: float = 0.5
    dp_rate: float = 0.2
    num_layers: int = 2


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str = "JAAD_14K"
    folder: str = "datasets"
    paths: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "JAAD_14K": {
            "train": "datasets/JAAD_14K_TRAIN.csv",
            "test": "datasets/TEST_JAAD_ALL.csv"
        },
        "JAAD_16K": {
            "train": "datasets/JAAD_16K_TRAIN.csv",
            "test": "datasets/TEST_JAAD_ALL.csv"
        },
        "JAAD_18K": {
            "train": "datasets/JAAD_18K_TRAIN.csv",
            "test": "datasets/TEST_JAAD_ALL.csv"
        },
        "LING_14K": {
            "train": "datasets/LINGUISTIC_JAAD_TRAIN_14K.csv",
            "test": "datasets/LINGUISTIC_TEST_JAAD_ALL.csv"
        }
    })

    def get_train_path(self) -> str:
        """Get training dataset path."""
        return self.paths[self.name]["train"]

    def get_test_path(self) -> str:
        """Get test dataset path."""
        return self.paths[self.name]["test"]


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 100
    max_epochs: int = 50
    checkpoint_path: str = "./checkpoints"
    learning_rate: float = 0.001
    lr_scheduler_patience: int = 1
    early_stopping_patience: int = 3
    train_val_split: float = 0.8
    num_workers: int = 4
    seed: int = 42


@dataclass
class WandbConfig:
    """Weights & Biases configuration."""
    enabled: bool = True
    project: str = "tfg"
    entity: Optional[str] = None
    log_model: bool = True
    tags: List[str] = field(default_factory=lambda: ["baseline"])


@dataclass
class ExperimentConfig:
    """Experiment execution configuration."""
    multi_seed: bool = False
    seeds: List[int] = field(default_factory=lambda: list(range(30)))
    clean_processed: bool = True


@dataclass
class Config:
    """Main configuration class."""
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            graph=GraphConfig(**config_dict.get('graph', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            dataset=DatasetConfig(**config_dict.get('dataset', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            wandb=WandbConfig(**config_dict.get('wandb', {})),
            experiment=ExperimentConfig(**config_dict.get('experiment', {}))
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def save(self, path: str):
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


# ============================================================================
# GRAPH CONSTRUCTION FUNCTIONS (Preserved from original)
# ============================================================================

def build_spatial_edges(graph_type):
    """Build spatial edges based on graph type."""
    if graph_type == 0:
        return np.array([[0, 0, 0, 0, 0],
                         [1, 2, 3, 4, 5]], dtype=int)
    elif graph_type == 1:
        return np.array([[0, 0, 0, 0, 4],
                         [1, 2, 3, 4, 5]], dtype=int)
    elif graph_type == 2:
        return np.array([[0, 0, 0, 0, 0, 4],
                         [1, 2, 3, 4, 5, 5]], dtype=int)
    else:
        raise ValueError("graph_type inválido")


def build_temporal_edges_between_frames(temporal_type, prev_offset, curr_offset, n_feats):
    """Build temporal edges between frames."""
    if temporal_type == 0 or temporal_type == 3:
        return []
    edges = []
    if temporal_type == 1:
        edges.append(np.array([[prev_offset + 0], [curr_offset + 0]]))
    elif temporal_type == 2:
        for node in range(n_feats + 1):
            edges.append(np.array([[prev_offset + node], [curr_offset + node]]))
    return edges


# ============================================================================
# MODEL CLASSES (Preserved from original)
# ============================================================================

gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv
}


class GNNModel(nn.Module):
    """GNN model with multiple layers."""
    def __init__(self, c_in, c_hidden, c_out, num_layers=2, layer_name="GCN", dp_rate=0.1, **kwargs):
        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]
        layers = []
        in_channels = c_in

        for _ in range(num_layers - 1):
            layers += [
                gnn_layer(in_channels, c_hidden, **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden

        layers += [gnn_layer(in_channels, c_out, **kwargs)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        for l in self.layers:
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index)
            else:
                x = l(x)
        return x


class GraphGNNModel(nn.Module):
    """Graph-level GNN model with pooling."""
    def __init__(self, c_in, c_hidden, c_out, dp_rate_linear=0.5, **kwargs):
        super().__init__()
        self.GNN = GNNModel(c_in, c_hidden, c_hidden, **kwargs)
        self.head = nn.Sequential(
            nn.Dropout(dp_rate_linear),
            nn.Linear(c_hidden, c_out)
        )

    def forward(self, x, edge_index, batch_idx):
        x = self.GNN(x, edge_index)
        x = geom_nn.global_mean_pool(x, batch_idx)
        return self.head(x)


class GraphLevelGNN(pl.LightningModule):
    """PyTorch Lightning module for graph-level classification."""
    def __init__(self, learning_rate=1e-3, lr_scheduler_patience=1, **model_kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = GraphGNNModel(**model_kwargs)
        self.loss_module = nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate
        self.lr_scheduler_patience = lr_scheduler_patience

    def forward(self, data, mode="train"):
        x, edge_index, batch_idx = data.x, data.edge_index, data.batch
        x = self.model(x, edge_index, batch_idx)
        x = x.squeeze(-1)
        preds = (x > 0).float()
        loss = self.loss_module(x, data.y.float())
        acc = (preds == data.y).sum().float() / preds.shape[0]
        return loss, acc

    def training_step(self, batch, _):
        loss, acc = self.forward(batch)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss, acc = self.forward(batch)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, _):
        _, acc = self.forward(batch)
        self.log('test_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=self.lr_scheduler_patience
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


# ============================================================================
# DATASET CLASS (Preserved from original)
# ============================================================================

class JAAD(InMemoryDataset):
    """JAAD pedestrian crossing dataset."""
    
    @property
    def processed_file_names(self):
        return ['data_jaad.pt', 'data_jaad_test.pt']

    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None,
                 mode='train', config=None):
        """
        Initialize JAAD dataset.
        
        Args:
            root: Root directory for data
            transform: Data transform
            pre_transform: Pre-transform
            pre_filter: Pre-filter
            mode: 'train' or 'test'
            config: Config object containing all parameters
        """
        if config is None:
            raise ValueError("config parameter is required")
        
        self._csv_train = config.dataset.get_train_path()
        self._csv_test = config.dataset.get_test_path()
        self._graph_type = config.graph.type
        self._temporal_type = config.graph.temporal_type
        self._window_size = config.graph.window_size
        self._window_step = config.graph.window_step
        self._multi_pedestrian = config.graph.multi_pedestrian

        super().__init__(root, transform, pre_transform, pre_filter)

        self.mode = mode
        if mode == "train":
            self.load(self.processed_paths[0])
        else:
            self.load(self.processed_paths[1])

    def process(self):
        """Process raw data into graphs."""

        def create_graphs(csv_path, desc):
            """Create single-pedestrian graphs (temporal types 0-3)."""
            df = pd.read_csv(csv_path)
            df['cross'] = df['cross'].astype(str)

            n_rows = len(df)
            n_feats = 6

            # Feature encoding
            def pad_feature(arr, left, right, total=24):
                arr = np.pad(arr, [(0, 0), (left, max(0, right))])
                if arr.shape[1] < total:
                    arr = np.pad(arr, [(0, 0), (0, total - arr.shape[1])])
                return arr[:, :total]

            block_list = [
                pad_feature(np.zeros((n_rows, 1)), 0, 19),
                pad_feature(pd.get_dummies(df.get('attention', 0)).to_numpy(float), 1, 19),
                pad_feature(pd.get_dummies(df.get('orientation', 0)).to_numpy(float), 3, 17),
                pad_feature(pd.get_dummies(df.get('proximity', 0)).to_numpy(float), 7, 13),
                pad_feature(df[['distance']].to_numpy(float), 10, 9),
                pad_feature(pd.get_dummies(df.get('action', 0)).to_numpy(float), 15, 5),
                pad_feature(pd.get_dummies(df.get('zebra_cross', 0)).to_numpy(float), 18, 2),
            ]

            x = np.vstack(block_list)
            spatial_base = build_spatial_edges(self._graph_type)
            data_list = []

            # Multi-pedestrian single-frame graphs
            if self._multi_pedestrian:
                scene_groups = df.groupby(["video", "frame"])

                for (video, frame), scene_df in tqdm(scene_groups, desc=desc):
                    row_idxs = scene_df.index.to_numpy()
                    x_scene = x[row_idxs]
                    num_nodes = x_scene.shape[0]

                    if num_nodes < 2:
                        continue  # Skip scenes with 1 pedestrian

                    edges = []
                    for i in range(num_nodes):
                        for j in range(num_nodes):
                            if i != j:
                                edges.append([i, j])

                    edge_index = torch.tensor(edges, dtype=torch.long).T

                    # Label: positive if at least one pedestrian is crossing
                    label_val = scene_df['cross'].map(
                        {'not-crossing': 0, 'crossing': 1,
                         'noCrossRoad': 0, 'CrossRoad': 1}
                    ).max()

                    graph = Data(
                        x=torch.tensor(x_scene).float(),
                        edge_index=edge_index,
                        y=torch.tensor([label_val], dtype=torch.long)
                    )

                    data_list.append(graph)

                return data_list

            # Remove extra columns if they exist
            for col in ['video', 'frame', 'person']:
                if col in df.columns:
                    df = df.drop(columns=[col])

            # Sliding window graphs (temporal_type=3)
            if self._temporal_type == 3:
                for start in tqdm(range(0, n_rows - self._window_size + 1, self._window_step), desc=desc):
                    end = start + self._window_size
                    frames_window = df.iloc[start:end]

                    x_window = []
                    for f in range(n_feats + 1):
                        idxs = [i + n_rows * f for i in range(start, end)]
                        x_window.append(x[idxs].mean(axis=0))

                    x_combined = np.vstack(x_window)
                    edges_combined = spatial_base.copy()

                    label_val = frames_window['cross'].map(
                        {'not-crossing': 0, 'crossing': 1, 'noCrossRoad': 0, 'CrossRoad': 1}
                    ).mode()[0]

                    graph = Data(
                        x=torch.tensor(x_combined).float(),
                        edge_index=torch.tensor(edges_combined),
                        y=torch.tensor([label_val], dtype=torch.long)
                    )
                    data_list.append(graph)

            # Spatiotemporal graphs (temporal_type=0,1,2)
            else:
                for start in tqdm(range(0, n_rows - self._window_size + 1, self._window_step), desc=desc):
                    end = start + self._window_size
                    frames_window = df.iloc[start:end]

                    x_blocks = []
                    edges = []

                    for i_frame, frame_idx in enumerate(range(start, end)):
                        xf = np.empty((0, 24))
                        xf = np.vstack([xf, x[frame_idx]])
                        for j in range(n_feats):
                            xf = np.vstack([xf, x[n_rows * (j + 1) + frame_idx]])

                        x_blocks.append(xf)

                        offset = i_frame * (n_feats + 1)
                        edges.append(spatial_base + offset)

                        if i_frame > 0:
                            prev_o = (i_frame - 1) * (n_feats + 1)
                            curr_o = i_frame * (n_feats + 1)
                            temporal_edges = build_temporal_edges_between_frames(
                                self._temporal_type, prev_o, curr_o, n_feats
                            )
                            edges.extend(temporal_edges)

                    x_combined = np.vstack(x_blocks)
                    edges_combined = np.hstack(edges) if len(edges) > 0 else np.zeros((2, 0), int)

                    label_val = frames_window['cross'].map(
                        {'not-crossing': 0, 'crossing': 1, 'noCrossRoad': 0, 'CrossRoad': 1}
                    ).mode()[0]

                    graph = Data(
                        x=torch.tensor(x_combined).float(),
                        edge_index=torch.tensor(edges_combined),
                        y=torch.tensor([label_val], dtype=torch.long)
                    )
                    data_list.append(graph)

            return data_list

        def create_scene_temporal_graphs(csv_path, desc):
            """Create multi-pedestrian temporal graphs (temporal_type=4)."""
            df = pd.read_csv(csv_path)
            df['cross'] = df['cross'].astype(str)

            scene_groups = df.groupby(["video", "frame"])
            scenes = list(scene_groups)

            data_list = []
            n_feats = 24

            for i in tqdm(range(1, len(scenes) - 1), desc=desc):
                graphs_x = []
                graphs_edges = []

                # Initialize offsets
                offset = 0
                prev_offsets = []  # Save offset for each scene t-1, t, t+1

                # Scenes t-1, t, t+1
                for t, (_, scene_df) in enumerate(scenes[i - 1:i + 2]):
                    scene_df = scene_df.reset_index(drop=True)
                    num_nodes = len(scene_df)

                    if num_nodes < 2:
                        break  # Skip scenes with 1 pedestrian

                    # Features
                    block_list = [
                        np.zeros((num_nodes, 1)),
                        pd.get_dummies(scene_df.get('attention', 0)).to_numpy(float),
                        pd.get_dummies(scene_df.get('orientation', 0)).to_numpy(float),
                        pd.get_dummies(scene_df.get('proximity', 0)).to_numpy(float),
                        scene_df[['distance']].to_numpy(float),
                        pd.get_dummies(scene_df.get('action', 0)).to_numpy(float),
                        pd.get_dummies(scene_df.get('zebra_cross', 0)).to_numpy(float),
                    ]

                    x_scene = np.hstack(block_list)
                    x_scene = np.pad(x_scene, ((0, 0), (0, n_feats - x_scene.shape[1])))

                    graphs_x.append(x_scene)

                    # Map pedestrians to local nodes
                    person_to_node = {pid: idx for idx, pid in enumerate(scene_df['person'].values)}

                    # Spatial edges
                    spatial_edges = []
                    for u in range(num_nodes):
                        for v in range(num_nodes):
                            if u != v:
                                spatial_edges.append([offset + u, offset + v])
                    graphs_edges.append(np.array(spatial_edges).T)

                    # Temporal edges
                    if t > 0:
                        prev_scene_df = scenes[i - 1 + t - 1][1].reset_index(drop=True)
                        prev_person_to_node = {pid: idx for idx, pid in enumerate(prev_scene_df['person'].values)}
                        prev_offset = prev_offsets[t - 1]

                        for pid in scene_df['person'].unique():
                            if pid in prev_person_to_node:
                                u = prev_person_to_node[pid]
                                v = person_to_node[pid]
                                graphs_edges.append(
                                    np.array([[prev_offset + u], [offset + v]])
                                )

                    prev_offsets.append(offset)
                    offset += num_nodes

                # Only continue if we have all 3 scenes
                if len(graphs_x) < 3:
                    continue

                # Combine features and edges
                x_combined = np.vstack(graphs_x)
                edge_index = np.hstack(graphs_edges) if len(graphs_edges) > 0 else np.zeros((2, 0), int)

                # Label from central scene: positive if at least one pedestrian is crossing
                label_val = scenes[i][1]['cross'].map(
                    {'not-crossing': 0, 'crossing': 1,
                     'noCrossRoad': 0, 'CrossRoad': 1}
                ).max()

                graph = Data(
                    x=torch.tensor(x_combined).float(),
                    edge_index=torch.tensor(edge_index, dtype=torch.long),
                    y=torch.tensor([label_val], dtype=torch.long)
                )

                data_list.append(graph)

            return data_list

        # Create training and test datasets
        if self._multi_pedestrian and self._temporal_type == 4:
            train_list = create_scene_temporal_graphs(self._csv_train, "Processing JAAD temporal (train)")
        else:
            train_list = create_graphs(self._csv_train, "Processing JAAD (train)")
        self.save(train_list, self.processed_paths[0])

        if self._multi_pedestrian and self._temporal_type == 4:
            test_list = create_scene_temporal_graphs(self._csv_test, "Processing JAAD temporal (test)")
        else:
            test_list = create_graphs(self._csv_test, "Processing JAAD (test)")
        self.save(test_list, self.processed_paths[1])


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_processed_data():
    """Remove processed .pt files to force reprocessing."""
    for f in ["data/processed/data_jaad.pt", "data/processed/data_jaad_test.pt"]:
        if os.path.exists(f):
            os.remove(f)
            print(f"Deleted: {f}")


def create_wandb_logger(config: Config, run_name_suffix: str = "") -> Optional[WandbLogger]:
    """
    Create WandB logger with informative run name.
    
    Args:
        config: Configuration object
        run_name_suffix: Optional suffix for run name (e.g., "_seed_0")
    
    Returns:
        WandbLogger if enabled, None otherwise
    """
    if not config.wandb.enabled:
        return None
    
    # Create informative run name
    run_name = (
        f"{config.model.name}_{config.dataset.name}_"
        f"G{config.graph.type}_T{config.graph.temporal_type}_"
        f"MP{config.graph.multi_pedestrian}{run_name_suffix}"
    )
    
    logger = WandbLogger(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=run_name,
        log_model=config.wandb.log_model,
        tags=config.wandb.tags
    )
    
    # Log configuration
    logger.experiment.config.update(config.to_dict())
    
    return logger


def train_single_run(config: Config, run_name_suffix: str = "") -> float:
    """
    Train a single model run.
    
    Args:
        config: Configuration object
        run_name_suffix: Optional suffix for run name
    
    Returns:
        Test accuracy
    """
    # Set seed
    pl.seed_everything(config.training.seed)
    
    # Clean processed data if requested
    if config.experiment.clean_processed:
        clean_processed_data()
    
    # Create datasets
    dts = JAAD(
        root='data',
        transform=T.Compose([T.ToUndirected()]),
        mode='train',
        config=config
    )
    
    dts_test = JAAD(
        root='data',
        transform=T.Compose([T.ToUndirected()]),
        mode='test',
        config=config
    )
    
    # Split train/val
    train_size = int(config.training.train_val_split * len(dts))
    val_size = len(dts) - train_size
    print(f"Dataset total: {len(dts)} | Train: {train_size} | Val: {val_size}")
    
    # Create data loaders
    train_loader = DataLoader(
        dts[:train_size],
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers
    )
    val_loader = DataLoader(
        dts[train_size:],
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers
    )
    test_loader = DataLoader(
        dts_test,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers
    )
    
    # Create model
    model = GraphLevelGNN(
        c_in=config.model.c_in,
        c_out=config.model.c_out,
        c_hidden=config.model.c_hidden,
        dp_rate_linear=config.model.dp_rate_linear,
        dp_rate=config.model.dp_rate,
        num_layers=config.model.num_layers,
        layer_name=config.model.name,
        learning_rate=config.training.learning_rate,
        lr_scheduler_patience=config.training.lr_scheduler_patience
    )
    
    # Create WandB logger
    wandb_logger = create_wandb_logger(config, run_name_suffix)
    
    # Create trainer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    trainer = pl.Trainer(
        default_root_dir=os.path.join(config.training.checkpoint_path, "GraphLevelJAAD"),
        callbacks=[
            ModelCheckpoint(save_weights_only=True, monitor="val_acc", mode="max"),
            EarlyStopping(
                monitor="val_loss",
                patience=config.training.early_stopping_patience,
                mode="min"
            )
        ],
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=config.training.max_epochs,
        enable_progress_bar=True,
        logger=wandb_logger
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    # Test
    best_model = GraphLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    print("\nFinal evaluation:")
    test_result = trainer.test(best_model, dataloaders=test_loader, verbose=True)
    test_acc = test_result[0]["test_acc"]
    
    # Cleanup WandB
    if config.wandb.enabled:
        wandb.finish()
    
    return test_acc


def train_multi_seed(config: Config) -> Dict[str, float]:
    """
    Run multiple seeds and compute statistics.
    
    Args:
        config: Configuration object
    
    Returns:
        Dictionary with mean, std, min, max accuracy
    """
    results = []
    
    for seed in config.experiment.seeds:
        print(f"\n{'='*80}")
        print(f"Running experiment with seed {seed}")
        print(f"{'='*80}")
        
        # Create config copy with specific seed
        seed_config = Config(
            graph=config.graph,
            model=config.model,
            dataset=config.dataset,
            training=TrainingConfig(
                **{**asdict(config.training), 'seed': seed}
            ),
            wandb=config.wandb,
            experiment=config.experiment
        )
        
        # Train single run
        test_acc = train_single_run(seed_config, run_name_suffix=f"_seed_{seed}")
        
        print(f"Seed {seed}: Test Acc = {test_acc:.4f}")
        results.append({"seed": seed, "test_acc": test_acc})
    
    # Compute statistics
    accs = np.array([r["test_acc"] for r in results])
    stats = {
        "mean": float(accs.mean()),
        "std": float(accs.std()),
        "min": float(accs.min()),
        "max": float(accs.max()),
        "range": float(accs.max() - accs.min())
    }
    
    print(f"\n{'='*80}")
    print("MULTI-SEED STATISTICS")
    print(f"{'='*80}")
    print(f"Mean:  {stats['mean']:.4f}")
    print(f"Std:   {stats['std']:.4f}")
    print(f"Min:   {stats['min']:.4f}")
    print(f"Max:   {stats['max']:.4f}")
    print(f"Range: {stats['range']:.4f}")
    print(f"{'='*80}")
    
    return stats


# ============================================================================
# CLI ARGUMENT PARSING
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="JAAD GNN Experiments with Configuration System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default config
  python src/JAAD_refactored.py
  
  # Use custom config file
  python src/JAAD_refactored.py --config configs/experiments/gcn_spatial.yaml
  
  # Override specific parameters
  python src/JAAD_refactored.py --model.name GAT --training.batch_size 64
  
  # Combine config file with overrides
  python src/JAAD_refactored.py --config configs/experiments/gat_temporal.yaml --training.max_epochs 200
  
  # Multi-seed experiment
  python src/JAAD_refactored.py --experiment.multi_seed true
        """
    )
    
    # Config file
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to YAML configuration file (default: configs/default.yaml)'
    )
    
    # Graph parameters
    parser.add_argument('--graph.type', type=int, help='Graph type (0-2)')
    parser.add_argument('--graph.temporal_type', type=int, help='Temporal type (0-4)')
    parser.add_argument('--graph.multi_pedestrian', type=lambda x: x.lower() == 'true', help='Multi-pedestrian mode')
    parser.add_argument('--graph.window_size', type=int, help='Window size')
    parser.add_argument('--graph.window_step', type=int, help='Window step')
    
    # Model parameters
    parser.add_argument('--model.name', type=str, help='Model name (GCN, GAT, GraphConv)')
    parser.add_argument('--model.c_hidden', type=int, help='Hidden channels')
    parser.add_argument('--model.num_layers', type=int, help='Number of layers')
    parser.add_argument('--model.dp_rate', type=float, help='Dropout rate')
    parser.add_argument('--model.dp_rate_linear', type=float, help='Linear dropout rate')
    
    # Dataset parameters
    parser.add_argument('--dataset.name', type=str, help='Dataset name')
    parser.add_argument('--dataset.folder', type=str, help='Dataset folder')
    
    # Training parameters
    parser.add_argument('--training.batch_size', type=int, help='Batch size')
    parser.add_argument('--training.max_epochs', type=int, help='Max epochs')
    parser.add_argument('--training.learning_rate', type=float, help='Learning rate')
    parser.add_argument('--training.seed', type=int, help='Random seed')
    parser.add_argument('--training.num_workers', type=int, help='Number of workers')
    
    # WandB parameters
    parser.add_argument('--wandb.enabled', type=lambda x: x.lower() == 'true', help='Enable WandB')
    parser.add_argument('--wandb.project', type=str, help='WandB project')
    parser.add_argument('--wandb.entity', type=str, help='WandB entity')
    
    # Experiment parameters
    parser.add_argument('--experiment.multi_seed', type=lambda x: x.lower() == 'true', help='Run multi-seed experiment')
    parser.add_argument('--experiment.clean_processed', type=lambda x: x.lower() == 'true', help='Clean processed data')
    
    return parser.parse_args()


def update_config_from_args(config: Config, args: argparse.Namespace) -> Config:
    """
    Update configuration with CLI arguments.
    
    Args:
        config: Base configuration
        args: Parsed arguments
    
    Returns:
        Updated configuration
    """
    overrides = []
    
    # Helper to get arg value (handles dot notation in attribute names)
    def get_arg(arg_obj, name):
        return getattr(arg_obj, name, None)
    
    # Graph overrides
    val = get_arg(args, 'graph.type')
    if val is not None:
        config.graph.type = val
        overrides.append(f"graph.type = {val}")
    
    val = get_arg(args, 'graph.temporal_type')
    if val is not None:
        config.graph.temporal_type = val
        overrides.append(f"graph.temporal_type = {val}")
    
    val = get_arg(args, 'graph.multi_pedestrian')
    if val is not None:
        config.graph.multi_pedestrian = val
        overrides.append(f"graph.multi_pedestrian = {val}")
    
    val = get_arg(args, 'graph.window_size')
    if val is not None:
        config.graph.window_size = val
        overrides.append(f"graph.window_size = {val}")
    
    val = get_arg(args, 'graph.window_step')
    if val is not None:
        config.graph.window_step = val
        overrides.append(f"graph.window_step = {val}")
    
    # Model overrides
    val = get_arg(args, 'model.name')
    if val is not None:
        config.model.name = val
        overrides.append(f"model.name = {val}")
    
    val = get_arg(args, 'model.c_hidden')
    if val is not None:
        config.model.c_hidden = val
        overrides.append(f"model.c_hidden = {val}")
    
    val = get_arg(args, 'model.num_layers')
    if val is not None:
        config.model.num_layers = val
        overrides.append(f"model.num_layers = {val}")
    
    val = get_arg(args, 'model.dp_rate')
    if val is not None:
        config.model.dp_rate = val
        overrides.append(f"model.dp_rate = {val}")
    
    val = get_arg(args, 'model.dp_rate_linear')
    if val is not None:
        config.model.dp_rate_linear = val
        overrides.append(f"model.dp_rate_linear = {val}")
    
    # Dataset overrides
    val = get_arg(args, 'dataset.name')
    if val is not None:
        config.dataset.name = val
        overrides.append(f"dataset.name = {val}")
    
    val = get_arg(args, 'dataset.folder')
    if val is not None:
        config.dataset.folder = val
        overrides.append(f"dataset.folder = {val}")
    
    # Training overrides
    val = get_arg(args, 'training.batch_size')
    if val is not None:
        config.training.batch_size = val
        overrides.append(f"training.batch_size = {val}")
    
    val = get_arg(args, 'training.max_epochs')
    if val is not None:
        config.training.max_epochs = val
        overrides.append(f"training.max_epochs = {val}")
    
    val = get_arg(args, 'training.learning_rate')
    if val is not None:
        config.training.learning_rate = val
        overrides.append(f"training.learning_rate = {val}")
    
    val = get_arg(args, 'training.seed')
    if val is not None:
        config.training.seed = val
        overrides.append(f"training.seed = {val}")
    
    val = get_arg(args, 'training.num_workers')
    if val is not None:
        config.training.num_workers = val
        overrides.append(f"training.num_workers = {val}")
    
    # WandB overrides
    val = get_arg(args, 'wandb.enabled')
    if val is not None:
        config.wandb.enabled = val
        overrides.append(f"wandb.enabled = {val}")
    
    val = get_arg(args, 'wandb.project')
    if val is not None:
        config.wandb.project = val
        overrides.append(f"wandb.project = {val}")
    
    val = get_arg(args, 'wandb.entity')
    if val is not None:
        config.wandb.entity = val
        overrides.append(f"wandb.entity = {val}")
    
    # Experiment overrides
    val = get_arg(args, 'experiment.multi_seed')
    if val is not None:
        config.experiment.multi_seed = val
        overrides.append(f"experiment.multi_seed = {val}")
    
    val = get_arg(args, 'experiment.clean_processed')
    if val is not None:
        config.experiment.clean_processed = val
        overrides.append(f"experiment.clean_processed = {val}")
    
    # Print overrides
    if overrides:
        print("\nCLI Overrides Applied:")
        for override in overrides:
            print(f"  - {override}")
    
    return config


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    print(f"\nLoading configuration from: {args.config}")
    config = Config.from_yaml(args.config)
    
    # Apply CLI overrides
    config = update_config_from_args(config, args)
    
    # Print final configuration
    print("\n" + "="*80)
    print("FINAL CONFIGURATION")
    print("="*80)
    print(f"Graph: type={config.graph.type}, temporal_type={config.graph.temporal_type}, "
          f"multi_pedestrian={config.graph.multi_pedestrian}")
    print(f"Model: {config.model.name}, hidden={config.model.c_hidden}, layers={config.model.num_layers}")
    print(f"Dataset: {config.dataset.name}")
    print(f"Training: batch_size={config.training.batch_size}, max_epochs={config.training.max_epochs}, "
          f"seed={config.training.seed}")
    print(f"WandB: enabled={config.wandb.enabled}, project={config.wandb.project}")
    print(f"Experiment: multi_seed={config.experiment.multi_seed}")
    print("="*80 + "\n")
    
    # Save effective configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_config_path = f"configs/runs/run_{timestamp}.yaml"
    config.save(run_config_path)
    print(f"Configuration saved to: {run_config_path}\n")
    
    # Run experiment
    if config.experiment.multi_seed:
        stats = train_multi_seed(config)
    else:
        test_acc = train_single_run(config)
        print(f"\nFinal Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
