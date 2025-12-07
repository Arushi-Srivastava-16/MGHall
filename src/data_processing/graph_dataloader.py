"""
Graph Data Loader.

This module provides data loaders for PyTorch Geometric graphs with caching.
"""

import json
import pickle
from pathlib import Path
from typing import List, Optional, Callable
from torch.utils.data import Dataset
from torch_geometric.data import Data, DataLoader
from tqdm import tqdm

from ..data_processing.unified_schema import ReasoningChain
from ..graph_construction.crg_builder import build_crg
from ..graph_construction.feature_extractor import FeatureExtractor


class ReasoningChainDataset(Dataset):
    """Dataset for reasoning chains converted to graphs."""
    
    def __init__(
        self,
        data_path: Path,
        feature_extractor: Optional[FeatureExtractor] = None,
        cache_path: Optional[Path] = None,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to JSONL file with reasoning chains
            feature_extractor: Feature extractor instance
            cache_path: Optional path to cache processed graphs
            transform: Optional transform function
        """
        self.data_path = data_path
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.cache_path = cache_path
        self.transform = transform
        
        # Load chains
        self.chains = self._load_chains()
        
        # Load or build cache
        if cache_path and cache_path.exists():
            print(f"Loading cached graphs from {cache_path}")
            self.graphs = self._load_cache()
        else:
            print("Building graphs (this may take a while)...")
            self.graphs = self._build_graphs()
            if cache_path:
                self._save_cache()
    
    def _load_chains(self) -> List[ReasoningChain]:
        """Load reasoning chains from JSONL file."""
        chains = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Loading {self.data_path.name}"):
                chain_dict = json.loads(line)
                chain = ReasoningChain.from_dict(chain_dict)
                chains.append(chain)
        return chains
    
    def _build_graphs(self) -> List[Data]:
        """Build PyG graphs from reasoning chains."""
        graphs = []
        for chain in tqdm(self.chains, desc="Building graphs"):
            # Extract features
            node_features = self.feature_extractor.extract_all_features(chain)
            
            # Build CRG
            graph = build_crg(chain, node_features=node_features)
            
            if self.transform:
                graph = self.transform(graph)
            
            graphs.append(graph)
        
        return graphs
    
    def _load_cache(self) -> List[Data]:
        """Load graphs from cache."""
        with open(self.cache_path, "rb") as f:
            return pickle.load(f)
    
    def _save_cache(self) -> None:
        """Save graphs to cache."""
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.graphs, f)
            print(f"Cached graphs to {self.cache_path}")
    
    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx: int) -> Data:
        return self.graphs[idx]


def create_dataloaders(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    batch_size: int = 32,
    num_workers: int = 0,
    cache_dir: Optional[Path] = None,
    feature_extractor: Optional[FeatureExtractor] = None,
) -> tuple:
    """
    Create train/val/test data loaders.
    
    Args:
        train_path: Path to training JSONL file
        val_path: Path to validation JSONL file
        test_path: Path to test JSONL file
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        cache_dir: Optional directory for caching graphs
        feature_extractor: Optional feature extractor instance
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    cache_dir = cache_dir or Path("data/graphs")
    
    train_cache = cache_dir / f"{train_path.stem}.pkl" if cache_dir else None
    val_cache = cache_dir / f"{val_path.stem}.pkl" if cache_dir else None
    test_cache = cache_dir / f"{test_path.stem}.pkl" if cache_dir else None
    
    train_dataset = ReasoningChainDataset(
        train_path,
        feature_extractor=feature_extractor,
        cache_path=train_cache,
    )
    val_dataset = ReasoningChainDataset(
        val_path,
        feature_extractor=feature_extractor,
        cache_path=val_cache,
    )
    test_dataset = ReasoningChainDataset(
        test_path,
        feature_extractor=feature_extractor,
        cache_path=test_cache,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    data_dir = Path("../../data/processed/splits")
    
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    test_path = data_dir / "test.jsonl"
    
    if train_path.exists() and val_path.exists() and test_path.exists():
        train_loader, val_loader, test_loader = create_dataloaders(
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            batch_size=16,
        )
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Test a batch
        for batch in train_loader:
            print(f"Batch size: {batch.num_graphs}")
            print(f"Node features shape: {batch.x.shape}")
            print(f"Edge index shape: {batch.edge_index.shape}")
            break
    else:
        print("Split files not found. Run splitter first.")

