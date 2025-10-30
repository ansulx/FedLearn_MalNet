"""
Research-Grade Graph Data Loader
==============================
Professional graph dataset handling for federated learning research.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from typing import Dict, List, Tuple, Optional, Union
import logging
from collections import defaultdict
import yaml

logger = logging.getLogger(__name__)


class MalNetGraphLoader:
    """
    Research-grade graph data loader for MalNet dataset
    
    Features:
    - Efficient graph loading and preprocessing
    - Memory-optimized for large graphs
    - Research-grade data augmentation
    - Comprehensive statistics and analysis
    """
    
    def __init__(self, config: Dict):
        """
        Initialize graph data loader
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_dir = config['dataset']['path']
        self.max_nodes = config['dataset']['max_nodes']
        self.batch_size = config['dataset']['batch_size']
        self.num_workers = config['dataset']['num_workers']
        
        # Class mapping
        self.class_to_idx = {
            'benign': 0,
            'adware': 1, 
            'downloader': 2,
            'trojan': 3,
            'addisplay': 4
        }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        logger.info(f"Initialized MalNetGraphLoader with max_nodes={self.max_nodes}")
    
    def load_graph(self, graph_path: str) -> Data:
        """
        Load and preprocess a single graph
        
        Args:
            graph_path: Path to .edgelist file
            
        Returns:
            PyTorch Geometric Data object
        """
        try:
            # Read edgelist file
            edges = []
            num_nodes = 0
            
            with open(graph_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue
                    
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        try:
                            from_node = int(parts[0])
                            to_node = int(parts[1])
                            edges.append([from_node, to_node])
                            num_nodes = max(num_nodes, from_node, to_node)
                        except ValueError:
                            continue
            
            if not edges:
                return self._create_empty_graph()
            
            # Convert to tensor - ensure edge_index is Long type
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            num_nodes = num_nodes + 1
            
            # Graph size optimization
            if num_nodes > self.max_nodes:
                edge_index, num_nodes = self._subsample_graph(edge_index, num_nodes)
            
            # Ensure edge_index is still Long after subsampling
            if edge_index.dtype != torch.long:
                edge_index = edge_index.long()
            
            # Create node features (returns float32)
            node_features = self._create_node_features(edge_index, num_nodes)
            
            # Create graph data with explicit types
            data = Data(
                x=node_features.float(),  # Ensure float32
                edge_index=edge_index.long(),  # Ensure Long/int64
                num_nodes=num_nodes
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading graph {graph_path}: {e}")
            return self._create_empty_graph()
    
    def _subsample_graph(self, edge_index: torch.Tensor, num_nodes: int) -> Tuple[torch.Tensor, int]:
        """Subsample large graphs to fit memory constraints"""
        # Sample nodes uniformly
        node_indices = torch.randperm(num_nodes)[:self.max_nodes]
        node_mask = torch.zeros(num_nodes, dtype=torch.bool)
        node_mask[node_indices] = True
        
        # Filter edges (ensure Long type throughout)
        edge_mask = node_mask[edge_index[0].long()] & node_mask[edge_index[1].long()]
        edge_index = edge_index[:, edge_mask]
        
        # Remap node indices - create new tensor to ensure Long type
        new_edge_index = torch.zeros_like(edge_index, dtype=torch.long)
        node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(node_indices)}
        for i in range(edge_index.size(1)):
            new_edge_index[0, i] = node_mapping[edge_index[0, i].item()]
            new_edge_index[1, i] = node_mapping[edge_index[1, i].item()]
        
        return new_edge_index, self.max_nodes
    
    def _create_node_features(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Create simple node features (optimized for speed)"""
        if num_nodes == 0:
            return torch.zeros(1, 3, dtype=torch.float32)
        
        # Calculate node degrees efficiently using scatter operations
        # FIXED: Explicit dtype to prevent casting errors
        degrees = torch.zeros(num_nodes, dtype=torch.float32)
        in_degrees = torch.zeros(num_nodes, dtype=torch.float32)
        out_degrees = torch.zeros(num_nodes, dtype=torch.float32)
        
        if edge_index.size(1) > 0:
            # Create ones with same dtype as target tensors
            ones = torch.ones(edge_index.size(1), dtype=torch.float32)
            
            # Count out-degrees (ensure indices are long)
            out_degrees.scatter_add_(0, edge_index[0].long(), ones)
            # Count in-degrees
            in_degrees.scatter_add_(0, edge_index[1].long(), ones)
            # Total degree
            degrees = in_degrees + out_degrees
        
        # Stack features: [degree, in_degree, out_degree]
        features = torch.stack([degrees, in_degrees, out_degrees], dim=1)
        
        # RESEARCH-GRADE: Normalize features for better learning
        # Log transform to handle skewed degree distributions
        features = torch.log1p(features)  # log(1+x) to handle zeros
        # Standardize to mean=0, std=1
        feature_std = features.std(dim=0)
        if feature_std.sum() > 0:
            feature_mean = features.mean(dim=0)
            features = (features - feature_mean) / (feature_std + 1e-8)
        
        return features.float()
    
    def _calculate_clustering_coefficient(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Calculate local clustering coefficient for each node"""
        clustering = torch.zeros(num_nodes)
        
        # Simple clustering coefficient calculation
        for node in range(num_nodes):
            neighbors = set()
            for edge in edge_index.t():
                if edge[0] == node:
                    neighbors.add(edge[1].item())
                elif edge[1] == node:
                    neighbors.add(edge[0].item())
            
            if len(neighbors) < 2:
                clustering[node] = 0.0
            else:
                # Count triangles
                triangles = 0
                for n1 in neighbors:
                    for n2 in neighbors:
                        if n1 < n2:
                            # Check if edge exists
                            for edge in edge_index.t():
                                if (edge[0] == n1 and edge[1] == n2) or (edge[0] == n2 and edge[1] == n1):
                                    triangles += 1
                                    break
                
                max_possible = len(neighbors) * (len(neighbors) - 1) // 2
                clustering[node] = triangles / max_possible if max_possible > 0 else 0.0
        
        return clustering
    
    def _create_empty_graph(self) -> Data:
        """Create empty graph for error cases"""
        return Data(
            x=torch.zeros(1, 3, dtype=torch.float32),  # 3 features with explicit float type
            edge_index=torch.zeros(2, 0, dtype=torch.long),  # Edge indices must be Long
            num_nodes=1
        )
    
    def create_data_loaders(self) -> Tuple[PyGDataLoader, PyGDataLoader, PyGDataLoader]:
        """Create train, validation, and test data loaders"""
        # Load datasets
        train_dataset = self._create_dataset('train')
        val_dataset = self._create_dataset('val')
        test_dataset = self._create_dataset('test')
        
        # Create data loaders
        persistent_workers = self.num_workers > 0  # Only enable if workers are used
        
        train_loader = PyGDataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers
        )
        
        val_loader = PyGDataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers
        )
        
        test_loader = PyGDataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers
        )
        
        return train_loader, val_loader, test_loader
    
    def _create_dataset(self, split: str) -> 'GraphDataset':
        """Create dataset for specific split"""
        return GraphDataset(self, split)
    
    def get_dataset_statistics(self) -> Dict:
        """Get comprehensive dataset statistics"""
        stats = {}
        
        for split in ['train', 'val', 'test']:
            dataset = self._create_dataset(split)
            stats[split] = {
                'num_samples': len(dataset),
                'num_classes': len(self.class_to_idx),
                'class_distribution': dataset.get_class_distribution(),
                'graph_statistics': dataset.get_graph_statistics()
            }
        
        return stats


class GraphDataset(Dataset):
    """PyTorch Dataset for graph data"""
    
    def __init__(self, loader: MalNetGraphLoader, split: str):
        self.loader = loader
        self.split = split
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Dict]:
        """Load samples for the split with proper train/val/test splitting"""
        samples = []
        
        # Define class directories
        class_dirs = {
            'benign': 'benign/benign',
            'adware': 'adware/airpush',
            'downloader': 'downloader/jiagu', 
            'trojan': 'trojan/artemis',
            'addisplay': 'addisplay/kuguo'
        }
        
        for class_name, class_dir in class_dirs.items():
            class_path = os.path.join(self.loader.data_dir, class_dir)
            if os.path.exists(class_path):
                for filename in os.listdir(class_path):
                    if filename.endswith('.edgelist'):
                        samples.append({
                            'graph_path': os.path.join(class_path, filename),
                            'class_name': class_name,
                            'class_idx': self.loader.class_to_idx[class_name]
                        })
        
        # RESEARCH-GRADE: Proper train/val/test split
        import random
        random.seed(42)  # Reproducibility
        random.shuffle(samples)
        
        # 70% train, 15% val, 15% test split
        total = len(samples)
        train_end = int(0.7 * total)
        val_end = int(0.85 * total)
        
        if self.split == 'train':
            return samples[:train_end]
        elif self.split == 'val':
            return samples[train_end:val_end]
        else:  # test
            return samples[val_end:]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Data:
        """Get graph and label as PyG Data object"""
        sample = self.samples[idx]
        graph = self.loader.load_graph(sample['graph_path'])
        # Attach label to graph object (proper PyG format)
        graph.y = torch.tensor(sample['class_idx'], dtype=torch.long)
        return graph
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution"""
        distribution = defaultdict(int)
        for sample in self.samples:
            distribution[sample['class_name']] += 1
        return dict(distribution)
    
    def get_graph_statistics(self) -> Dict:
        """Get graph statistics"""
        if not self.samples:
            return {}
        
        # Sample a few graphs for statistics
        sample_graphs = []
        for i in range(min(100, len(self.samples))):
            graph, _ = self[i]
            sample_graphs.append(graph)
        
        if not sample_graphs:
            return {}
        
        num_nodes = [g.num_nodes for g in sample_graphs]
        num_edges = [g.edge_index.size(1) for g in sample_graphs]
        
        return {
            'avg_nodes': np.mean(num_nodes),
            'std_nodes': np.std(num_nodes),
            'avg_edges': np.mean(num_edges),
            'std_edges': np.std(num_edges),
            'max_nodes': max(num_nodes),
            'min_nodes': min(num_nodes)
        }
