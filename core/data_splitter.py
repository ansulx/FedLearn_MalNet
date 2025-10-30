"""
Research-Grade Data Splitter
===========================
Professional data splitting for federated learning research.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from typing import List, Dict, Tuple, Optional
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class ResearchDataSplitter:
    """
    Research-grade data splitter for federated learning
    
    Features:
    - Multiple splitting strategies
    - Non-IID data distribution
    - Research-grade evaluation metrics
    - Reproducible results
    """
    
    def __init__(self, num_clients: int, split_strategy: str = 'dirichlet', 
                 alpha: float = 0.5, seed: int = 42):
        """
        Initialize research data splitter
        
        Args:
            num_clients: Number of clients to split data across
            split_strategy: Strategy for splitting ('iid', 'non_iid', 'dirichlet')
            alpha: Parameter for Dirichlet distribution
            seed: Random seed for reproducibility
        """
        self.num_clients = num_clients
        self.split_strategy = split_strategy
        self.alpha = alpha
        self.seed = seed
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        logger.info(f"Initialized ResearchDataSplitter: {num_clients} clients, "
                   f"strategy={split_strategy}, alpha={alpha}")
    
    def split_dataset(self, dataset: Dataset) -> List[Subset]:
        """
        Split dataset across clients
        
        Args:
            dataset: Dataset to split
        
        Returns:
            List of client datasets
        """
        if self.split_strategy == 'iid':
            return self._split_iid(dataset)
        elif self.split_strategy == 'non_iid':
            return self._split_non_iid(dataset)
        elif self.split_strategy == 'dirichlet':
            return self._split_dirichlet(dataset)
        else:
            raise ValueError(f"Unknown split strategy: {self.split_strategy}")
    
    def _split_iid(self, dataset: Dataset) -> List[Subset]:
        """Split dataset in IID manner"""
        logger.info("Splitting dataset in IID manner")
        
        total_samples = len(dataset)
        indices = list(range(total_samples))
        
        # Shuffle indices
        np.random.shuffle(indices)
        
        # Split into equal parts
        samples_per_client = total_samples // self.num_clients
        client_indices = []
        
        for i in range(self.num_clients):
            start_idx = i * samples_per_client
            if i == self.num_clients - 1:  # Last client gets remaining samples
                end_idx = total_samples
            else:
                end_idx = (i + 1) * samples_per_client
            
            client_indices.append(indices[start_idx:end_idx])
        
        # Create subsets
        client_datasets = [Subset(dataset, indices) for indices in client_indices]
        
        logger.info(f"IID split: {total_samples} samples across {self.num_clients} clients")
        return client_datasets
    
    def _split_non_iid(self, dataset: Dataset) -> List[Subset]:
        """Split dataset in non-IID manner (each client gets specific classes)"""
        logger.info("Splitting dataset in non-IID manner")
        
        # Group samples by class
        class_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            class_to_indices[label].append(idx)
        
        # Get class names
        classes = list(class_to_indices.keys())
        num_classes = len(classes)
        
        # Assign classes to clients
        classes_per_client = max(1, num_classes // self.num_clients)
        client_indices = [[] for _ in range(self.num_clients)]
        
        for i, class_label in enumerate(classes):
            client_id = i % self.num_clients
            client_indices[client_id].extend(class_to_indices[class_label])
        
        # Create subsets
        client_datasets = [Subset(dataset, indices) for indices in client_indices]
        
        logger.info(f"Non-IID split: {num_classes} classes across {self.num_clients} clients")
        return client_datasets
    
    def _split_dirichlet(self, dataset: Dataset) -> List[Subset]:
        """Split dataset using Dirichlet distribution"""
        logger.info(f"Splitting dataset using Dirichlet distribution (alpha={self.alpha})")
        
        # Group samples by class
        class_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            class_to_indices[label].append(idx)
        
        # Get class names
        classes = list(class_to_indices.keys())
        num_classes = len(classes)
        
        # Initialize client indices
        client_indices = [[] for _ in range(self.num_clients)]
        
        # Split each class using Dirichlet distribution
        for class_label in classes:
            class_indices = class_to_indices[class_label]
            num_samples = len(class_indices)
            
            if num_samples == 0:
                continue
            
            # Generate Dirichlet distribution
            proportions = np.random.dirichlet([self.alpha] * self.num_clients)
            
            # Assign samples to clients based on proportions
            start_idx = 0
            for client_id in range(self.num_clients):
                end_idx = start_idx + int(proportions[client_id] * num_samples)
                if client_id == self.num_clients - 1:  # Last client gets remaining samples
                    end_idx = num_samples
                
                client_indices[client_id].extend(class_indices[start_idx:end_idx])
                start_idx = end_idx
        
        # Create subsets
        client_datasets = [Subset(dataset, indices) for indices in client_indices]
        
        logger.info(f"Dirichlet split: {num_classes} classes across {self.num_clients} clients")
        return client_datasets
    
    def get_split_statistics(self, dataset: Dataset, client_datasets: List[Subset]) -> Dict:
        """
        Get comprehensive split statistics
        
        Args:
            dataset: Original dataset
            client_datasets: List of client datasets
        
        Returns:
            Split statistics dictionary
        """
        stats = {
            'total_samples': len(dataset),
            'num_clients': len(client_datasets),
            'samples_per_client': [len(client_dataset) for client_dataset in client_datasets],
            'class_distribution_per_client': [],
            'split_strategy': self.split_strategy,
            'alpha': self.alpha if self.split_strategy == 'dirichlet' else None
        }
        
        # Analyze class distribution for each client
        for client_id, client_dataset in enumerate(client_datasets):
            class_counts = defaultdict(int)
            for idx in client_dataset.indices:
                _, label = dataset[idx]
                class_counts[label] += 1
            
            stats['class_distribution_per_client'].append(dict(class_counts))
        
        # Calculate additional metrics
        stats['data_imbalance'] = self._calculate_data_imbalance(stats['samples_per_client'])
        stats['class_imbalance'] = self._calculate_class_imbalance(stats['class_distribution_per_client'])
        
        return stats
    
    def _calculate_data_imbalance(self, samples_per_client: List[int]) -> float:
        """Calculate data imbalance across clients"""
        if not samples_per_client:
            return 0.0
        
        mean_samples = np.mean(samples_per_client)
        std_samples = np.std(samples_per_client)
        
        return std_samples / mean_samples if mean_samples > 0 else 0.0
    
    def _calculate_class_imbalance(self, class_distributions: List[Dict]) -> float:
        """Calculate class imbalance across clients"""
        if not class_distributions:
            return 0.0
        
        # Calculate total class distribution
        total_classes = defaultdict(int)
        for client_dist in class_distributions:
            for class_name, count in client_dist.items():
                total_classes[class_name] += count
        
        # Calculate imbalance for each class
        class_imbalances = []
        for class_name in total_classes.keys():
            class_counts = [client_dist.get(class_name, 0) for client_dist in class_distributions]
            mean_count = np.mean(class_counts)
            std_count = np.std(class_counts)
            
            if mean_count > 0:
                class_imbalances.append(std_count / mean_count)
        
        return np.mean(class_imbalances) if class_imbalances else 0.0


def create_federated_datasets(dataset: Dataset, num_clients: int, 
                            split_strategy: str = 'dirichlet', 
                            alpha: float = 0.5) -> List[Subset]:
    """
    Create federated datasets
    
    Args:
        dataset: Dataset to split
        num_clients: Number of clients
        split_strategy: Strategy for splitting ('iid', 'non_iid', 'dirichlet')
        alpha: Parameter for Dirichlet distribution
    
    Returns:
        List of client datasets
    """
    splitter = ResearchDataSplitter(num_clients, split_strategy, alpha)
    client_datasets = splitter.split_dataset(dataset)
    
    return client_datasets
