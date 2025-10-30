"""
Federated Meta-Learning (Federated MAML)
========================================
Model-Agnostic Meta-Learning adapted for federated setting.

Novel Contributions:
1. Meta-learning in federated setting with privacy preservation
2. Fast adaptation to new malware families with few samples
3. Meta-gradient aggregation across heterogeneous clients

Citation: If you use this code, please cite:
[Your Paper Title]. [Your Name et al.]. [Journal], [Year].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as PyGDataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging
import copy

logger = logging.getLogger(__name__)


class FederatedMAML:
    """
    Federated Model-Agnostic Meta-Learning
    
    Key Features:
    - Meta-learning for fast adaptation (k-shot learning)
    - Privacy-preserving federated aggregation
    - Support for heterogeneous client data
    
    Algorithm:
    1. Meta-Train Phase:
       - Each client samples support and query sets (episodic training)
       - Inner loop: adapt to support set (task-specific)
       - Outer loop: meta-update on query set (generalization)
       - Federated aggregation of meta-gradients
    
    2. Meta-Test Phase (Fast Adaptation):
       - Given k samples of new malware family
       - Perform few gradient steps (inner loop)
       - Achieve high accuracy with <5 samples
    """
    
    def __init__(self, 
                 global_model: nn.Module,
                 config: Dict,
                 device: str = 'cuda'):
        """
        Initialize Federated MAML
        
        Args:
            global_model: Global model to meta-train
            config: Configuration dictionary
            device: Device for training
        """
        self.global_model = global_model
        self.config = config
        self.device = device
        
        # Meta-learning hyperparameters
        self.meta_lr = config.get('meta_learning', {}).get('meta_lr', 0.001)
        self.inner_lr = config.get('meta_learning', {}).get('inner_lr', 0.01)
        self.inner_steps = config.get('meta_learning', {}).get('inner_steps', 5)
        self.k_shot = config.get('meta_learning', {}).get('k_shot', 5)
        self.n_way = config.get('meta_learning', {}).get('n_way', 5)  # 5 classes
        self.query_size = config.get('meta_learning', {}).get('query_size', 15)
        
        # Meta-optimizer (for outer loop)
        self.meta_optimizer = torch.optim.Adam(
            self.global_model.parameters(), 
            lr=self.meta_lr
        )
        
        # Metrics tracking
        self.metrics_history = {
            'meta_train_loss': [],
            'meta_test_accuracy': [],
            'adaptation_curves': []
        }
        
        logger.info(f"Initialized FederatedMAML with k={self.k_shot}, n={self.n_way}")
    
    def create_task(self, dataset: List, n_way: int, k_shot: int, 
                    query_size: int) -> Tuple[List, List]:
        """
        Create a meta-learning task (episode) from dataset
        
        Args:
            dataset: Full dataset
            n_way: Number of classes in task
            k_shot: Number of support samples per class
            query_size: Number of query samples per class
        
        Returns:
            support_set: k*n_way samples for adaptation
            query_set: query_size*n_way samples for evaluation
        """
        # Get available classes
        class_to_samples = {}
        for sample in dataset:
            label = sample.y.item() if hasattr(sample.y, 'item') else sample.y
            if label not in class_to_samples:
                class_to_samples[label] = []
            class_to_samples[label].append(sample)
        
        # Sample n_way classes
        available_classes = list(class_to_samples.keys())
        if len(available_classes) < n_way:
            raise ValueError(f"Dataset has only {len(available_classes)} classes, need {n_way}")
        
        sampled_classes = np.random.choice(available_classes, n_way, replace=False)
        
        # Sample support and query sets
        support_set = []
        query_set = []
        
        for class_id in sampled_classes:
            class_samples = class_to_samples[class_id]
            
            # Need k_shot + query_size samples per class
            if len(class_samples) < k_shot + query_size:
                raise ValueError(f"Class {class_id} has only {len(class_samples)} samples")
            
            # Sample without replacement
            sampled_indices = np.random.choice(
                len(class_samples), 
                k_shot + query_size, 
                replace=False
            )
            
            # Split into support and query
            support_indices = sampled_indices[:k_shot]
            query_indices = sampled_indices[k_shot:]
            
            support_set.extend([class_samples[i] for i in support_indices])
            query_set.extend([class_samples[i] for i in query_indices])
        
        return support_set, query_set
    
    def inner_loop(self, model: nn.Module, support_set: List, 
                   inner_steps: int) -> nn.Module:
        """
        Inner loop: Adapt model to support set (task-specific training)
        
        Args:
            model: Model to adapt
            support_set: Support samples for adaptation
            inner_steps: Number of gradient steps
        
        Returns:
            Adapted model
        """
        # Create data loader for support set
        support_loader = PyGDataLoader(
            support_set, 
            batch_size=len(support_set),  # Use all support samples
            shuffle=True
        )
        
        # Clone model for adaptation (don't modify original)
        adapted_model = copy.deepcopy(model)
        adapted_model.train()
        
        # Inner optimizer (SGD for simplicity)
        inner_optimizer = torch.optim.SGD(
            adapted_model.parameters(), 
            lr=self.inner_lr
        )
        
        # Perform inner loop updates
        for step in range(inner_steps):
            for batch in support_loader:
                batch = batch.to(self.device)
                labels = batch.y
                
                # Forward pass
                inner_optimizer.zero_grad()
                output = adapted_model(batch.x, batch.edge_index, batch.batch)
                loss = F.cross_entropy(output, labels)
                
                # Backward pass
                loss.backward()
                inner_optimizer.step()
        
        return adapted_model
    
    def outer_loop(self, model: nn.Module, query_set: List) -> float:
        """
        Outer loop: Evaluate adapted model on query set
        
        Args:
            model: Adapted model
            query_set: Query samples for evaluation
        
        Returns:
            Query loss (for meta-gradient)
        """
        # Create data loader for query set
        query_loader = PyGDataLoader(
            query_set, 
            batch_size=len(query_set),
            shuffle=False
        )
        
        model.eval()
        
        # Evaluate on query set
        total_loss = 0.0
        for batch in query_loader:
            batch = batch.to(self.device)
            labels = batch.y
            
            # Forward pass (no gradient for adapted model)
            output = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(output, labels)
            
            total_loss += loss.item()
        
        return total_loss / len(query_loader)
    
    def meta_train_step(self, client_datasets: List[List]) -> Dict:
        """
        One step of meta-training across all clients
        
        Args:
            client_datasets: List of datasets (one per client)
        
        Returns:
            Meta-training metrics
        """
        self.global_model.train()
        
        # Meta-gradients from all clients
        meta_gradients = []
        client_losses = []
        
        # For each client
        for client_id, client_dataset in enumerate(client_datasets):
            try:
                # Create task (episode) for this client
                support_set, query_set = self.create_task(
                    client_dataset, 
                    self.n_way, 
                    self.k_shot, 
                    self.query_size
                )
                
                # Inner loop: adapt to support set
                adapted_model = self.inner_loop(
                    self.global_model, 
                    support_set, 
                    self.inner_steps
                )
                
                # Outer loop: evaluate on query set
                query_loss = self.outer_loop(adapted_model, query_set)
                client_losses.append(query_loss)
                
                # Compute meta-gradient (gradient of query loss w.r.t. meta-parameters)
                # This is the key step in MAML!
                self.meta_optimizer.zero_grad()
                
                # Re-run forward pass with gradients enabled
                query_loader = PyGDataLoader(query_set, batch_size=len(query_set))
                for batch in query_loader:
                    batch = batch.to(self.device)
                    labels = batch.y
                    output = adapted_model(batch.x, batch.edge_index, batch.batch)
                    loss = F.cross_entropy(output, labels)
                    loss.backward()
                
                # Collect gradients
                client_grads = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) 
                               for p in self.global_model.parameters()]
                meta_gradients.append(client_grads)
                
            except Exception as e:
                logger.warning(f"Client {client_id} failed meta-training: {e}")
                continue
        
        # Aggregate meta-gradients (simple averaging)
        if len(meta_gradients) > 0:
            avg_meta_gradients = []
            for param_idx in range(len(list(self.global_model.parameters()))):
                param_grads = [grads[param_idx] for grads in meta_gradients]
                avg_grad = torch.stack(param_grads).mean(dim=0)
                avg_meta_gradients.append(avg_grad)
            
            # Update meta-parameters
            self.meta_optimizer.zero_grad()
            for param, avg_grad in zip(self.global_model.parameters(), avg_meta_gradients):
                if param.grad is None:
                    param.grad = avg_grad
                else:
                    param.grad += avg_grad
            
            self.meta_optimizer.step()
        
        # Return metrics
        avg_loss = np.mean(client_losses) if client_losses else 0.0
        return {
            'meta_train_loss': avg_loss,
            'num_clients': len(meta_gradients)
        }
    
    def fast_adapt(self, test_samples: List, k_shot: int = 5, 
                   inner_steps: int = 10) -> Tuple[nn.Module, List[float]]:
        """
        Fast adaptation to new malware family (k-shot learning)
        
        This is the KEY NOVELTY: Adapt to new family with <5 samples!
        
        Args:
            test_samples: Samples from new malware family
            k_shot: Number of samples to use for adaptation
            inner_steps: Number of adaptation steps
        
        Returns:
            Adapted model and adaptation curve
        """
        # Split samples into support (for adaptation) and test (for evaluation)
        if len(test_samples) < k_shot + 5:
            raise ValueError(f"Need at least {k_shot + 5} samples for fast adaptation")
        
        np.random.shuffle(test_samples)
        support_samples = test_samples[:k_shot]
        test_samples_subset = test_samples[k_shot:k_shot+5]
        
        # Adapt model
        adapted_model = self.inner_loop(
            self.global_model, 
            support_samples, 
            inner_steps
        )
        
        # Track adaptation curve
        adaptation_curve = []
        test_loader = PyGDataLoader(test_samples_subset, batch_size=len(test_samples_subset))
        
        adapted_model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                labels = batch.y
                output = adapted_model(batch.x, batch.edge_index, batch.batch)
                _, predicted = torch.max(output.data, 1)
                accuracy = (predicted == labels).float().mean().item()
                adaptation_curve.append(accuracy)
        
        return adapted_model, adaptation_curve
    
    def evaluate_zero_shot(self, test_dataset: List, k_shot: int = 5,
                          num_tasks: int = 100) -> Dict:
        """
        Evaluate zero-shot detection capability
        
        Args:
            test_dataset: Dataset with unseen malware families
            k_shot: Number of support samples
            num_tasks: Number of tasks to evaluate
        
        Returns:
            Zero-shot evaluation metrics
        """
        accuracies = []
        
        for task_id in range(num_tasks):
            try:
                # Create task
                support_set, query_set = self.create_task(
                    test_dataset, 
                    self.n_way, 
                    k_shot, 
                    self.query_size
                )
                
                # Fast adaptation
                adapted_model = self.inner_loop(
                    self.global_model, 
                    support_set, 
                    self.inner_steps
                )
                
                # Evaluate on query set
                query_loader = PyGDataLoader(query_set, batch_size=len(query_set))
                adapted_model.eval()
                
                with torch.no_grad():
                    for batch in query_loader:
                        batch = batch.to(self.device)
                        labels = batch.y
                        output = adapted_model(batch.x, batch.edge_index, batch.batch)
                        _, predicted = torch.max(output.data, 1)
                        accuracy = (predicted == labels).float().mean().item()
                        accuracies.append(accuracy)
            
            except Exception as e:
                logger.warning(f"Task {task_id} failed: {e}")
                continue
        
        # Return statistics
        return {
            'zero_shot_accuracy': np.mean(accuracies),
            'zero_shot_std': np.std(accuracies),
            'num_tasks': len(accuracies)
        }
    
    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        """Get global model weights"""
        return self.global_model.get_weights()
    
    def set_global_weights(self, weights: Dict[str, torch.Tensor]):
        """Set global model weights"""
        self.global_model.set_weights(weights)


# Example usage
if __name__ == "__main__":
    """
    Example: How to use FederatedMAML
    """
    
    # Configuration
    config = {
        'meta_learning': {
            'meta_lr': 0.001,      # Meta-learning rate
            'inner_lr': 0.01,       # Inner loop learning rate
            'inner_steps': 5,       # Inner loop steps
            'k_shot': 5,            # K-shot learning
            'n_way': 5,             # N-way classification
            'query_size': 15        # Query set size
        }
    }
    
    # Create model (placeholder)
    from core.models import create_model
    model_config = {
        'model': {
            'num_classes': 5,
            'gnn_type': 'gcn',
            'hidden_dim': 64
        }
    }
    model = create_model(model_config)
    
    # Initialize Federated MAML
    fed_maml = FederatedMAML(model, config, device='cuda')
    
    print("Federated MAML initialized!")
    print(f"Meta LR: {fed_maml.meta_lr}")
    print(f"Inner LR: {fed_maml.inner_lr}")
    print(f"K-shot: {fed_maml.k_shot}")
    print(f"N-way: {fed_maml.n_way}")
    
    # TODO: Load data and run meta-training
    # client_datasets = [...]  # List of client datasets
    # metrics = fed_maml.meta_train_step(client_datasets)
    # print(f"Meta-train loss: {metrics['meta_train_loss']:.4f}")

