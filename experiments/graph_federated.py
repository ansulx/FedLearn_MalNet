"""
Graph Federated Learning Training
===============================
Implements federated learning training with graph data.
"""

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader as PyGDataLoader
import time
import logging
from typing import Dict, List, Tuple, Optional
import os
import yaml
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphFederatedTrainer:
    """
    Federated learning trainer for graph-based malware detection
    
    Coordinates the federated learning process between server and clients.
    """
    
    def __init__(self, global_model: nn.Module, client_datasets: List[PyGDataLoader], 
                 test_loader: PyGDataLoader, config: Dict, device: str = 'cpu'):
        """
        Initialize graph federated trainer
        
        Args:
            global_model: Global GNN model to train
            client_datasets: List of client data loaders
            test_loader: Test data loader for evaluation
            config: Training configuration
            device: Device to run training on
        """
        self.global_model = global_model
        self.client_datasets = client_datasets
        self.test_loader = test_loader
        self.config = config
        self.device = device
        
        # Import here to avoid circular imports
        from federated.server import FLServer
        from federated.client import ClientManager, FLClient
        
        # Create server
        self.server = FLServer(
            global_model=global_model,
            aggregation_strategy=config.get('federated', {}).get('aggregation', 'fedavg')
        )
        
        # Create clients
        self.clients = []
        for i, client_data in enumerate(client_datasets):
            client = FLClient(
                client_id=i,
                local_data=client_data,
                model=global_model.__class__(**self._get_model_config()),
                learning_rate=config.get('training', {}).get('learning_rate', 0.001),
                device=device
            )
            self.clients.append(client)
        
        # Create client manager
        self.client_manager = ClientManager(self.clients)
        
        # Training history
        self.training_history = {
            'rounds': [],
            'global_accuracies': [],
            'global_losses': [],
            'client_accuracies': [],
            'round_times': [],
            'aggregation_times': []
        }
        
        logger.info(f"Initialized GraphFederatedTrainer with {len(client_datasets)} clients")
    
    def _get_model_config(self) -> Dict:
        """Get model configuration"""
        return {
            'num_classes': self.config.get('model', {}).get('num_classes', 5),
            'gnn_type': self.config.get('model', {}).get('gnn_type', 'gcn'),
            'hidden_dim': self.config.get('model', {}).get('hidden_dim', 64),
            'num_layers': self.config.get('model', {}).get('num_layers', 3),
            'dropout': self.config.get('model', {}).get('dropout', 0.5)
        }
    
    def train_round(self, round_num: int) -> Dict:
        """
        Train one federated learning round
        
        Args:
            round_num: Current round number
        
        Returns:
            Round results
        """
        round_start_time = time.time()
        
        logger.info(f"Starting graph federated round {round_num}")
        
        # Get global weights
        global_weights = self.server.get_global_weights()
        
        # Train clients
        participation_rate = self.config.get('federated', {}).get('participation_rate', 1.0)
        local_epochs = self.config.get('federated', {}).get('local_epochs', 5)
        
        client_updates = self.client_manager.train_clients(
            global_weights=global_weights,
            num_epochs=local_epochs,
            participation_rate=participation_rate
        )
        
        # Aggregate updates
        aggregation_start_time = time.time()
        round_results = self.server.run_federated_round(
            client_updates=client_updates,
            test_data=self.test_loader
        )
        aggregation_time = time.time() - aggregation_start_time
        
        # Evaluate clients
        client_results = self.client_manager.evaluate_clients(self.test_loader)
        client_accuracies = [result['accuracy'] for result in client_results]
        
        # Record metrics
        round_time = time.time() - round_start_time
        self.training_history['rounds'].append(round_num)
        self.training_history['global_accuracies'].append(round_results.get('accuracy', 0.0))
        self.training_history['global_losses'].append(round_results.get('loss', 0.0))
        self.training_history['client_accuracies'].append(client_accuracies)
        self.training_history['round_times'].append(round_time)
        self.training_history['aggregation_times'].append(aggregation_time)
        
        # Log results
        logger.info(f"Round {round_num} completed: "
                   f"Global Accuracy: {round_results.get('accuracy', 0.0):.2f}%, "
                   f"Global Loss: {round_results.get('loss', 0.0):.4f}, "
                   f"Round Time: {round_time:.2f}s")
        
        return round_results
    
    def train(self, num_rounds: Optional[int] = None) -> Dict:
        """
        Train the federated model for specified number of rounds
        
        Args:
            num_rounds: Number of rounds to train (uses config if None)
        
        Returns:
            Training history
        """
        if num_rounds is None:
            num_rounds = self.config.get('federated', {}).get('num_rounds', 10)
        
        logger.info(f"Starting graph federated training for {num_rounds} rounds")
        
        # Initial evaluation
        initial_results = self.server.evaluate_global_model(self.test_loader)
        logger.info(f"Initial Global Model - Accuracy: {initial_results['accuracy']:.2f}%, "
                   f"Loss: {initial_results['loss']:.4f}")
        
        # Training rounds
        for round_num in range(1, num_rounds + 1):
            round_results = self.train_round(round_num)
        
        # Final evaluation
        final_results = self.server.evaluate_global_model(self.test_loader)
        logger.info(f"Final Global Model - Accuracy: {final_results['accuracy']:.2f}%, "
                   f"Loss: {final_results['loss']:.4f}")
        
        logger.info("Graph federated training completed!")
        return self.training_history
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the federated model
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Global model evaluation
        global_results = self.server.evaluate_global_model(self.test_loader)
        
        # Client evaluations
        client_results = self.client_manager.evaluate_clients(self.test_loader)
        
        # Calculate statistics
        client_accuracies = [result['accuracy'] for result in client_results]
        client_losses = [result['loss'] for result in client_results]
        
        results = {
            'global_accuracy': global_results['accuracy'],
            'global_loss': global_results['loss'],
            'avg_client_accuracy': np.mean(client_accuracies),
            'std_client_accuracy': np.std(client_accuracies),
            'avg_client_loss': np.mean(client_losses),
            'std_client_loss': np.std(client_losses),
            'client_results': client_results
        }
        
        logger.info(f"Graph Federated Evaluation Results:")
        logger.info(f"  Global Accuracy: {results['global_accuracy']:.2f}%")
        logger.info(f"  Average Client Accuracy: {results['avg_client_accuracy']:.2f}% ± {results['std_client_accuracy']:.2f}%")
        logger.info(f"  Global Loss: {results['global_loss']:.4f}")
        logger.info(f"  Average Client Loss: {results['avg_client_loss']:.4f} ± {results['std_client_loss']:.4f}")
        
        return results


def run_graph_federated_experiment(data_dir: str, config_path: Optional[str] = None) -> Dict:
    """
    Run graph federated learning experiment
    
    Args:
        data_dir: Path to dataset directory
        config_path: Path to configuration file
    
    Returns:
        Experiment results
    """
    # Load configuration
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Import here to avoid circular imports
    from data.graph_dataset import MalNetGraphDataset
    from data.graph_splitter import create_federated_graph_datasets
    from models.gnn import create_gnn_model
    
    # Create datasets
    logger.info("Creating graph datasets...")
    
    # Load training data
    train_dataset = MalNetGraphDataset(data_dir, split='train', max_nodes=config.get('dataset', {}).get('max_nodes', 1000))
    test_dataset = MalNetGraphDataset(data_dir, split='test', max_nodes=config.get('dataset', {}).get('max_nodes', 1000))
    
    if len(train_dataset) == 0:
        raise ValueError("No training data found!")
    
    # Split data for federated learning
    logger.info("Splitting data for federated learning...")
    num_clients = config.get('federated', {}).get('num_clients', 5)
    split_strategy = config.get('federated', {}).get('split_strategy', 'dirichlet')
    alpha = config.get('federated', {}).get('alpha', 0.5)
    
    client_datasets, split_stats = create_federated_graph_datasets(
        train_dataset, num_clients, split_strategy, alpha
    )
    
    # Create data loaders for clients
    client_data_loaders = []
    for client_dataset in client_datasets:
        loader = PyGDataLoader(
            client_dataset,
            batch_size=config.get('dataset', {}).get('batch_size', 32),
            shuffle=True,
            num_workers=config.get('dataset', {}).get('num_workers', 2)
        )
        client_data_loaders.append(loader)
    
    # Create test loader
    test_loader = PyGDataLoader(
        test_dataset,
        batch_size=config.get('dataset', {}).get('batch_size', 32),
        shuffle=False,
        num_workers=config.get('dataset', {}).get('num_workers', 2)
    )
    
    # Create global model
    logger.info("Creating global GNN model...")
    global_model = create_gnn_model(
        model_type=config.get('model', {}).get('gnn_type', 'gcn'),
        num_classes=config.get('model', {}).get('num_classes', 5),
        hidden_dim=config.get('model', {}).get('hidden_dim', 64),
        num_layers=config.get('model', {}).get('num_layers', 3),
        dropout=config.get('model', {}).get('dropout', 0.5)
    )
    
    # Create federated trainer
    trainer = GraphFederatedTrainer(
        global_model=global_model,
        client_datasets=client_data_loaders,
        test_loader=test_loader,
        config=config,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Train federated model
    logger.info("Starting graph federated training...")
    training_history = trainer.train()
    
    # Evaluate federated model
    logger.info("Evaluating graph federated model...")
    evaluation_results = trainer.evaluate()
    
    # Save results
    if config.get('evaluation', {}).get('save_model', True):
        os.makedirs('models', exist_ok=True)
        trainer.server.save_global_model('models/graph_federated_model.pth')
    
    # Compile results
    results = {
        'training_history': training_history,
        'evaluation_results': evaluation_results,
        'split_stats': split_stats,
        'config': config,
        'model_info': {
            'num_parameters': sum(p.numel() for p in global_model.parameters() if p.requires_grad),
            'model_size_mb': sum(p.numel() * p.element_size() for p in global_model.parameters()) / 1024**2
        }
    }
    
    logger.info("Graph federated experiment completed!")
    return results


if __name__ == "__main__":
    # Test graph federated training
    print("Testing graph federated training...")
    
    data_dir = "malnet-graphs-tiny"
    config_path = "config.yaml"
    
    if os.path.exists(data_dir):
        results = run_graph_federated_experiment(data_dir, config_path)
        
        print(f"\nGraph Federated Results:")
        print(f"  Final Global Accuracy: {results['evaluation_results']['global_accuracy']:.2f}%")
        print(f"  Average Client Accuracy: {results['evaluation_results']['avg_client_accuracy']:.2f}%")
        print(f"  Model Parameters: {results['model_info']['num_parameters']:,}")
        print(f"  Model Size: {results['model_info']['model_size_mb']:.2f} MB")
    else:
        print(f"Dataset directory {data_dir} not found!")
