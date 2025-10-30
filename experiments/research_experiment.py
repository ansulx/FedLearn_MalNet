"""
Research-Grade Federated Learning Experiment
===========================================
Professional experiment framework for graph-based federated learning research.
"""

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader as PyGDataLoader
import yaml
import os
import time
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResearchExperiment:
    """
    Research-grade federated learning experiment
    
    Features:
    - Professional experiment design
    - Comprehensive evaluation metrics
    - Research-grade logging and monitoring
    - Reproducible results
    """
    
    def __init__(self, config_path: str):
        """
        Initialize research experiment
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = self.config.get('server', {}).get('device', 'cuda')
        if self.device == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'
            logger.warning("CUDA not available, using CPU")
        
        # Initialize components
        self._setup_data_loaders()
        self._setup_models()
        self._setup_federated_learning()
        
        # Experiment tracking
        self.experiment_results = {
            'config': self.config,
            'start_time': time.time(),
            'rounds': [],
            'metrics': defaultdict(list),
            'model_info': {}
        }
        
        logger.info(f"Initialized ResearchExperiment on {self.device}")
    
    def _setup_data_loaders(self):
        """Setup data loaders"""
        from core.data_loader import MalNetGraphLoader
        
        self.data_loader = MalNetGraphLoader(self.config)
        self.train_loader, self.val_loader, self.test_loader = self.data_loader.create_data_loaders()
        
        # Get dataset statistics
        self.dataset_stats = self.data_loader.get_dataset_statistics()
        
        logger.info(f"Data loaders created: {len(self.train_loader.dataset)} train samples")
    
    def _setup_models(self):
        """Setup models"""
        from core.models import create_model
        
        # Create global model
        self.global_model = create_model(self.config)
        self.global_model = self.global_model.to(self.device)
        
        # Record model info
        self.experiment_results['model_info'] = {
            'num_parameters': self.global_model.count_parameters(),
            'model_size_mb': self.global_model.get_model_size(),
            'model_type': self.config['model']['gnn_type']
        }
        
        logger.info(f"Global model created: {self.experiment_results['model_info']['num_parameters']:,} parameters")
    
    def _setup_federated_learning(self):
        """Setup federated learning components"""
        from core.federated_learning import FederatedServer, FederatedClient
        from core.data_loader import GraphDataset
        
        # Create server
        self.server = FederatedServer(self.global_model, self.config)
        
        # Create clients
        self.clients = []
        num_clients = self.config['federated']['num_clients']
        
        # Split data for clients
        from core.data_splitter import create_federated_datasets
        client_datasets = create_federated_datasets(
            self.train_loader.dataset, 
            num_clients,
            self.config['federated']['split_strategy'],
            self.config['federated']['alpha']
        )
        
        # Create client data loaders
        for i, client_dataset in enumerate(client_datasets):
            client_loader = PyGDataLoader(
                client_dataset,
                batch_size=self.config['dataset']['batch_size'],
                shuffle=True,
                num_workers=self.config['dataset']['num_workers'],
                pin_memory=True
            )
            
            # Create client model
            from core.models import create_model
            client_model = create_model(self.config)
            
            # Create client
            client = FederatedClient(
                client_id=i,
                local_data=client_loader,
                model=client_model,
                config=self.config,
                device=self.device
            )
            
            self.clients.append(client)
        
        logger.info(f"Federated learning setup: {len(self.clients)} clients")
    
    def run_experiment(self) -> Dict:
        """
        Run the complete federated learning experiment
        
        Returns:
            Experiment results
        """
        logger.info("Starting research experiment...")
        
        # Initial evaluation
        initial_results = self.server.evaluate_global_model(self.test_loader)
        self._record_metrics(0, initial_results, 'initial')
        
        # Federated learning rounds
        num_rounds = self.config['federated']['num_rounds']
        participation_rate = self.config['federated']['participation_rate']
        local_epochs = self.config['federated']['local_epochs']
        
        for round_num in range(1, num_rounds + 1):
            logger.info(f"Starting round {round_num}/{num_rounds}")
            
            # Select participating clients
            num_participants = max(1, int(participation_rate * len(self.clients)))
            participating_clients = np.random.choice(
                len(self.clients), 
                size=num_participants, 
                replace=False
            )
            
            # Train participating clients
            client_updates = []
            for client_id in participating_clients:
                client = self.clients[client_id]
                
                # Get global weights
                global_weights = self.server.get_global_weights()
                
                # Train client
                client_results = client.train_local_model(global_weights, local_epochs)
                client_updates.append(client_results)
            
            # Run federated round
            round_results = self.server.run_federated_round(client_updates, self.test_loader)
            
            # Record metrics
            self._record_metrics(round_num, round_results, 'federated')
            
            # Log progress
            logger.info(f"Round {round_num} completed: "
                       f"Accuracy: {round_results['accuracy']:.2f}%, "
                       f"Loss: {round_results['loss']:.4f}")
            
            # Check for early stopping
            if self._should_early_stop():
                logger.info("Early stopping triggered")
                break
        
        # Final evaluation
        final_results = self.server.evaluate_global_model(self.test_loader)
        self._record_metrics(num_rounds + 1, final_results, 'final')
        
        # Compile results
        self.experiment_results['end_time'] = time.time()
        self.experiment_results['duration'] = self.experiment_results['end_time'] - self.experiment_results['start_time']
        
        logger.info("Research experiment completed!")
        return self.experiment_results
    
    def _record_metrics(self, round_num: int, results: Dict, phase: str):
        """Record experiment metrics"""
        self.experiment_results['rounds'].append(round_num)
        
        for metric, value in results.items():
            self.experiment_results['metrics'][f'{phase}_{metric}'].append(value)
    
    def _should_early_stop(self) -> bool:
        """Check if early stopping should be triggered"""
        patience = self.config.get('evaluation', {}).get('early_stopping_patience', 20)
        
        if len(self.experiment_results['rounds']) < patience:
            return False
        
        # Check if accuracy has improved in the last patience rounds
        recent_accuracies = self.experiment_results['metrics']['federated_accuracy'][-patience:]
        if len(recent_accuracies) < patience:
            return False
        
        best_accuracy = max(recent_accuracies)
        current_accuracy = recent_accuracies[-1]
        
        return current_accuracy < best_accuracy * 0.99  # 1% tolerance
    
    def save_results(self, filepath: str):
        """Save experiment results"""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_serializable(self.experiment_results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def _make_serializable(self, obj):
        """Make results JSON serializable"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        else:
            return obj
    
    def generate_report(self) -> str:
        """Generate experiment report"""
        report = []
        report.append("=" * 60)
        report.append("RESEARCH EXPERIMENT REPORT")
        report.append("=" * 60)
        
        # Configuration
        report.append(f"\nConfiguration:")
        report.append(f"  Model: {self.config['model']['gnn_type']}")
        report.append(f"  Clients: {self.config['federated']['num_clients']}")
        report.append(f"  Rounds: {self.config['federated']['num_rounds']}")
        report.append(f"  Device: {self.device}")
        
        # Model info
        report.append(f"\nModel Information:")
        report.append(f"  Parameters: {self.experiment_results['model_info']['num_parameters']:,}")
        report.append(f"  Size: {self.experiment_results['model_info']['model_size_mb']:.2f} MB")
        
        # Results
        if 'final_accuracy' in self.experiment_results['metrics']:
            final_acc = self.experiment_results['metrics']['final_accuracy'][-1]
            report.append(f"\nFinal Results:")
            report.append(f"  Accuracy: {final_acc:.2f}%")
        
        # Duration
        report.append(f"\nExperiment Duration: {self.experiment_results['duration']:.2f} seconds")
        
        return "\n".join(report)


def run_research_experiment(config_path: str = "config/research_config.yaml") -> Dict:
    """
    Run research-grade federated learning experiment
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Experiment results
    """
    # Create experiment
    experiment = ResearchExperiment(config_path)
    
    # Run experiment
    results = experiment.run_experiment()
    
    # Save results
    os.makedirs('results', exist_ok=True)
    experiment.save_results('results/experiment_results.json')
    
    # Generate report
    report = experiment.generate_report()
    print(report)
    
    # Save report
    with open('results/experiment_report.txt', 'w') as f:
        f.write(report)
    
    return results


if __name__ == "__main__":
    # Run research experiment
    results = run_research_experiment()
    
    print(f"\nExperiment completed!")
    print(f"Results saved to results/experiment_results.json")
    print(f"Report saved to results/experiment_report.txt")
