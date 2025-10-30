"""
Research-Grade Federated Learning Framework
==========================================
Professional federated learning implementation for graph-based malware detection.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader as PyGDataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


class FederatedServer:
    """
    Research-grade federated learning server
    
    Features:
    - Multiple aggregation strategies
    - Privacy-preserving mechanisms
    - Comprehensive logging and monitoring
    - Research-grade evaluation metrics
    """
     
    def __init__(self, global_model: nn.Module, config: Dict):
        """
        Initialize federated server
        
        Args:
            global_model: Global model to coordinate
            config: Server configuration
        """
        self.global_model = global_model
        self.config = config
        self.device = config.get('server', {}).get('device', 'cuda')
        
        # Aggregation strategy
        self.aggregation_strategy = config.get('federated', {}).get('aggregation', 'fedavg')
        
        # Server state
        self.round_number = 0
        self.global_weights = self.global_model.get_weights()
        
        # Privacy mechanisms
        self.privacy_enabled = config.get('privacy', {}).get('enabled', False)
        if self.privacy_enabled:
            from core.privacy import DifferentialPrivacy
            self.privacy_mechanism = DifferentialPrivacy(config['privacy'])
        
        # Metrics tracking
        self.metrics_history = {
            'rounds': [],
            'global_accuracies': [],
            'global_losses': [],
            'client_participation': [],
            'aggregation_times': [],
            'privacy_budgets': []
        }
        
        logger.info(f"Initialized FederatedServer with {self.aggregation_strategy} aggregation")
    
    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        """Get current global model weights"""
        return self.global_weights
    
    def aggregate_updates(self, client_updates: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates using specified strategy
        
        Args:
            client_updates: List of client updates (weights, sample_count)
        
        Returns:
            Aggregated weights
        """
        start_time = time.time()
        
        if self.aggregation_strategy == 'fedavg':
            aggregated_weights = self._fedavg_aggregation(client_updates)
        elif self.aggregation_strategy == 'fedmedian':
            aggregated_weights = self._fedmedian_aggregation(client_updates)
        elif self.aggregation_strategy == 'krum':
            aggregated_weights = self._krum_aggregation(client_updates)
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.aggregation_strategy}")
        
        # Apply privacy mechanisms
        if self.privacy_enabled:
            aggregated_weights = self.privacy_mechanism.add_noise(aggregated_weights)
        
        # Update global weights
        self.global_weights = aggregated_weights
        self.global_model.set_weights(aggregated_weights)
        
        # Record metrics
        aggregation_time = time.time() - start_time
        self.metrics_history['aggregation_times'].append(aggregation_time)
        
        logger.info(f"Aggregation completed in {aggregation_time:.2f}s using {self.aggregation_strategy}")
        
        return aggregated_weights
    
    def _fedavg_aggregation(self, client_updates: List[Dict]) -> Dict[str, torch.Tensor]:
        """Federated Averaging aggregation"""
        if not client_updates:
            return self.global_weights
        
        # Calculate total samples
        total_samples = sum(update['sample_count'] for update in client_updates)
        
        # Weighted average
        aggregated_weights = {}
        for key in self.global_weights.keys():
            weighted_sum = torch.zeros_like(self.global_weights[key])
            
            for update in client_updates:
                weight = update['sample_count'] / total_samples
                weighted_sum += weight * update['weights'][key]
            
            aggregated_weights[key] = weighted_sum
        
        return aggregated_weights
    
    def _fedmedian_aggregation(self, client_updates: List[Dict]) -> Dict[str, torch.Tensor]:
        """FedMedian aggregation (robust to outliers)"""
        if not client_updates:
            return self.global_weights
        
        aggregated_weights = {}
        for key in self.global_weights.keys():
            # Collect all client weights for this parameter
            client_weights = [update['weights'][key] for update in client_updates]
            
            # Calculate median
            stacked_weights = torch.stack(client_weights, dim=0)
            median_weights = torch.median(stacked_weights, dim=0)[0]
            
            aggregated_weights[key] = median_weights
        
        return aggregated_weights
    
    def _krum_aggregation(self, client_updates: List[Dict]) -> Dict[str, torch.Tensor]:
        """Krum aggregation (Byzantine-robust)"""
        if len(client_updates) < 3:
            return self._fedavg_aggregation(client_updates)
        
        # Select most reliable client using Krum
        selected_client = self._select_krum_client(client_updates)
        
        # Use selected client's weights
        return client_updates[selected_client]['weights']
    
    def _select_krum_client(self, client_updates: List[Dict]) -> int:
        """Select client using Krum algorithm"""
        num_clients = len(client_updates)
        f = max(1, num_clients // 4)  # Maximum number of Byzantine clients
        
        krum_scores = []
        for i in range(num_clients):
            # Calculate distances to other clients
            distances = []
            for j in range(num_clients):
                if i != j:
                    distance = self._calculate_weight_distance(
                        client_updates[i]['weights'],
                        client_updates[j]['weights']
                    )
                    distances.append(distance)
            
            # Sort distances and select closest clients
            distances.sort()
            krum_score = sum(distances[:num_clients - f - 1])
            krum_scores.append(krum_score)
        
        # Return client with minimum Krum score
        return krum_scores.index(min(krum_scores))
    
    def _calculate_weight_distance(self, weights1: Dict, weights2: Dict) -> float:
        """Calculate distance between two weight dictionaries"""
        total_distance = 0.0
        for key in weights1.keys():
            if key in weights2:
                distance = torch.norm(weights1[key] - weights2[key]).item()
                total_distance += distance
        return total_distance
    
    def evaluate_global_model(self, test_loader: PyGDataLoader) -> Dict[str, float]:
        """Evaluate global model on test data"""
        self.global_model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch_data in test_loader:
                if isinstance(batch_data, list):
                    batch, labels = batch_data
                else:
                    batch = batch_data
                    labels = batch.y
                
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                output = self.global_model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(output, labels)
                
                # Update metrics
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(test_loader)
        
        # Additional metrics
        try:
            from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
            
            precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
            recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
            f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
            
            results = {
                'accuracy': accuracy,
                'loss': avg_loss,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'correct': correct,
                'total': total
            }
            
        except ImportError:
            results = {
                'accuracy': accuracy,
                'loss': avg_loss,
                'correct': correct,
                'total': total
            }
        
        return results
    
    def run_federated_round(self, client_updates: List[Dict], test_data: PyGDataLoader) -> Dict:
        """Run one federated learning round"""
        self.round_number += 1
        
        # Aggregate updates
        aggregated_weights = self.aggregate_updates(client_updates)
        
        # Evaluate global model
        evaluation_results = self.evaluate_global_model(test_data)
        
        # Update metrics
        self.metrics_history['rounds'].append(self.round_number)
        self.metrics_history['global_accuracies'].append(evaluation_results['accuracy'])
        self.metrics_history['global_losses'].append(evaluation_results['loss'])
        self.metrics_history['client_participation'].append(len(client_updates))
        
        if self.privacy_enabled:
            privacy_budget = self.privacy_mechanism.get_privacy_budget()
            self.metrics_history['privacy_budgets'].append(privacy_budget)
        
        return evaluation_results
    
    def save_global_model(self, filepath: str):
        """Save global model"""
        torch.save({
            'model_state_dict': self.global_model.state_dict(),
            'global_weights': self.global_weights,
            'round_number': self.round_number,
            'metrics_history': self.metrics_history,
            'config': self.config
        }, filepath)
        
        logger.info(f"Global model saved to {filepath}")
    
    def load_global_model(self, filepath: str):
        """Load global model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.global_model.load_state_dict(checkpoint['model_state_dict'])
        self.global_weights = checkpoint['global_weights']
        self.round_number = checkpoint['round_number']
        self.metrics_history = checkpoint['metrics_history']
        
        logger.info(f"Global model loaded from {filepath}")


class FederatedClient:
    """
    Research-grade federated learning client
    
    Features:
    - Local training with privacy mechanisms
    - Comprehensive evaluation metrics
    - Research-grade logging
    """
    
    def __init__(self, client_id: int, local_data: PyGDataLoader, 
                 model: nn.Module, config: Dict, device: str = 'cuda'):
        """
        Initialize federated client
        
        Args:
            client_id: Unique client identifier
            local_data: Client's local data loader
            model: Local model instance
            config: Client configuration
            device: Device to run on
        """
        self.client_id = client_id
        self.local_data = local_data
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Training configuration
        self.learning_rate = float(config.get('training', {}).get('learning_rate', 0.001))
        self.weight_decay = float(config.get('training', {}).get('weight_decay', 1e-4))
        self.momentum = float(config.get('training', {}).get('momentum', 0.9))
        
        # Privacy mechanisms
        self.privacy_enabled = config.get('privacy', {}).get('enabled', False)
        if self.privacy_enabled:
            from core.privacy import DifferentialPrivacy
            self.privacy_mechanism = DifferentialPrivacy(config['privacy'])
        
        # Client metrics
        self.local_metrics = {
            'training_losses': [],
            'training_accuracies': [],
            'evaluation_losses': [],
            'evaluation_accuracies': [],
            'privacy_budgets': []
        }
        
        logger.info(f"Initialized FederatedClient {client_id}")
    
    def train_local_model(self, global_weights: Dict[str, torch.Tensor], 
                         num_epochs: int) -> Dict[str, Any]:
        """
        Train local model on client data
        
        Args:
            global_weights: Global model weights to initialize from
            num_epochs: Number of local training epochs
        
        Returns:
            Training results and updated weights
        """
        # Initialize with global weights
        self.model.set_weights(global_weights)
        
        # Setup optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Setup scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        # Training loop
        self.model.train()
        epoch_losses = []
        epoch_accuracies = []
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch_data in self.local_data:
                if isinstance(batch_data, list):
                    batch, labels = batch_data
                else:
                    batch = batch_data
                    labels = batch.y
                
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                output = self.model(batch.x, batch.edge_index, batch.batch)
                loss = nn.CrossEntropyLoss()(output, labels)
                
                # Backward pass
                loss.backward()
                
                # Apply privacy mechanisms
                if self.privacy_enabled:
                    self.privacy_mechanism.clip_gradients(self.model)
                
                optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # Calculate epoch metrics
            avg_loss = total_loss / len(self.local_data)
            accuracy = 100.0 * correct / total
            
            epoch_losses.append(avg_loss)
            epoch_accuracies.append(accuracy)
            
            # Update scheduler
            scheduler.step()
        
        # Record metrics
        self.local_metrics['training_losses'].extend(epoch_losses)
        self.local_metrics['training_accuracies'].extend(epoch_accuracies)
        
        # Get updated weights
        updated_weights = self.model.get_weights()
        
        # Calculate sample count
        sample_count = len(self.local_data.dataset)
        
        return {
            'weights': updated_weights,
            'sample_count': sample_count,
            'training_loss': epoch_losses[-1],
            'training_accuracy': epoch_accuracies[-1],
            'client_id': self.client_id
        }
    
    def evaluate_local_model(self, test_loader: PyGDataLoader) -> Dict[str, float]:
        """Evaluate local model"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data in test_loader:
                if isinstance(batch_data, list):
                    batch, labels = batch_data
                else:
                    batch = batch_data
                    labels = batch.y
                
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                
                output = self.model(batch.x, batch.edge_index, batch.batch)
                loss = nn.CrossEntropyLoss()(output, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(test_loader)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'correct': correct,
            'total': total
        }
