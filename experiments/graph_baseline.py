"""
Graph Baseline Training
=====================
Implements centralized training baseline for graph-based malware detection.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader as PyGDataLoader
import time
import logging
from typing import Dict, List, Tuple, Optional
import os
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphBaselineTrainer:
    """
    Centralized training baseline for graph-based malware detection
    
    This serves as a comparison point for federated learning performance.
    """
    
    def __init__(self, model: nn.Module, train_loader: PyGDataLoader, val_loader: PyGDataLoader, 
                 test_loader: PyGDataLoader, device: str = 'cpu', config: Optional[Dict] = None):
        """
        Initialize graph baseline trainer
        
        Args:
            model: GNN model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            device: Device to run training on
            config: Training configuration
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Default configuration
        self.config = config or {
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'num_epochs': 20,
            'scheduler': 'step',
            'step_size': 5,
            'gamma': 0.1
        }
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        if self.config['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config['step_size'],
                gamma=self.config['gamma']
            )
        else:
            self.scheduler = None
        
        # Training history
        self.training_history = {
            'train_losses': [],
            'train_accuracies': [],
            'val_losses': [],
            'val_accuracies': [],
            'learning_rates': [],
            'epoch_times': []
        }
        
        logger.info(f"Initialized GraphBaselineTrainer with {len(train_loader.dataset)} training samples")
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in self.train_loader:
            batch = batch.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(batch.x, batch.edge_index, batch.batch)
            loss = nn.CrossEntropyLoss()(output, batch.y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += batch.y.size(0)
            correct += (predicted == batch.y).sum().item()
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float]:
        """
        Validate the model
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                
                output = self.model(batch.x, batch.edge_index, batch.batch)
                loss = nn.CrossEntropyLoss()(output, batch.y)
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += batch.y.size(0)
                correct += (predicted == batch.y).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, num_epochs: Optional[int] = None) -> Dict:
        """
        Train the model for specified number of epochs
        
        Args:
            num_epochs: Number of epochs to train (uses config if None)
        
        Returns:
            Training history
        """
        if num_epochs is None:
            num_epochs = self.config['num_epochs']
        
        logger.info(f"Starting graph baseline training for {num_epochs} epochs")
        
        best_val_accuracy = 0.0
        best_model_state = None
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Train
            train_loss, train_accuracy = self.train_epoch()
            
            # Validate
            val_loss, val_accuracy = self.validate()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.config['learning_rate']
            
            # Record metrics
            epoch_time = time.time() - epoch_start_time
            self.training_history['train_losses'].append(train_loss)
            self.training_history['train_accuracies'].append(train_accuracy)
            self.training_history['val_losses'].append(val_loss)
            self.training_history['val_accuracies'].append(val_accuracy)
            self.training_history['learning_rates'].append(current_lr)
            self.training_history['epoch_times'].append(epoch_time)
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = self.model.state_dict().copy()
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, "
                       f"LR: {current_lr:.6f}, Time: {epoch_time:.2f}s")
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Loaded best model with validation accuracy: {best_val_accuracy:.2f}%")
        
        logger.info("Graph baseline training completed!")
        return self.training_history
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on test set
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(self.device)
                
                output = self.model(batch.x, batch.edge_index, batch.batch)
                loss = nn.CrossEntropyLoss()(output, batch.y)
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += batch.y.size(0)
                correct += (predicted == batch.y).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(batch.y.cpu().numpy())
        
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(self.test_loader)
        
        # Calculate additional metrics
        try:
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            precision = precision_score(all_targets, all_predictions, average='weighted')
            recall = recall_score(all_targets, all_predictions, average='weighted')
            f1 = f1_score(all_targets, all_predictions, average='weighted')
            
            results = {
                'test_loss': avg_loss,
                'test_accuracy': accuracy,
                'test_precision': precision,
                'test_recall': recall,
                'test_f1': f1,
                'correct': correct,
                'total': total
            }
            
        except ImportError:
            logger.warning("Scikit-learn not available for advanced metrics")
            results = {
                'test_loss': avg_loss,
                'test_accuracy': accuracy,
                'correct': correct,
                'total': total
            }
        
        logger.info(f"Test Results: Accuracy: {accuracy:.2f}%")
        
        return results
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Model loaded from {filepath}")


def run_graph_baseline_experiment(data_dir: str, config_path: Optional[str] = None) -> Dict:
    """
    Run graph baseline centralized training experiment
    
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
    from data.graph_dataset import create_graph_data_loaders
    from models.gnn import create_gnn_model
    
    # Create data loaders
    logger.info("Creating graph data loaders...")
    train_loader, val_loader, test_loader = create_graph_data_loaders(
        data_dir=data_dir,
        batch_size=config.get('dataset', {}).get('batch_size', 32),
        max_nodes=config.get('dataset', {}).get('max_nodes', 5000),
        num_workers=config.get('dataset', {}).get('num_workers', 2)
    )
    
    # Create model
    logger.info("Creating GNN model...")
    model = create_gnn_model(
        model_type=config.get('model', {}).get('gnn_type', 'gcn'),
        num_classes=config.get('model', {}).get('num_classes', 5),
        hidden_dim=config.get('model', {}).get('hidden_dim', 64),
        num_layers=config.get('model', {}).get('num_layers', 3),
        dropout=config.get('model', {}).get('dropout', 0.5)
    )
    
    # Create trainer
    trainer = GraphBaselineTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        config=config.get('training', {})
    )
    
    # Train model
    logger.info("Starting graph baseline training...")
    training_history = trainer.train()
    
    # Evaluate model
    logger.info("Evaluating model...")
    test_results = trainer.evaluate()
    
    # Save model
    if config.get('evaluation', {}).get('save_model', True):
        os.makedirs('models', exist_ok=True)
        trainer.save_model('models/graph_baseline_model.pth')
    
    # Compile results
    results = {
        'training_history': training_history,
        'test_results': test_results,
        'config': config,
        'model_info': {
            'num_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        }
    }
    
    logger.info("Graph baseline experiment completed!")
    return results


if __name__ == "__main__":
    # Test graph baseline training
    print("Testing graph baseline training...")
    
    data_dir = "malnet-graphs-tiny"
    config_path = "config.yaml"
    
    if os.path.exists(data_dir):
        results = run_graph_baseline_experiment(data_dir, config_path)
        
        print(f"\nGraph Baseline Results:")
        print(f"  Final Test Accuracy: {results['test_results']['test_accuracy']:.2f}%")
        print(f"  Model Parameters: {results['model_info']['num_parameters']:,}")
        print(f"  Model Size: {results['model_info']['model_size_mb']:.2f} MB")
    else:
        print(f"Dataset directory {data_dir} not found!")
