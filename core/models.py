"""
Research-Grade Graph Neural Network Models
=========================================
Professional GNN architectures for malware detection research.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import BatchNorm, LayerNorm
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ResearchGNN(nn.Module):
    """
    Research-grade GNN for malware detection
    
    Features:
    - Multiple GNN layer types (GCN, GAT, SAGE)
    - Advanced pooling strategies
    - Batch normalization and dropout
    - Configurable architecture
    """
    
    def __init__(self, 
                 input_dim: int = 3,
                 num_classes: int = 5,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 gnn_type: str = 'gcn',
                 dropout: float = 0.3,
                 activation: str = 'relu',
                 normalization: str = 'batch',
                 pooling: str = 'mean_max'):
        """
        Initialize research GNN
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            hidden_dim: Hidden dimension size
            num_layers: Number of GNN layers
            gnn_type: Type of GNN layer ('gcn', 'gat', 'sage')
            dropout: Dropout rate
            activation: Activation function
            normalization: Normalization type ('batch', 'layer', 'none')
            pooling: Pooling strategy ('mean', 'max', 'add', 'mean_max', 'all')
        """
        super(ResearchGNN, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.dropout = dropout
        self.activation = activation
        self.normalization = normalization
        self.pooling = pooling
        
        # Input projection (adaptive to input dimension)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            # GNN layer
            if gnn_type == 'gcn':
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == 'gat':
                self.gnn_layers.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            elif gnn_type == 'sage':
                self.gnn_layers.append(SAGEConv(hidden_dim, hidden_dim))
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")
            
            # Normalization
            if normalization == 'batch':
                self.norms.append(BatchNorm(hidden_dim))
            elif normalization == 'layer':
                self.norms.append(LayerNorm(hidden_dim))
            else:
                self.norms.append(nn.Identity())
        
        # Global pooling
        self.pooling_dim = self._get_pooling_dim()
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.pooling_dim, hidden_dim),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        logger.info(f"Initialized ResearchGNN: {gnn_type.upper()}, "
                   f"{num_layers} layers, {hidden_dim} hidden dim")
    
    def _get_pooling_dim(self) -> int:
        """Calculate pooling dimension"""
        if self.pooling == 'all':
            return self.hidden_dim * 3  # mean + max + add
        elif self.pooling in ['mean_max', 'max_mean']:
            return self.hidden_dim * 2  # mean + max
        else:
            return self.hidden_dim
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GNN
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]
        
        Returns:
            Graph-level predictions [batch_size, num_classes]
        """
        # Input projection
        x = self.input_proj(x)
        
        # GNN layers
        for i, (gnn_layer, norm) in enumerate(zip(self.gnn_layers, self.norms)):
            # GNN forward
            x = gnn_layer(x, edge_index)
            
            # Normalization
            x = norm(x)
            
            # Activation
            if self.activation == 'relu':
                x = F.relu(x)
            elif self.activation == 'gelu':
                x = F.gelu(x)
            
            # Dropout
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x_pooled = self._global_pooling(x, batch)
        
        # Classification
        output = self.classifier(x_pooled)
        
        return output
    
    def _global_pooling(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Apply global pooling strategy"""
        if self.pooling == 'mean':
            return global_mean_pool(x, batch)
        elif self.pooling == 'max':
            return global_max_pool(x, batch)
        elif self.pooling == 'add':
            return global_add_pool(x, batch)
        elif self.pooling in ['mean_max', 'max_mean']:
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            return torch.cat([x_mean, x_max], dim=1)
        elif self.pooling == 'all':
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x_add = global_add_pool(x, batch)
            return torch.cat([x_mean, x_max, x_add], dim=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
    
    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Get model weights as dictionary"""
        return {name: param.clone().detach() for name, param in self.named_parameters()}
    
    def set_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Set model weights from dictionary with type safety"""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in weights:
                    # Ensure weight has same dtype as parameter
                    weight_tensor = weights[name].to(dtype=param.dtype, device=param.device)
                    param.copy_(weight_tensor)
                else:
                    logger.warning(f"Weight {name} not found in provided weights")
    
    def count_parameters(self) -> int:
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size(self) -> float:
        """Get model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / 1024**2


class LightweightGNN(nn.Module):
    """
    Lightweight GNN for resource-constrained environments
    
    Optimized for:
    - Fast training and inference
    - Low memory usage
    - Mobile deployment
    """
    
    def __init__(self, num_classes: int = 5, hidden_dim: int = 64):
        super(LightweightGNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Simple GNN layers
        self.conv1 = GCNConv(5, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Lightweight classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # mean + max pooling
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
        logger.info(f"Initialized LightweightGNN with {num_classes} classes")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass through lightweight GNN"""
        # GNN layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        
        # Global pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_pooled = torch.cat([x_mean, x_max], dim=1)
        
        # Classification
        output = self.classifier(x_pooled)
        
        return output
    
    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Get model weights with dtype preservation"""
        return {name: param.clone().detach().to(dtype=param.dtype) for name, param in self.named_parameters()}
    
    def set_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Set model weights with type safety"""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in weights:
                    weight_tensor = weights[name].to(dtype=param.dtype, device=param.device)
                    param.copy_(weight_tensor)
    
    def count_parameters(self) -> int:
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size(self) -> float:
        """Get model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / 1024**2


def create_model(config: Dict) -> nn.Module:
    """
    Create GNN model based on configuration
    
    Args:
        config: Model configuration dictionary
    
    Returns:
        GNN model instance
    """
    model_config = config['model']
    model_type = model_config.get('gnn_type', 'gcn')
    
    if model_type == 'lightweight':
        return LightweightGNN(
            num_classes=model_config.get('num_classes', 5),
            hidden_dim=model_config.get('hidden_dim', 64)
        )
    else:
        return ResearchGNN(
            num_classes=model_config.get('num_classes', 5),
            hidden_dim=model_config.get('hidden_dim', 128),
            num_layers=model_config.get('num_layers', 4),
            gnn_type=model_type,
            dropout=model_config.get('dropout', 0.3),
            activation=model_config.get('activation', 'relu'),
            normalization=model_config.get('normalization', 'batch'),
            pooling=model_config.get('pooling', 'mean_max')
        )
