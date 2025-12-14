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
import math

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
        
        # GNN layers with residual connections and enhanced GAT
        self.gnn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_residual = True  # Enable residual connections for deeper networks
        self.gat_proj = None  # For GAT multi-head projection
        self.gnn_type = gnn_type  # Store for forward pass
        
        for i in range(num_layers):
            # GNN layer - Enhanced with 8 heads for GAT (research-grade)
            if gnn_type == 'gcn':
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == 'gat':
                # Use 8 heads for better representation (research-grade)
                num_heads = 8
                head_dim = hidden_dim // num_heads
                # Ensure head_dim is at least 1
                if head_dim < 1:
                    head_dim = 1
                    num_heads = hidden_dim
                self.gnn_layers.append(GATConv(hidden_dim, head_dim, heads=num_heads, concat=True, dropout=dropout))
                # Projection to maintain hidden_dim after concat
                if self.gat_proj is None:
                    self.gat_proj = nn.ModuleList()
                # Output dimension after concat is head_dim * num_heads
                actual_output_dim = head_dim * num_heads
                self.gat_proj.append(nn.Linear(actual_output_dim, hidden_dim))
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
        
        # Attention-based pooling (research-grade)
        if pooling == 'attention':
            self.attention_pool = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
            self.pooling_dim = hidden_dim
        else:
            self.attention_pool = None
            self.pooling_dim = self._get_pooling_dim()
        
        # Enhanced classifier with better capacity
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
        Forward pass through the GNN with residual connections
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]
        
        Returns:
            Graph-level predictions [batch_size, num_classes]
        """
        # Input projection
        x = self.input_proj(x)
        x_residual = x  # Store for residual connection
        
        # GNN layers with residual connections
        for i, (gnn_layer, norm) in enumerate(zip(self.gnn_layers, self.norms)):
            # GNN forward
            x_new = gnn_layer(x, edge_index)
            
            # Handle GAT multi-head output (concat=True) - project back to hidden_dim
            if self.gnn_type == 'gat' and self.gat_proj is not None and len(self.gat_proj) > i:
                if x_new.shape[1] != self.hidden_dim:
                    x_new = self.gat_proj[i](x_new)
            
            # Normalization
            x_new = norm(x_new)
            
            # Residual connection (if dimensions match and enabled)
            if self.use_residual and x_new.shape == x_residual.shape:
                x_new = x_new + x_residual
            
            # Activation
            if self.activation == 'relu':
                x_new = F.relu(x_new)
            elif self.activation == 'gelu':
                x_new = F.gelu(x_new)
            
            # Dropout
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            # Update for next iteration
            x_residual = x_new
            x = x_new
        
        # Global pooling (attention-based or standard)
        if self.attention_pool is not None:
            x_pooled = self._attention_pooling(x, batch)
        else:
            x_pooled = self._global_pooling(x, batch)
        
        # Classification
        output = self.classifier(x_pooled)
        
        return output
    
    def _attention_pooling(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Attention-based global pooling (research-grade) - efficient implementation"""
        # Compute attention scores
        attn_scores = self.attention_pool(x).squeeze(-1)  # [num_nodes]
        
        # Get unique batch IDs
        batch_ids = batch.unique(sorted=True)
        num_graphs = len(batch_ids)
        
        # Apply softmax per graph (stable implementation)
        x_pooled = torch.zeros(num_graphs, x.size(1), device=x.device, dtype=x.dtype)
        
        for i, graph_id in enumerate(batch_ids):
            mask = (batch == graph_id)
            if mask.sum() == 0:
                continue
            
            graph_nodes = x[mask]  # Nodes in this graph
            graph_scores = attn_scores[mask]  # Attention scores for this graph
            
            # Stable softmax
            max_score = graph_scores.max()
            exp_scores = torch.exp(graph_scores - max_score)
            sum_exp = exp_scores.sum()
            attn_weights = exp_scores / (sum_exp + 1e-8)
            
            # Weighted sum
            x_pooled[i] = (graph_nodes * attn_weights.unsqueeze(-1)).sum(dim=0)
        
        return x_pooled
    
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
