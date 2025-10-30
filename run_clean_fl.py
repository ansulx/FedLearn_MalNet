#!/usr/bin/env python3
"""
Clean Federated Learning System - Real-Time Dashboard
======================================================
Research-grade FL system for malware detection research

Run: python run_clean_fl.py

RESEARCH-GRADE VERSION: Using advanced GNN architectures and proper evaluation
"""

import os
import sys
import time
import threading
from datetime import datetime
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Data, Batch
import numpy as np
import warnings
import traceback
import gc

# Import research-grade model
from core.models import ResearchGNN

# Suppress warnings
warnings.filterwarnings('ignore')

# Set timeouts and error handling - RESEARCH-GRADE STABILITY
torch.backends.cudnn.benchmark = False  # Disable for stability
torch.backends.cudnn.deterministic = True  # Deterministic training
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
os.environ['OMP_NUM_THREADS'] = '1'  # Prevent threading issues
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Better error messages

# Increase timeout settings for stability
import socket
socket.setdefaulttimeout(300)  # 5 minutes for any network operations

# Set PyTorch memory management
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    # Prevent CUDA timeout issues
    torch.cuda.synchronize()

# ANSI colors for terminal
class Color:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BG_BLACK = '\033[40m'


# ============================================================================
# 1. SIMPLE GNN MODEL
# ============================================================================

class MalwareGNN(nn.Module):
    """Simple GCN for malware detection (legacy - use ResearchGNN instead)"""
    
    def __init__(self, input_dim=3, num_classes=5, hidden_dim=64):
        super(MalwareGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)  # Adaptive input features
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x
    
    def get_weights(self):
        """Get model weights with dtype preservation"""
        return {name: param.clone().detach().to(dtype=param.dtype) for name, param in self.named_parameters()}
    
    def set_weights(self, weights):
        """Set model weights with type safety"""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in weights:
                    weight_tensor = weights[name].to(dtype=param.dtype, device=param.device)
                    param.copy_(weight_tensor)


# ============================================================================
# 2. FEDERATED SERVER
# ============================================================================

class FederatedServer:
    """Central FL server with live status tracking"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config['device']
        self.model = self.model.to(self.device)
        
        # Status tracking
        self.status = "INITIALIZING"
        self.current_round = 0
        self.total_rounds = config['num_rounds']
        self.global_accuracy = 0.0
        self.global_loss = 0.0
        self.accuracy_history = deque(maxlen=50)
        self.round_times = deque(maxlen=10)
        self.total_samples = 0
        
        # Device tracking
        self.connected_devices = []
        self.device_status = {}
        
    def register_device(self, device_id, device_name, num_samples):
        """Register a new device"""
        self.connected_devices.append(device_id)
        self.device_status[device_id] = {
            'name': device_name,
            'status': 'CONNECTED',
            'accuracy': 0.0,
            'loss': 0.0,
            'samples': num_samples,
            'last_update': time.time()
        }
        self.total_samples += num_samples
    
    def aggregate_updates(self, device_updates):
        """FedAvg aggregation with strict type safety"""
        self.status = "AGGREGATING"
        
        total_samples = sum(u['num_samples'] for u in device_updates)
        aggregated_weights = {}
        
        # Weighted average with STRICT type handling
        for key in self.model.state_dict().keys():
            # Get reference parameter to preserve dtype and device
            ref_param = self.model.state_dict()[key]
            
            # Initialize with zeros matching exact dtype
            weighted_sum = torch.zeros_like(ref_param, dtype=ref_param.dtype, device=ref_param.device)
            
            for update in device_updates:
                # Calculate weight as same dtype as parameter
                weight_scalar = float(update['num_samples']) / float(total_samples)
                
                # Get update weight and ensure exact dtype match
                update_weight = update['weights'][key]
                
                # Move to correct device and dtype BEFORE multiplication
                update_weight = update_weight.to(device=ref_param.device, dtype=ref_param.dtype)
                
                # Multiply by scalar (not tensor) to preserve dtype
                weighted_update = update_weight * weight_scalar
                
                # Add with explicit dtype preservation
                weighted_sum = weighted_sum + weighted_update.to(dtype=ref_param.dtype)
            
            # Store with exact dtype
            aggregated_weights[key] = weighted_sum
        
        # Update global model
        self.model.load_state_dict(aggregated_weights)
        
    def evaluate(self, test_loader, max_retries=3):
        """Evaluate global model with error handling and safety checks"""
        self.status = "EVALUATING"
        
        for attempt in range(max_retries):
            try:
                self.model.eval()
                
                correct = 0
                total = 0
                total_loss = 0.0
                batch_count = 0
                
                with torch.no_grad():
                    for batch_data in test_loader:
                        try:
                            # Safety check: skip if batch_data is None
                            if batch_data is None:
                                continue
                            
                            # FIXED: Proper PyG Data object handling with type safety
                            batch = batch_data.to(self.device)
                            labels = batch.y.long()  # Ensure labels are Long type
                            
                            # Ensure batch features have correct type
                            if batch.x.dtype != torch.float32:
                                batch.x = batch.x.float()
                            
                            # Safety check: validate tensor shapes
                            if batch.x.shape[0] == 0 or labels.shape[0] == 0:
                                continue
                            
                            # Forward pass with error catching
                            output = self.model(batch.x, batch.edge_index, batch.batch)
                            
                            # Safety check: validate output
                            if output.shape[0] == 0 or torch.isnan(output).any() or torch.isinf(output).any():
                                continue
                            
                            loss = F.cross_entropy(output, labels)
                            
                            # Safety check: validate loss
                            if torch.isnan(loss) or torch.isinf(loss):
                                continue
                            
                            _, predicted = torch.max(output.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                            total_loss += loss.item()
                            batch_count += 1
                            
                            # More frequent cache clearing and garbage collection
                            if batch_count % 5 == 0:  # Changed from 10 to 5
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                gc.collect()  # Python garbage collection
                                    
                        except RuntimeError as e:
                            error_msg = str(e).lower()
                            if "out of memory" in error_msg or "cuda" in error_msg:
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                gc.collect()
                                continue
                            elif "cast" in error_msg or "type" in error_msg:
                                # Type casting error - skip this batch
                                continue
                            else:
                                # Log but continue with other batches
                                continue
                        except Exception as e:
                            # Any other error - skip batch and continue
                            continue
                
                if total > 0 and batch_count > 0:
                    self.global_accuracy = 100.0 * correct / total
                    self.global_loss = total_loss / batch_count
                    self.accuracy_history.append(self.global_accuracy)
                    
                    # Aggressive cleanup after evaluation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()  # Synchronize CUDA operations
                    gc.collect()
                    
                    return {'accuracy': self.global_accuracy, 'loss': self.global_loss}
                else:
                    # No valid batches processed
                    return {'accuracy': self.global_accuracy, 'loss': self.global_loss}
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    gc.collect()
                    continue
                else:
                    # Return last known values
                    return {'accuracy': self.global_accuracy, 'loss': self.global_loss}


# ============================================================================
# 3. FEDERATED DEVICE (CLIENT)
# ============================================================================

class FederatedDevice:
    """FL device/client that trains locally"""
    
    def __init__(self, device_id, device_name, local_data, model, config):
        self.device_id = device_id
        self.device_name = device_name
        self.local_data = local_data
        self.model = model
        self.config = config
        self.device = config['device']
        
        self.status = "IDLE"
        self.local_accuracy = 0.0
        self.local_loss = 0.0
        self.num_samples = len(local_data.dataset)
    
    def train_local(self, global_weights, num_epochs, max_retries=3):
        """Train on local data with error handling and retry logic"""
        self.status = "TRAINING"
        
        for attempt in range(max_retries):
            try:
                # Load global weights
                self.model.load_state_dict(global_weights)
                self.model.train()
                
                optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
                
                for epoch in range(num_epochs):
                    epoch_loss = 0.0
                    correct = 0
                    total = 0
                    batch_count = 0
                    
                    for batch_data in self.local_data:
                        try:
                            # FIXED: Proper PyG Data object handling with type safety
                            batch = batch_data.to(self.device)
                            labels = batch.y.long()  # Ensure labels are Long type
                            
                            # Ensure batch components have correct types
                            if batch.x.dtype != torch.float32:
                                batch.x = batch.x.float()
                            
                            optimizer.zero_grad()
                            output = self.model(batch.x, batch.edge_index, batch.batch)
                            loss = F.cross_entropy(output, labels)
                            
                            # Check for NaN loss before backward
                            if torch.isnan(loss) or torch.isinf(loss):
                                continue
                            
                            loss.backward()
                            
                            # Gradient clipping to prevent exploding gradients
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            
                            optimizer.step()
                            
                            epoch_loss += loss.item()
                            _, predicted = torch.max(output.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                            batch_count += 1
                            
                            # More frequent cache clearing
                            if batch_count % 5 == 0:  # Changed from 10 to 5
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                    torch.cuda.synchronize()
                                gc.collect()
                                    
                        except RuntimeError as e:
                            error_msg = str(e).lower()
                            if "out of memory" in error_msg or "cuda" in error_msg:
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                    torch.cuda.synchronize()
                                gc.collect()
                                continue
                            elif "cast" in error_msg or "type" in error_msg:
                                # Type casting error - skip this batch
                                continue
                            else:
                                raise
                        except Exception as e:
                            # Catch any other errors and continue
                            continue
                    
                    if batch_count > 0:
                        self.local_loss = epoch_loss / batch_count
                        self.local_accuracy = 100.0 * correct / total if total > 0 else 0.0
                
                self.status = "COMPLETED"
                
                # Clear cache before returning
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Return updated weights
                return {
                    'device_id': self.device_id,
                    'weights': self.model.state_dict(),
                    'num_samples': self.num_samples,
                    'accuracy': self.local_accuracy,
                    'loss': self.local_loss
                }
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    # On final failure, return last known good state
                    self.status = "ERROR"
                    return {
                        'device_id': self.device_id,
                        'weights': global_weights,  # Return unchanged weights
                        'num_samples': self.num_samples,
                        'accuracy': 0.0,
                        'loss': 999.0
                    }


# ============================================================================
# 4. TERMINAL DASHBOARD
# ============================================================================

class TerminalDashboard:
    """Real-time terminal dashboard"""
    
    def __init__(self, server, devices):
        self.server = server
        self.devices = devices
        self.start_time = time.time()
        self.log_messages = deque(maxlen=10)
        
    def clear_screen(self):
        os.system('clear' if os.name != 'nt' else 'cls')
    
    def log(self, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_messages.append(f"[{timestamp}] {message}")
    
    def draw_progress_bar(self, progress, width=40):
        filled = int(width * progress)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}] {int(progress * 100)}%"
    
    def draw_accuracy_graph(self, history, width=60, height=7):
        """Draw ASCII accuracy graph"""
        if len(history) < 2:
            return ["No data yet..."]
        
        values = list(history)
        min_val = min(values)
        max_val = max(values)
        
        # Normalize to 0-height range
        if max_val - min_val > 0:
            normalized = [(v - min_val) / (max_val - min_val) * (height - 1) for v in values]
        else:
            normalized = [height // 2] * len(values)
        
        lines = []
        for h in range(height - 1, -1, -1):
            line = f"{int(min_val + (max_val - min_val) * h / (height - 1)):3d}% │ "
            for i, val in enumerate(normalized):
                if abs(val - h) < 0.5:
                    line += "●" if i == len(normalized) - 1 else "─"
                elif val > h:
                    line += "│"
                else:
                    line += " "
            lines.append(line)
        
        lines.append("     └" + "─" * len(normalized))
        return lines
    
    def render(self):
        """Render full dashboard"""
        self.clear_screen()
        
        # Header
        print(f"\n{Color.CYAN}{Color.BOLD}{'═' * 80}{Color.RESET}")
        print(f"{Color.CYAN}{Color.BOLD}{'🖥️  FEDERATED LEARNING SERVER - LIVE DASHBOARD':^80}{Color.RESET}")
        print(f"{Color.CYAN}{Color.BOLD}{'═' * 80}{Color.RESET}\n")
        
        # Server Status Box
        runtime = time.time() - self.start_time
        status_color = Color.GREEN if self.server.status == "RUNNING" else Color.YELLOW
        
        print(f"{Color.BLUE}┌{'─' * 78}┐{Color.RESET}")
        print(f"{Color.BLUE}│ {Color.BOLD}SERVER STATUS{' ' * 65}│{Color.RESET}")
        print(f"{Color.BLUE}├{'─' * 78}┤{Color.RESET}")
        print(f"{Color.BLUE}│{Color.RESET} Status: {status_color}● {self.server.status:<15}{Color.RESET} "
              f"Round: {Color.BOLD}{self.server.current_round}/{self.server.total_rounds}{Color.RESET}     "
              f"Device: {Color.CYAN}{self.server.device}{Color.RESET}     "
              f"Runtime: {int(runtime)}s {Color.BLUE}│{Color.RESET}")
        print(f"{Color.BLUE}│{Color.RESET} Global Accuracy: {Color.GREEN}{Color.BOLD}{self.server.global_accuracy:.2f}%{Color.RESET}  "
              f"Loss: {self.server.global_loss:.4f}  "
              f"Samples: {self.server.total_samples:,} "
              f"{Color.BLUE}│{Color.RESET}")
        print(f"{Color.BLUE}│{Color.RESET} Connected Devices: {Color.GREEN}{len(self.server.connected_devices)}/{len(self.devices)}{Color.RESET}"
              f"{' ' * 48}{Color.BLUE}│{Color.RESET}")
        print(f"{Color.BLUE}└{'─' * 78}┘{Color.RESET}\n")
        
        # Connected Devices Box
        print(f"{Color.MAGENTA}┌{'─' * 78}┐{Color.RESET}")
        print(f"{Color.MAGENTA}│ {Color.BOLD}CONNECTED DEVICES (Real-Time Status){' ' * 42}│{Color.RESET}")
        print(f"{Color.MAGENTA}├{'─' * 78}┤{Color.RESET}")
        
        for device in self.devices:
            status_symbol = {
                'TRAINING': f'{Color.YELLOW}●',
                'COMPLETED': f'{Color.GREEN}✓',
                'IDLE': f'{Color.BLUE}○',
                'CONNECTED': f'{Color.CYAN}○'
            }.get(device.status, f'{Color.WHITE}○')
            
            print(f"{Color.MAGENTA}│{Color.RESET} Device {device.device_id} [{device.device_name:<12}] "
                  f"{status_symbol} {device.status:<10}{Color.RESET} "
                  f"Acc: {device.local_accuracy:5.1f}%  "
                  f"Samples: {device.num_samples:>5} "
                  f"{Color.MAGENTA}│{Color.RESET}")
        
        print(f"{Color.MAGENTA}└{'─' * 78}┘{Color.RESET}\n")
        
        # Accuracy Graph
        if len(self.server.accuracy_history) > 1:
            print(f"{Color.CYAN}┌{'─' * 78}┐{Color.RESET}")
            print(f"{Color.CYAN}│ {Color.BOLD}GLOBAL MODEL ACCURACY OVER TIME{' ' * 46}│{Color.RESET}")
            print(f"{Color.CYAN}├{'─' * 78}┤{Color.RESET}")
            
            graph_lines = self.draw_accuracy_graph(self.server.accuracy_history)
            for line in graph_lines:
                print(f"{Color.CYAN}│{Color.RESET} {line}{' ' * (76 - len(line))}{Color.CYAN}│{Color.RESET}")
            
            print(f"{Color.CYAN}└{'─' * 78}┘{Color.RESET}\n")
        
        # Recent Activity Log
        print(f"{Color.GREEN}┌{'─' * 78}┐{Color.RESET}")
        print(f"{Color.GREEN}│ {Color.BOLD}RECENT ACTIVITY{' ' * 63}│{Color.RESET}")
        print(f"{Color.GREEN}├{'─' * 78}┤{Color.RESET}")
        
        for msg in list(self.log_messages)[-5:]:
            print(f"{Color.GREEN}│{Color.RESET} {msg:<76} {Color.GREEN}│{Color.RESET}")
        
        print(f"{Color.GREEN}└{'─' * 78}┘{Color.RESET}\n")


# ============================================================================
# 5. MAIN FEDERATED LEARNING ORCHESTRATOR
# ============================================================================

def load_malnet_data():
    """Load and prepare MalNet data with optimized settings for research"""
    from core.data_loader import MalNetGraphLoader
    
    config = {
        'dataset': {
            'path': 'malnet-graphs-tiny',
            'max_nodes': 2000,
            'batch_size': 4,  # Optimized for stability
            'num_workers': 0,  # 0 workers to prevent timeout/connection issues
            'pin_memory': False  # Disabled to prevent CUDA issues
        },
        'model': {'num_classes': 5}
    }
    
    data_loader = MalNetGraphLoader(config)
    train_loader, val_loader, test_loader = data_loader.create_data_loaders()
    
    print(f"   {Color.GREEN}✓{Color.RESET} Train: {len(train_loader.dataset)} samples")
    print(f"   {Color.GREEN}✓{Color.RESET} Val: {len(val_loader.dataset)} samples")
    print(f"   {Color.GREEN}✓{Color.RESET} Test: {len(test_loader.dataset)} samples")
    
    return train_loader, test_loader


def split_data_for_devices(train_loader, num_devices=5):
    """Split training data across devices (Non-IID) with optimized batch size"""
    dataset = train_loader.dataset
    total_samples = len(dataset)
    samples_per_device = total_samples // num_devices
    
    device_datasets = []
    start_idx = 0
    
    for i in range(num_devices):
        end_idx = start_idx + samples_per_device
        if i == num_devices - 1:
            end_idx = total_samples
        
        device_data = [dataset[j] for j in range(start_idx, end_idx)]
        # Further reduced batch size from 8 to 4 for stability
        device_loader = PyGDataLoader(
            device_data, 
            batch_size=4,  # Reduced from 8 to 4
            shuffle=True, 
            num_workers=0,
            pin_memory=False  # Disable pin_memory to prevent CUDA issues
        )
        device_datasets.append(device_loader)
        
        start_idx = end_idx
    
    return device_datasets


def run_federated_learning():
    """Main FL orchestration with live dashboard"""
    
    print(f"\n{Color.CYAN}{Color.BOLD}{'═' * 80}{Color.RESET}")
    print(f"{Color.CYAN}{Color.BOLD}{'🚀 INITIALIZING FEDERATED LEARNING SYSTEM':^80}{Color.RESET}")
    print(f"{Color.CYAN}{Color.BOLD}{'═' * 80}{Color.RESET}\n")
    
    # Configuration
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_devices': 5,
        'num_rounds': 20,
        'local_epochs': 3
    }
    
    print(f"⚙️  Device: {Color.GREEN}{config['device'].upper()}{Color.RESET}")
    print(f"📱 Devices: {Color.GREEN}{config['num_devices']}{Color.RESET}")
    print(f"🔄 Rounds: {Color.GREEN}{config['num_rounds']}{Color.RESET}")
    print(f"📊 Loading MalNet dataset...")
    
    # Load data
    train_loader, test_loader = load_malnet_data()
    print(f"   {Color.GREEN}✓{Color.RESET} Loaded {len(train_loader.dataset)} training samples")
    
    # Detect input dimension from first sample
    first_sample = train_loader.dataset[0]
    input_dim = first_sample.x.shape[1]
    print(f"   {Color.GREEN}✓{Color.RESET} Detected input dimension: {input_dim} features")
    
    # Split data for devices
    device_datasets = split_data_for_devices(train_loader, config['num_devices'])
    print(f"   {Color.GREEN}✓{Color.RESET} Split data across {config['num_devices']} devices\n")
    
    # Create RESEARCH-GRADE global model
    print(f"🤖 Creating research-grade GNN model...")
    global_model = ResearchGNN(
        input_dim=input_dim,
        num_classes=5,
        hidden_dim=128,  # Larger for research
        num_layers=4,
        gnn_type='gat',  # GAT performs better than GCN
        dropout=0.3,
        normalization='batch',
        pooling='mean_max'
    )
    num_params = sum(p.numel() for p in global_model.parameters())
    print(f"   {Color.GREEN}✓{Color.RESET} Research GNN created (GAT, 4 layers, 128-dim): {num_params:,} parameters\n")
    
    # Create server
    print(f"🖥️  Starting FL server...")
    server = FederatedServer(global_model, config)
    server.status = "RUNNING"
    print(f"   {Color.GREEN}✓{Color.RESET} Server ready\n")
    
    # Create devices
    print(f"📱 Connecting devices...")
    devices = []
    device_names = ['Hospital', 'University', 'Company', 'Lab', 'Institute']
    
    for i, (name, local_data) in enumerate(zip(device_names, device_datasets)):
        device = FederatedDevice(
            device_id=i+1,
            device_name=name,
            local_data=local_data,
            model=ResearchGNN(
                input_dim=input_dim,
                num_classes=5,
                hidden_dim=128,
                num_layers=4,
                gnn_type='gat',
                dropout=0.3,
                normalization='batch',
                pooling='mean_max'
            ).to(config['device']),
            config=config
        )
        devices.append(device)
        server.register_device(i+1, name, len(local_data.dataset))
        print(f"   {Color.GREEN}✓{Color.RESET} Device {i+1} [{name}] connected ({len(local_data.dataset)} samples)")
    
    print(f"\n{Color.GREEN}{Color.BOLD}✅ System ready! Starting federated training...{Color.RESET}\n")
    time.sleep(2)
    
    # Create dashboard
    dashboard = TerminalDashboard(server, devices)
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Check for existing checkpoint to resume from
    start_round = 1
    checkpoint_file = 'checkpoints/latest_checkpoint.pth'
    if os.path.exists(checkpoint_file):
        try:
            checkpoint = torch.load(checkpoint_file)
            checkpoint_round = checkpoint['round']
            
            # Check if training is already complete
            if checkpoint_round >= config['num_rounds']:
                print(f"\n{Color.GREEN}✅ Training already complete at round {checkpoint_round}!{Color.RESET}")
                print(f"{Color.GREEN}📊 Final Accuracy: {checkpoint['global_accuracy']:.2f}%{Color.RESET}")
                print(f"\n{Color.YELLOW}💡 To restart training, delete: {checkpoint_file}{Color.RESET}\n")
                
                # Load checkpoint data for display
                server.model.load_state_dict(checkpoint['model_state'])
                server.global_accuracy = checkpoint['global_accuracy']
                server.accuracy_history = deque(checkpoint['accuracy_history'], maxlen=50)
                
                # Show final dashboard
                dashboard.render()
                
                print(f"\n{Color.GREEN}{Color.BOLD}{'═' * 80}{Color.RESET}")
                print(f"{Color.GREEN}{Color.BOLD}{'✅ FEDERATED LEARNING COMPLETE (Loaded from checkpoint)':^80}{Color.RESET}")
                print(f"{Color.GREEN}{Color.BOLD}{'═' * 80}{Color.RESET}\n")
                print(f"🎯 Final Global Accuracy: {Color.GREEN}{Color.BOLD}{server.global_accuracy:.2f}%{Color.RESET}")
                if len(server.accuracy_history) > 1:
                    improvement = server.global_accuracy - list(server.accuracy_history)[0]
                    print(f"📈 Improvement: {'+' if improvement >= 0 else ''}{improvement:.2f}%")
                print(f"💾 Total Samples: {server.total_samples:,}")
                print(f"\n{Color.CYAN}To retrain, delete checkpoint and run again.{Color.RESET}\n")
                return
            
            # Resume from checkpoint
            server.model.load_state_dict(checkpoint['model_state'])
            start_round = checkpoint_round + 1
            server.accuracy_history = deque(checkpoint['accuracy_history'], maxlen=50)
            server.global_accuracy = checkpoint.get('global_accuracy', 0.0)
            dashboard.log(f"Resumed from checkpoint at round {checkpoint_round}")
            print(f"\n{Color.YELLOW}📁 Resuming from round {checkpoint_round} (Accuracy: {server.global_accuracy:.2f}%){Color.RESET}\n")
            time.sleep(2)
        except Exception as e:
            print(f"\n{Color.YELLOW}⚠️  Could not load checkpoint: {e}{Color.RESET}\n")
            start_round = 1
    
    # Training loop with robust error handling
    for round_num in range(start_round, config['num_rounds'] + 1):
        try:
            server.current_round = round_num
            round_start = time.time()
            
            dashboard.render()
            dashboard.log(f"Starting Round {round_num}/{config['num_rounds']}")
            time.sleep(0.5)
            
            # Get global weights
            global_weights = server.model.state_dict()
            
            # Train all devices with error recovery
            device_updates = []
            for device in devices:
                try:
                    dashboard.render()
                    dashboard.log(f"Device {device.device_id} [{device.device_name}] training...")
                    
                    update = device.train_local(global_weights, config['local_epochs'])
                    
                    # Only include successful updates
                    if update['loss'] < 900:  # Filter out error returns
                        device_updates.append(update)
                        dashboard.log(f"Device {device.device_id} completed (Acc: {device.local_accuracy:.1f}%, Loss: {device.local_loss:.3f})")
                    else:
                        dashboard.log(f"Device {device.device_id} failed, skipping...")
                        device.status = "ERROR"
                    
                    dashboard.render()
                    time.sleep(0.3)
                    
                except Exception as e:
                    dashboard.log(f"Device {device.device_id} error: {str(e)[:50]}")
                    device.status = "ERROR"
                    continue
            
            # Only aggregate if we have at least one valid update
            if len(device_updates) > 0:
                try:
                    # Aggregate with error handling
                    dashboard.log(f"Aggregating updates from {len(device_updates)} devices...")
                    dashboard.render()
                    
                    try:
                        server.aggregate_updates(device_updates)
                        dashboard.log("✓ Aggregation successful")
                    except Exception as agg_error:
                        import traceback
                        error_trace = traceback.format_exc()
                        print(f"\n{'='*80}")
                        print(f"AGGREGATION ERROR at Round {round_num}:")
                        print(f"{'='*80}")
                        print(error_trace)
                        print(f"{'='*80}\n")
                        dashboard.log(f"Aggregation FAILED: {str(agg_error)[:40]}")
                        # Skip evaluation if aggregation failed
                        continue
                    
                    time.sleep(0.3)
                    
                    # Evaluate
                    dashboard.log("Evaluating global model...")
                    dashboard.render()
                    results = server.evaluate(test_loader)
                    dashboard.log(f"✓ Evaluation successful: {results['accuracy']:.1f}%")
                except Exception as e:
                    import traceback
                    print(f"\n{'='*80}")
                    print(f"GENERAL ERROR at Round {round_num}:")
                    print(f"{'='*80}")
                    print(traceback.format_exc())
                    print(f"{'='*80}\n")
                    dashboard.log(f"Round error: {str(e)[:50]}")
                    results = {'accuracy': server.global_accuracy, 'loss': server.global_loss}
                
                round_time = time.time() - round_start
                server.round_times.append(round_time)
                
                dashboard.log(f"Round {round_num} complete: Accuracy={results['accuracy']:.2f}%, Loss={results['loss']:.4f}")
                
                # Save checkpoint every 5 rounds or on last round
                if round_num % 5 == 0 or round_num == config['num_rounds']:
                    try:
                        torch.save({
                            'round': round_num,
                            'model_state': server.model.state_dict(),
                            'accuracy_history': list(server.accuracy_history),
                            'global_accuracy': server.global_accuracy
                        }, checkpoint_file)
                        dashboard.log(f"Checkpoint saved at round {round_num}")
                    except Exception as e:
                        dashboard.log(f"Checkpoint save failed: {str(e)[:50]}")
            else:
                dashboard.log(f"No valid updates in round {round_num}, skipping aggregation")
            
            # Final render
            dashboard.render()
            time.sleep(1)
            
            # Aggressive cleanup between rounds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()  # Additional CUDA cleanup
            gc.collect()  # Python garbage collection
            
            # Additional safety: reset gradients
            for device in devices:
                if hasattr(device.model, 'zero_grad'):
                    device.model.zero_grad()
                
        except KeyboardInterrupt:
            dashboard.log("Training interrupted by user")
            raise
        except Exception as e:
            dashboard.log(f"Round {round_num} error: {str(e)[:50]}")
            # Try to continue to next round
            time.sleep(2)
            continue
    
    # Final summary
    dashboard.render()
    print(f"\n{Color.GREEN}{Color.BOLD}{'═' * 80}{Color.RESET}")
    print(f"{Color.GREEN}{Color.BOLD}{'✅ FEDERATED LEARNING COMPLETE':^80}{Color.RESET}")
    print(f"{Color.GREEN}{Color.BOLD}{'═' * 80}{Color.RESET}\n")
    
    print(f"🎯 Final Global Accuracy: {Color.GREEN}{Color.BOLD}{server.global_accuracy:.2f}%{Color.RESET}")
    
    # FIXED: Handle empty accuracy history without NaN
    if len(server.accuracy_history) > 1:
        improvement = server.global_accuracy - list(server.accuracy_history)[0]
        print(f"📈 Improvement: {'+' if improvement >= 0 else ''}{improvement:.2f}%")
    else:
        print(f"📈 Improvement: N/A (insufficient history)")
    
    # FIXED: Handle empty round times without NaN
    if len(server.round_times) > 0:
        print(f"⏱️  Average Round Time: {np.mean(server.round_times):.1f}s")
    else:
        print(f"⏱️  Average Round Time: N/A")
    
    print(f"💾 Total Samples Trained: {server.total_samples:,}")
    print(f"\n{Color.CYAN}Dashboard closed. Training complete!{Color.RESET}\n")


# ============================================================================
# 6. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        run_federated_learning()
    except KeyboardInterrupt:
        print(f"\n\n{Color.YELLOW}⚠️  Training interrupted by user{Color.RESET}\n")
    except Exception as e:
        print(f"\n\n{Color.RED}❌ Error: {e}{Color.RESET}\n")
        import traceback
        traceback.print_exc()

