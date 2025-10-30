"""
Graph Quick Test
===============
Quick test script for graph-based federated learning with limited data and simple model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader as PyGDataLoader
import time
import logging
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def quick_graph_test():
    """Run a quick test with graph data and simple GNN"""
    print("🚀 Starting Graph Quick Test...")
    
    try:
        # Import modules
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))
        from graph_dataset import MalNetGraphDataset, create_graph_data_loaders
        
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'))
        from gnn import create_gnn_model
        
        # Check if dataset exists
        data_dir = "malnet-graphs-tiny"
        if not os.path.exists(data_dir):
            print(f"❌ Dataset directory {data_dir} not found!")
            return
        
        print("📊 Loading graph dataset...")
        
        # Create simple dataset with limited samples
        dataset = MalNetGraphDataset(data_dir, split='train', max_nodes=1000)  # Limit nodes for speed
        
        if len(dataset) == 0:
            print("❌ No samples found in dataset!")
            return
        
        print(f"✅ Loaded {len(dataset)} graph samples")
        print(f"📈 Classes: {list(dataset.class_to_idx.keys())}")
        
        # Create simple data loader
        train_loader = PyGDataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
        
        print("🧠 Creating simple GNN model...")
        
        # Create simple GNN model
        model = create_gnn_model('simple', num_classes=len(dataset.class_to_idx), hidden_dim=32)
        
        print(f"✅ Model created with {model.count_parameters():,} parameters")
        print(f"📏 Model size: {model.get_model_size():.2f} MB")
        
        # Setup training
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        print(f"🔧 Training on device: {device}")
        print("🏃 Starting training...")
        
        # Quick training loop
        model.train()
        for epoch in range(3):  # Just 3 epochs for quick test
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, batch_data in enumerate(train_loader):
                if batch_idx >= 5:  # Limit to 5 batches for speed
                    break
                
                # Handle the list format from DataLoader
                if isinstance(batch_data, list):
                    batch, labels = batch_data
                else:
                    batch = batch_data
                    labels = batch.y
                
                batch = batch.to(device)
                labels = labels.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                output = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(output, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # Calculate epoch metrics
            avg_loss = total_loss / min(5, len(train_loader))
            accuracy = 100.0 * correct / total
            
            print(f"  Epoch {epoch+1}/3: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        print("✅ Graph quick test completed successfully!")
        
        # Test model inference
        print("🔍 Testing model inference...")
        model.eval()
        with torch.no_grad():
            for batch_data in train_loader:
                if isinstance(batch_data, list):
                    batch, labels = batch_data
                else:
                    batch = batch_data
                    labels = batch.y
                
                batch = batch.to(device)
                output = model(batch.x, batch.edge_index, batch.batch)
                print(f"  Output shape: {output.shape}")
                print(f"  Sample predictions: {torch.softmax(output[:3], dim=1)}")
                break
        
        print("🎉 All tests passed! Graph-based system is working.")
        
    except Exception as e:
        print(f"❌ Error in graph quick test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    quick_graph_test()
