import os
import mne
import numpy as np
import torch
import matplotlib.pyplot as plt
from data_loading import create_data_loaders
import config
import warnings

# Ignore MNE warnings
warnings.filterwarnings("ignore", message="This.*does not conform to MNE naming conventions")
warnings.filterwarnings("ignore", message="filter_length.*is longer than the signal")

def test_data_pipeline():
    """Test the complete data pipeline, including preprocessing and batching"""
    print("Testing data loading and preprocessing...")
    
    # Temporarily modify batch size to speed up testing
    original_batch_size = config.BATCH_SIZE
    config.BATCH_SIZE = 4
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders()
    
    print(f"Number of batches in train loader: {len(train_loader)}")
    
    # Get only a few batches for testing
    batch_count = min(3, len(train_loader))
    
    for i, batch in enumerate(train_loader):
        if i >= batch_count:
            break
            
        meg_data = batch['meg_data']
        text_data = batch['text_data']['input_ids']
        raw_text = batch['raw_text']
        
        print(f"\nBatch {i+1}:")
        print(f"MEG data shape: {meg_data.shape}")
        print(f"Text data shape: {text_data.shape}")
        print(f"Number of samples: {len(raw_text)}")
        
        # Check MEG data statistics
        print(f"MEG data statistics: min={meg_data.min().item():.4f}, max={meg_data.max().item():.4f}, mean={meg_data.mean().item():.4f}")
        
        # Check text ID range
        if text_data.shape[0] > 0:
            print(f"Text ID range: min={text_data.min().item()}, max={text_data.max().item()}")
        
        # Visualize MEG data from the first sample
        if i == 0:
            sample_data = meg_data[0].numpy()
            plt.figure(figsize=(10, 6))
            # Show only first 10 channels to avoid overcrowding
            for j in range(min(10, sample_data.shape[0])):
                plt.plot(sample_data[j] + j*5)  # Add offset for visualization
            plt.title("MEG Data Example (First 10 Channels)")
            plt.savefig("meg_sample_visualization.png")
            print("MEG sample visualization saved to meg_sample_visualization.png")
    
    # Restore original batch size
    config.BATCH_SIZE = original_batch_size
    
    print("\nData pipeline test completed")

if __name__ == "__main__":
    test_data_pipeline()
