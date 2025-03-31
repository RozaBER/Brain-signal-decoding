# Create test_data_loading.py
import torch
from data_loading import create_data_loaders
import config

def test_data_loading():
    print("Testing data loading...")
    train_loader, val_loader, test_loader = create_data_loaders()
    
    # Get one batch
    batch = next(iter(train_loader))
    print(f"Batch size: {len(batch['meg_data'])}")
    print(f"MEG data shape: {batch['meg_data'].shape}")
    print(f"Text example: {batch['raw_text'][0]}")
    
    print("Data loading test completed!")

if __name__ == "__main__":
    test_data_loading()
