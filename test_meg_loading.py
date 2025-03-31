import os
import mne
import warnings
import numpy as np
from matplotlib import pyplot as plt

# Ignore MNE warnings about file naming conventions
warnings.filterwarnings("ignore", message="This.*does not conform to MNE naming conventions")

def test_meg_reading():
    """Test different methods for reading MEG files"""
    # Test file path
    base_path = r"F:\MachinLearning\v1\data\MASC-MEG"
    sample_file = os.path.join(base_path, "sub-01", "ses-0", "meg", "sub-01_ses-0_task-0_meg.con")
    sample_file = os.path.normpath(sample_file)
    
    print(f"Testing file reading: {sample_file}")
    
    if not os.path.exists(sample_file):
        print(f"Error: File does not exist!")
        return
    
    # Try using read_raw_kit
    try:
        print("\nTrying mne.io.read_raw_kit...")
        raw_kit = mne.io.read_raw_kit(sample_file, preload=True)
        print("Success! Data shape:", raw_kit.get_data().shape)
        
        # Plot a segment of a few channels
        data = raw_kit.get_data()[:5, :1000]  # First 5 channels, 1000 time points
        plt.figure(figsize=(10, 6))
        for i in range(5):
            plt.plot(data[i] + i*10)  # Add offset for visualization
        plt.title("MEG Data Visualization (using read_raw_kit)")
        plt.savefig("meg_data_visualization.png")
        print("Data visualization saved to meg_data_visualization.png")
        
    except Exception as e:
        print(f"Failed: {str(e)}")
    
    # Try using read_raw (auto-detect)
    try:
        print("\nTrying mne.io.read_raw...")
        raw_auto = mne.io.read_raw(sample_file, preload=True)
        print("Success! Data shape:", raw_auto.get_data().shape)
    except Exception as e:
        print(f"Failed: {str(e)}")

if __name__ == "__main__":
    test_meg_reading()
