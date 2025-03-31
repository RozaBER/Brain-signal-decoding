import os
import mne
import numpy as np
import warnings

# Ignore naming convention warnings
warnings.filterwarnings("ignore", message="This.*does not conform to MNE naming conventions")

def test_filters():
    """Test MNE filter behavior"""
    # Generate test data
    sfreq = 1000  # Sampling rate
    t = np.arange(0, 1, 1/sfreq)  # 1 second of data
    n_channels = 10
    data = np.random.randn(n_channels, len(t))  # Random data
    
    print("Testing bandpass filter for short signals")
    try:
        # Test normal filtering
        filtered_data = mne.filter.filter_data(
            data, sfreq, l_freq=1, h_freq=40, filter_length='auto'
        )
        print("Success: Using auto filter length")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\nTesting notch filter")
    try:
        # Correct parameter order
        notched_data = mne.filter.notch_filter(
            data, sfreq, freqs=60
        )
        print("Success: Correct parameter order")
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        # Incorrect parameter order
        notched_data = mne.filter.notch_filter(
            data, sfreq=sfreq, freqs=60
        )
        print("Success: Keyword arguments also work?")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_filters()
