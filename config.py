"""
Configuration parameters for the Brain-to-Text decoding system.
This file contains all hyperparameters and settings for the model.
"""
import os
import torch

# Path configurations
BASE_PATH = ".\\data\\MASC-MEG"
CHECKPOINTS_PATH = ".\\checkpoints"
RESULTS_PATH = ".\\results"
os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

# Data configurations
MEG_CHANNELS = 208  # Number of MEG channels in MASC-MEG dataset
SAMPLING_RATE = 1000  # Hz
TIME_WINDOW = 2.0  # Time window in seconds for signal segmentation
STRIDE = 0.5  # Stride in seconds for sliding window
BATCH_SIZE = 4
NUM_WORKERS = 4
TRAIN_VAL_TEST_SPLIT = [0.7, 0.15, 0.15]  # Split ratios
MAX_SEQ_LENGTH = 128  # Maximum sequence length for text

# Signal processing configurations
BANDPASS_FILTER = [1, 40.0]  # Bandpass filter range in Hz
NOTCH_FILTER = 60  # Notch filter frequency in Hz
NORMALIZATION = 500  # Options: "global", "subject", "trial"

# MEG Encoder configurations
MEG_ENCODER_TYPE = "2d"  # Options: "1d", "2d"
PATCH_SIZE = (4, 100)  # Patch size for 2D encoder (channels, time points)
EMBED_DIM = 768  # Embedding dimension
ENCODER_HIDDEN_DIMS = [64, 128, 256, 512]  # Hidden dimensions for CNN layers
ENCODER_KERNEL_SIZES = [3, 5, 5, 3]  # Kernel sizes for CNN layers
NUM_HEADS = 8  # Number of attention heads
DROPOUT_RATE = 0.1

# Alignment module configurations
ALIGNMENT_HIDDEN_DIM = 1024
PROJECTION_DIM = 512  # Dimension for contrastive learning
TEMPERATURE = 0.07  # Temperature parameter for contrastive loss

# LLaVA configurations
LLAVA_MODEL_PATH = "llava-hf/llava-1.5-7b-hf"  # HuggingFace model path
LLAVA_OUTPUT_DIM = 4096  # Output dimension of LLaVA's vision encoder

# Training configurations
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
WARMUP_STEPS = 1000
LR_SCHEDULER = "cosine"  # Options: "linear", "cosine"
GRADIENT_ACCUMULATION_STEPS = 4
FP16 = True  # Use mixed precision training
SEED = 42
EARLY_STOPPING_PATIENCE = 5

# Stage-wise training
STAGE1_EPOCHS = 10  # Pretraining encoder
STAGE2_EPOCHS = 10  # Cross-modal alignment
STAGE3_EPOCHS = 5   # LLaVA adapter training
STAGE4_EPOCHS = 5   # End-to-end fine-tuning

# Preprocessing configuration
VERBOSE_PREPROCESSING = False  # Set to True to see detailed preprocessing information
DEBUG_MODE = True  # Enable more debug output during training

# Safety configuration
MIN_SIGNAL_LENGTH = 50  # Minimum accepted signal length, too short signals will be skipped

# MEG data standardization configuration
MEG_STANDARD_LENGTH = 100  # Standard length for all MEG segments
MIN_SIGNAL_LENGTH = 50    # Minimum accepted signal length, skip complex filtering below this length

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
