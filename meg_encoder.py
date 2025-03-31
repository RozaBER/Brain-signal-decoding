"""
MEG Signal Encoder implementation.
This module contains the neural network architecture for encoding MEG signals.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class MEG1DEncoder(nn.Module):
    """
    1D Convolutional Encoder for MEG signals.
    This encoder applies 1D convolutions to extract temporal features from MEG channels.
    """
    def __init__(
        self,
        in_channels=config.MEG_CHANNELS,
        hidden_dims=config.ENCODER_HIDDEN_DIMS,
        kernel_sizes=config.ENCODER_KERNEL_SIZES,
        embed_dim=config.EMBED_DIM,
        dropout=config.DROPOUT_RATE
    ):
        super().__init__()
        
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First conv layer
        self.conv_layers.append(
            nn.Conv1d(in_channels, hidden_dims[0], kernel_size=kernel_sizes[0], padding='same')
        )
        self.batch_norms.append(nn.BatchNorm1d(hidden_dims[0]))
        
        # Add remaining conv layers
        for i in range(1, len(hidden_dims)):
            self.conv_layers.append(
                nn.Conv1d(
                    hidden_dims[i-1], 
                    hidden_dims[i], 
                    kernel_size=kernel_sizes[i], 
                    stride=2,
                    padding=kernel_sizes[i]//2
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dims[i]))
        
        # Final projection to embedding dimension
        self.fc = nn.Linear(hidden_dims[-1], embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Positional encoding for temporal information
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        
        # Transformer layers for contextual information
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=config.NUM_HEADS,
            dim_feedforward=embed_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
    
    def forward(self, x):
        """
        Forward pass through the MEG encoder.
        
        Args:
            x: Input MEG signal of shape [batch_size, channels, time_points]
            
        Returns:
            MEG embeddings of shape [batch_size, sequence_length, embed_dim]
        """
        # Apply 1D convolutions
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.batch_norms)):
            x = F.relu(bn(conv(x)))
        
        # Rearrange from [batch, channels, time] to [batch, time, channels]
        x = x.permute(0, 2, 1)
        
        # Project to embedding dimension
        x = self.fc(x)
        x = self.dropout(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        return x

class MEG2DEncoder(nn.Module):
    """
    2D Convolutional Encoder for MEG signals.
    This encoder applies 2D convolutions to extract spatiotemporal features from MEG data.
    """
    def __init__(
        self,
        in_channels=1,  # Single channel for 2D input
        patch_size=config.PATCH_SIZE,
        hidden_dims=config.ENCODER_HIDDEN_DIMS,
        embed_dim=config.EMBED_DIM,
        dropout=config.DROPOUT_RATE
    ):
        super().__init__()
        
        self.patch_size = patch_size
        
        # Patchify and embed layer
        self.patchify = nn.Conv2d(
            in_channels,
            hidden_dims[0],
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(len(hidden_dims)-1):
            self.conv_layers.append(
                nn.Conv2d(
                    hidden_dims[i],
                    hidden_dims[i+1],
                    kernel_size=3,
                    padding=1,
                    stride=1
                )
            )
            self.batch_norms.append(nn.BatchNorm2d(hidden_dims[i+1]))
        
        # Calculate output size after convolutions
        self.flatten = nn.Flatten(1, 2)
        
        # Final projection to embedding dimension
        self.projection = nn.Linear(hidden_dims[-1], embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Positional encoding for sequence information
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        
        # Transformer layers for contextual information
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=config.NUM_HEADS,
            dim_feedforward=embed_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
    
    def forward(self, x):
        """
        Forward pass through the MEG encoder.
        
        Args:
            x: Input MEG signal of shape [batch_size, channels, time_points]
            
        Returns:
            MEG embeddings of shape [batch_size, sequence_length, embed_dim]
        """
        batch_size = x.shape[0]
        
        # Reshape input to 2D format [batch_size, 1, channels, time_points]
        x = x.unsqueeze(1)
        
        # Patchify
        x = self.patchify(x)
        
        # Apply convolutions
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            x = F.relu(bn(conv(x)))
        
        # Reshape to sequence format [batch_size, seq_len, features]
        x = x.permute(0, 2, 3, 1)
        seq_len = x.shape[1] * x.shape[2]
        x = x.reshape(batch_size, seq_len, -1)
        
        # Project to embedding dimension
        x = self.projection(x)
        x = self.dropout(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        return x

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    This adds information about the position of tokens in the sequence.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

def get_meg_encoder():
    """Factory function to create the appropriate MEG encoder based on config."""
    if config.MEG_ENCODER_TYPE == "1d":
        return MEG1DEncoder()
    elif config.MEG_ENCODER_TYPE == "2d":
        return MEG2DEncoder()
    else:
        raise ValueError(f"Unknown encoder type: {config.MEG_ENCODER_TYPE}")