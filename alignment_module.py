"""
Cross-Modal Alignment Module.
This module aligns MEG signal representations with language representations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class AlignmentModule(nn.Module):
    """
    Alignment module that maps MEG embeddings to language embedding space.
    Uses contrastive learning to align the two modalities.
    """
    def __init__(
        self,
        meg_dim=config.EMBED_DIM,
        text_dim=config.EMBED_DIM,
        hidden_dim=config.ALIGNMENT_HIDDEN_DIM,
        projection_dim=config.PROJECTION_DIM,
        dropout=config.DROPOUT_RATE
    ):
        super().__init__()
        
        # MEG projection network
        self.meg_projection = nn.Sequential(
            nn.Linear(meg_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
        # Text projection network
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
        # Temperature parameter for contrastive loss
        self.temperature = nn.Parameter(torch.ones([]) * config.TEMPERATURE)
        
        # Final projection to LLaVA's expected input dimension
        self.to_llava = nn.Linear(projection_dim, config.LLAVA_OUTPUT_DIM)
    
    def forward(self, meg_embeddings, text_embeddings=None):
        """
        Forward pass through the alignment module.
        
        Args:
            meg_embeddings: MEG embeddings from the encoder
            text_embeddings: Text embeddings (optional, used only during training)
            
        Returns:
            If text_embeddings is provided (training):
                meg_projected: Projected MEG embeddings
                text_projected: Projected text embeddings
                llava_compatible: LLaVA-compatible MEG embeddings
            If text_embeddings is None (inference):
                llava_compatible: LLaVA-compatible MEG embeddings
        """
        # Get global MEG representation by mean pooling
        if meg_embeddings.dim() > 2:
            meg_global = meg_embeddings.mean(dim=1)
        else:
            meg_global = meg_embeddings
        
        # Project MEG embeddings
        meg_projected = self.meg_projection(meg_global)
        meg_projected = F.normalize(meg_projected, dim=-1)
        
        # Convert to LLaVA-compatible format
        llava_compatible = self.to_llava(meg_projected)
        
        if text_embeddings is not None:
            # Get global text representation by mean pooling if needed
            if text_embeddings.dim() > 2:
                text_global = text_embeddings.mean(dim=1)
            else:
                text_global = text_embeddings
            
            # Project text embeddings
            text_projected = self.text_projection(text_global)
            text_projected = F.normalize(text_projected, dim=-1)
            
            return meg_projected, text_projected, llava_compatible
        else:
            return llava_compatible
    
    def compute_contrastive_loss(self, meg_projections, text_projections):
        """
        Compute contrastive loss between MEG and text projections.
        
        Args:
            meg_projections: Projected MEG embeddings [batch_size, projection_dim]
            text_projections: Projected text embeddings [batch_size, projection_dim]
            
        Returns:
            Contrastive loss value
        """
        # Compute similarity matrix
        logits = torch.matmul(meg_projections, text_projections.T) / self.temperature
        
        # Contrastive loss (InfoNCE)
        batch_size = meg_projections.shape[0]
        labels = torch.arange(batch_size, device=meg_projections.device)
        
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        
        return (loss_i + loss_t) / 2