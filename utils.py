"""
Utility functions for the Brain-to-Text decoding system.
"""
import os
import random
import numpy as np
import torch
from transformers import AutoModel
import config

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(model, optimizer, scheduler, epoch, loss, filename):
    """Save model checkpoint."""
    state = {
        'model': model.state_dict(),
        'optimizer': None if optimizer is None else optimizer.state_dict(),
        'scheduler': None if scheduler is None else scheduler.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, filename, optimizer=None, scheduler=None):
    """Load model checkpoint."""
    state = torch.load(filename)
    model.load_state_dict(state['model'])
    
    if optimizer is not None and state['optimizer'] is not None:
        optimizer.load_state_dict(state['optimizer'])
    
    if scheduler is not None and state['scheduler'] is not None:
        scheduler.load_state_dict(state['scheduler'])
    
    print(f"Loaded checkpoint from {filename} (epoch {state['epoch']}, loss {state['loss']:.4f})")
    
    return model

# Cache for text encoder model
_text_encoder = None

def get_text_embeddings(input_ids, device):
    """
    Get text embeddings using a pretrained language model.
    Uses a cached model for efficiency.
    """
    global _text_encoder
    
    if _text_encoder is None:
        # Use the same model as LLaVA to ensure compatible embeddings
        _text_encoder = AutoModel.from_pretrained(config.LLAVA_MODEL_PATH)
        _text_encoder.to(device)
        _text_encoder.eval()
    
    with torch.no_grad():
        outputs = _text_encoder(input_ids=input_ids)
        # Get last hidden state
        embeddings = outputs.last_hidden_state
    
    return embeddings

def evaluate_bleu(references, hypotheses):
    """
    Calculate BLEU score for generated text.
    
    Args:
        references: List of reference sentences (ground truth)
        hypotheses: List of generated sentences
        
    Returns:
        BLEU score
    """
    from nltk.translate.bleu_score import corpus_bleu
    import nltk
    
    # Tokenize references and hypotheses
    tokenized_references = [[nltk.word_tokenize(ref)] for ref in references]
    tokenized_hypotheses = [nltk.word_tokenize(hyp) for hyp in hypotheses]
    
    # Calculate BLEU score
    bleu = corpus_bleu(tokenized_references, tokenized_hypotheses)
    
    return bleu

def normalize_meg_data(meg_data, method="global"):
    """
    Normalize MEG data using different methods.
    
    Args:
        meg_data: MEG data to normalize
        method: Normalization method ('global', 'channel', 'trial')
        
    Returns:
        Normalized MEG data
    """
    if method == "global":
        # Global normalization
        mean = np.mean(meg_data)
        std = np.std(meg_data)
        return (meg_data - mean) / (std + 1e-8)
    
    elif method == "channel":
        # Normalize each channel separately
        mean = np.mean(meg_data, axis=1, keepdims=True)
        std = np.std(meg_data, axis=1, keepdims=True)
        return (meg_data - mean) / (std + 1e-8)
    
    elif method == "trial":
        # Normalize each trial separately
        if meg_data.ndim == 3:  # Batch of trials
            mean = np.mean(meg_data, axis=(1, 2), keepdims=True)
            std = np.std(meg_data, axis=(1, 2), keepdims=True)
            return (meg_data - mean) / (std + 1e-8)
        else:  # Single trial
            mean = np.mean(meg_data)
            std = np.std(meg_data)
            return (meg_data - mean) / (std + 1e-8)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")