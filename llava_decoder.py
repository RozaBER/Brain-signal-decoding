"""
LLaVA Decoder for brain-to-text generation.
This module adapts the LLaVA model to work with brain signal embeddings.
"""
import torch
import torch.nn as nn
from transformers import AutoProcessor, LlavaForConditionalGeneration
import config

class BrainLLaVAAdapter(nn.Module):
    """
    Adapter module to make brain signal embeddings compatible with LLaVA's vision encoder output.
    This allows the language model to generate text based on brain signals.
    """
    def __init__(self, input_dim=config.PROJECTION_DIM, output_dim=config.LLAVA_OUTPUT_DIM):
        super().__init__()
        
        # Adapter network
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x):
        return self.adapter(x)

class BrainLLaVA(nn.Module):
    """
    Complete Brain-to-Text model using LLaVA.
    Combines MEG encoder, alignment module, and LLaVA for text generation.
    """
    def __init__(self, meg_encoder, alignment_module):
        super().__init__()
        
        self.meg_encoder = meg_encoder
        self.alignment_module = alignment_module
        
        # Load LLaVA model and processor
        self.processor = AutoProcessor.from_pretrained(config.LLAVA_MODEL_PATH)
        self.llava = LlavaForConditionalGeneration.from_pretrained(config.LLAVA_MODEL_PATH)
        
        # Freeze LLaVA parameters initially
        for param in self.llava.parameters():
            param.requires_grad = False
        
        # Only unfreeze the cross-attention layers to begin with
        for name, param in self.llava.named_parameters():
            if "cross_attn" in name:
                param.requires_grad = True
    
    def forward(self, meg_data, prompt=None, max_length=50):
        """
        Forward pass for text generation from MEG data.
        
        Args:
            meg_data: MEG signal data
            prompt: Optional prompt to guide generation
            max_length: Maximum length of generated text
            
        Returns:
            Generated text from the model
        """
        # Get MEG embeddings
        meg_embeddings = self.meg_encoder(meg_data)
        
        # Project to LLaVA-compatible format
        llava_compatible = self.alignment_module(meg_embeddings)
        
        # Default prompt if none provided
        if prompt is None:
            prompt = "What is being thought about based on this brain activity?"
        
        # Prepare inputs for LLaVA
        inputs = self.processor(
            text=prompt,
            return_tensors="pt"
        ).to(meg_data.device)
        
        # Replace vision encoder outputs with our brain signal embeddings
        # This is a key part that adapts LLaVA to work with brain signals
        with torch.no_grad():
            # Get the expected shape by running a dummy forward pass
            dummy_image = torch.zeros(1, 3, 336, 336, device=meg_data.device)
            dummy_inputs = self.processor(images=dummy_image, text=prompt, return_tensors="pt").to(meg_data.device)
            vision_outputs = self.llava.vision_model(dummy_inputs.pixel_values)
            expected_shape = vision_outputs.last_hidden_state.shape
        
        # Reshape our brain embeddings to match the expected shape
        batch_size = meg_data.shape[0]
        brain_vision_outputs = llava_compatible.view(batch_size, expected_shape[1], -1)
        
        # Generate text
        outputs = self.llava.generate(
            input_ids=inputs.input_ids.to(meg_data.device),
            image_features=brain_vision_outputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            attention_mask=inputs.attention_mask.to(meg_data.device),
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id
        )
        
        # Decode the generated text
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)
        
        return generated_text
    
    def unfreeze_llava(self):
        """Unfreeze all LLaVA parameters for fine-tuning."""
        for param in self.llava.parameters():
            param.requires_grad = True