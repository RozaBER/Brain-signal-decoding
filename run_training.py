import os
import torch
from train import train_encoder, train_alignment, train_llava_adapter, train_end_to_end
from meg_encoder import get_meg_encoder
from alignment_module import AlignmentModule
from llava_decoder import BrainLLaVA
from data_loading import create_data_loaders
import config
from utils import set_seed, save_checkpoint, load_checkpoint

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

def main():
    # Set random seed
    set_seed(config.SEED)
    
    # Create data loaders
    train_loader, val_loader, _ = create_data_loaders()
    
    # Decide which stage to start from
    start_stage = 1  # Modify this to start from different stages
    
    # Stage 1: Pretrain MEG encoder
    if start_stage <= 1:
        print("Starting Stage 1: Pretraining MEG encoder")
        meg_encoder = get_meg_encoder().to(config.DEVICE)
        optimizer = torch.optim.AdamW(meg_encoder.parameters(), lr=config.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.STAGE1_EPOCHS)
        
        meg_encoder = train_encoder(
            model=meg_encoder,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=config.STAGE1_EPOCHS
        )
    else:
        # Load from checkpoint
        meg_encoder = get_meg_encoder().to(config.DEVICE)
        meg_encoder = load_checkpoint(
            model=meg_encoder,
            filename=os.path.join(config.CHECKPOINTS_PATH, "encoder_best.pt"),
            optimizer=None,
            scheduler=None
        )
    
    # Stage 2: Train alignment module
    if start_stage <= 2:
        print("Starting Stage 2: Training alignment module")
        alignment_module = AlignmentModule().to(config.DEVICE)
        optimizer = torch.optim.AdamW(alignment_module.parameters(), lr=config.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.STAGE2_EPOCHS)
        
        alignment_module = train_alignment(
            meg_encoder=meg_encoder,
            alignment_module=alignment_module,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=config.STAGE2_EPOCHS
        )
    else:
        # Load from checkpoint
        alignment_module = AlignmentModule().to(config.DEVICE)
        alignment_module = load_checkpoint(
            model=alignment_module,
            filename=os.path.join(config.CHECKPOINTS_PATH, "alignment_best.pt"),
            optimizer=None,
            scheduler=None
        )
    
    # Initialize BrainLLaVA model
    brain_llava = BrainLLaVA(
        meg_encoder=meg_encoder,
        alignment_module=alignment_module
    ).to(config.DEVICE)
    
    # Stage 3: Train LLaVA adapter
    if start_stage <= 3:
        print("Starting Stage 3: Training LLaVA adapter")
        # Only train cross-attention layers
        optimizer = torch.optim.AdamW(
            [p for p in brain_llava.parameters() if p.requires_grad],
            lr=config.LEARNING_RATE / 10
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.STAGE3_EPOCHS)
        
        brain_llava = train_llava_adapter(
            brain_llava=brain_llava,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=config.STAGE3_EPOCHS
        )
    else:
        # Load from checkpoint
        brain_llava = load_checkpoint(
            model=brain_llava,
            filename=os.path.join(config.CHECKPOINTS_PATH, "brain_llava_adapter_best.pt"),
            optimizer=None,
            scheduler=None
        )
    
    # Stage 4: End-to-end fine-tuning
    if start_stage <= 4:
        print("Starting Stage 4: End-to-end fine-tuning")
        # Unfreeze all parameters
        brain_llava.unfreeze_llava()
        
        optimizer = torch.optim.AdamW([
            {'params': brain_llava.meg_encoder.parameters(), 'lr': config.LEARNING_RATE / 100},
            {'params': brain_llava.alignment_module.parameters(), 'lr': config.LEARNING_RATE / 100},
            {'params': brain_llava.llava.parameters(), 'lr': config.LEARNING_RATE / 1000}
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.STAGE4_EPOCHS)
        
        brain_llava = train_end_to_end(
            brain_llava=brain_llava,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=config.STAGE4_EPOCHS
        )
    
    # Save final model
    save_checkpoint(
        model=brain_llava,
        optimizer=None,
        scheduler=None,
        epoch=config.NUM_EPOCHS,
        loss=0.0,
        filename=os.path.join(config.CHECKPOINTS_PATH, "brain_llava_final.pt")
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()
