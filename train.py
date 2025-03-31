"""
Training script for the Brain-to-Text model.
Implements the multi-stage training process.
"""
import os
import time
import tqdm
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

import config
from data_loading import create_data_loaders
from meg_encoder import get_meg_encoder
from alignment_module import AlignmentModule
from llava_decoder import BrainLLaVA
from utils import set_seed, save_checkpoint, load_checkpoint, get_text_embeddings

def train_encoder(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    num_epochs=config.STAGE1_EPOCHS,
    device=config.DEVICE
):
    """Stage 1: 预训练MEG编码器"""
    print("开始编码器预训练 (阶段 1)")
    
    # 简单分类头
    num_classes = 10000  # 增大类别数以容纳更大范围的token ID
    vocab_size = 50000
    classifier = nn.Linear(config.EMBED_DIM, vocab_size).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0
        
        for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            meg_data = batch['meg_data'].to(device)
            text_data = batch['text_data']['input_ids'].to(device)
            
            # 前向传播
            meg_embeddings = model(meg_data)
            
            # 平均池化
            meg_embeddings = meg_embeddings.mean(dim=1)
            
            # 预测token IDs
            logits = classifier(meg_embeddings)
            
            # 获取目标 - 使用第一个token
            targets = text_data[:, 0]
            
            # 检查目标有效性
            max_target = torch.max(targets).item()
            if max_target >= num_classes:
                # 裁剪目标到有效范围
                targets = torch.clamp(targets, 0, num_classes-1)
            
            # 确保targets为long类型
            targets = targets.long()
            
            loss = criterion(logits, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 更新学习率
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        val_samples = 0
        
        print("\nStart validation...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")):
                try:
                    meg_data = batch['meg_data'].to(device)
                    text_data = batch['text_data']['input_ids'].to(device)
                    
                    # 安全检查
                    if meg_data.shape[0] == 0 or text_data.shape[0] == 0:
                        print(f"跳过空批次 {batch_idx}")
                        continue
                    
                    # 前向传播
                    meg_embeddings = model(meg_data)
                    meg_embeddings = meg_embeddings.mean(dim=1)
                    logits = classifier(meg_embeddings)
                    
                    # 目标安全处理
                    targets = text_data[:, 0].clone()
                    valid_mask = (targets >= 0) & (targets < vocab_size)
                    
                    if not valid_mask.all():
                        invalid_count = (~valid_mask).sum().item()
                        print(f"警告: 验证批次 {batch_idx} 有 {invalid_count} 个超出范围的目标索引")
                        targets[~valid_mask] = 0  # 使用安全的类别索引
                    
                    targets = targets.long()
                    
                    # 计算损失
                    loss = criterion(logits, targets)
                    
                    val_loss += loss.item()
                    val_samples += 1
                    
                except RuntimeError as e:
                    print(f"验证批次 {batch_idx} 出错: {str(e)}")
                    print(f"跳过此批次并继续")
                    # 打印更多调试信息
                    if 'meg_data' in locals() and 'text_data' in locals():
                        print(f"MEG数据形状: {meg_data.shape}")
                        print(f"文本数据形状: {text_data.shape}")
                        if text_data.shape[0] > 0:
                            print(f"目标值范围: {text_data[:, 0].min().item()} 到 {text_data[:, 0].max().item()}")
                    continue
        
        # 计算平均损失
        val_loss = val_loss / max(1, val_samples)  # 避免除零
        print(f"Epoch {epoch+1}/{num_epochs} - 验证损失: {val_loss:.4f}")

        # Print statistics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss=val_loss,
                filename=os.path.join(config.CHECKPOINTS_PATH, "encoder_best.pt")
            )
    
    # Load best model
    model = load_checkpoint(
        model=model,
        filename=os.path.join(config.CHECKPOINTS_PATH, "encoder_best.pt"),
        optimizer=None,
        scheduler=None
    )
    
    return model

def train_alignment(
    meg_encoder,
    alignment_module,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    num_epochs=config.STAGE2_EPOCHS,
    device=config.DEVICE
):
    """
    Stage 2: Training the cross-modal alignment module using contrastive learning.
    """
    print("Starting Alignment Training (Stage 2)")
    
    # Freeze encoder parameters
    for param in meg_encoder.parameters():
        param.requires_grad = False
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        alignment_module.train()
        train_loss = 0
        
        for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            meg_data = batch['meg_data'].to(device)
            text_data = batch['text_data']['input_ids'].to(device)
            
            # Get MEG embeddings
            with torch.no_grad():
                meg_embeddings = meg_encoder(meg_data)
            
            # Get text embeddings using a language model
            text_embeddings = get_text_embeddings(text_data, device)
            
            # Forward pass through alignment module
            meg_projected, text_projected, _ = alignment_module(meg_embeddings, text_embeddings)
            
            # Compute contrastive loss
            loss = alignment_module.compute_contrastive_loss(meg_projected, text_projected)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Update learning rate
        scheduler.step()
        
        # Validation
        alignment_module.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                meg_data = batch['meg_data'].to(device)
                text_data = batch['text_data']['input_ids'].to(device)
                
                # Get MEG embeddings
                meg_embeddings = meg_encoder(meg_data)
                
                # Get text embeddings
                text_embeddings = get_text_embeddings(text_data, device)
                
                # Forward pass through alignment module
                meg_projected, text_projected, _ = alignment_module(meg_embeddings, text_embeddings)
                
                # Compute contrastive loss
                loss = alignment_module.compute_contrastive_loss(meg_projected, text_projected)
                val_loss += loss.item()
        
        # Print statistics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model=alignment_module,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss=val_loss,
                filename=os.path.join(config.CHECKPOINTS_PATH, "alignment_best.pt")
            )
    
    # Load best model
    alignment_module = load_checkpoint(
        model=alignment_module,
        filename=os.path.join(config.CHECKPOINTS_PATH, "alignment_best.pt"),
        optimizer=None,
        scheduler=None
    )
    
    return alignment_module

def train_llava_adapter(
    brain_llava,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    num_epochs=config.STAGE3_EPOCHS,
    device=config.DEVICE
):
    """
    Stage 3: Training the LLaVA adapter while keeping most of LLaVA frozen.
    """
    print("Starting LLaVA Adapter Training (Stage 3)")
    
    # Freeze encoder and alignment module
    for param in brain_llava.meg_encoder.parameters():
        param.requires_grad = False
    
    for param in brain_llava.alignment_module.parameters():
        param.requires_grad = False
    
    # Only train cross-attention layers in LLaVA initially
    for name, param in brain_llava.llava.named_parameters():
        if "cross_attn" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        brain_llava.train()
        train_loss = 0
        
        for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            meg_data = batch['meg_data'].to(device)
            text_data = batch['text_data']['input_ids'].to(device)
            attention_mask = batch['text_data']['attention_mask'].to(device)
            
            # Get MEG embeddings
            with torch.no_grad():
                meg_embeddings = brain_llava.meg_encoder(meg_data)
                llava_compatible = brain_llava.alignment_module(meg_embeddings)
            
            # Forward pass through LLaVA
            outputs = brain_llava.llava(
                input_ids=text_data,
                attention_mask=attention_mask,
                image_features=llava_compatible,
                labels=text_data,
                return_dict=True
            )
            
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Update learning rate
        scheduler.step()
        
        # Validation
        brain_llava.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                meg_data = batch['meg_data'].to(device)
                text_data = batch['text_data']['input_ids'].to(device)
                attention_mask = batch['text_data']['attention_mask'].to(device)
                
                # Get MEG embeddings
                meg_embeddings = brain_llava.meg_encoder(meg_data)
                llava_compatible = brain_llava.alignment_module(meg_embeddings)
                
                # Forward pass through LLaVA
                outputs = brain_llava.llava(
                    input_ids=text_data,
                    attention_mask=attention_mask,
                    image_features=llava_compatible,
                    labels=text_data,
                    return_dict=True
                )
                
                loss = outputs.loss
                val_loss += loss.item()
        
        # Print statistics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model=brain_llava,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss=val_loss,
                filename=os.path.join(config.CHECKPOINTS_PATH, "brain_llava_adapter_best.pt")
            )
    
    # Load best model
    brain_llava = load_checkpoint(
        model=brain_llava,
        filename=os.path.join(config.CHECKPOINTS_PATH, "brain_llava_adapter_best.pt"),
        optimizer=None,
        scheduler=None
    )
    
    return brain_llava

def train_end_to_end(
    brain_llava,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    num_epochs=config.STAGE4_EPOCHS,
    device=config.DEVICE
):
    """
    Stage 4: End-to-end fine-tuning of the entire model.
    """
    print("Starting End-to-End Fine-tuning (Stage 4)")
    
    # Unfreeze all parameters
    for param in brain_llava.parameters():
        param.requires_grad = True
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        brain_llava.train()
        train_loss = 0
        
        for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            meg_data = batch['meg_data'].to(device)
            text_data = batch['text_data']['input_ids'].to(device)
            attention_mask = batch['text_data']['attention_mask'].to(device)
            
            # Forward pass
            outputs = brain_llava.llava(
                input_ids=text_data,
                attention_mask=attention_mask,
                image_features=brain_llava.alignment_module(brain_llava.meg_encoder(meg_data)),
                labels=text_data,
                return_dict=True
            )
            
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Update learning rate
        scheduler.step()
        
        # Validation
        brain_llava.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                meg_data = batch['meg_data'].to(device)
                text_data = batch['text_data']['input_ids'].to(device)
                attention_mask = batch['text_data']['attention_mask'].to(device)
                
                # Forward pass
                outputs = brain_llava.llava(
                    input_ids=text_data,
                    attention_mask=attention_mask,
                    image_features=brain_llava.alignment_module(brain_llava.meg_encoder(meg_data)),
                    labels=text_data,
                    return_dict=True
                )
                
                loss = outputs.loss
                val_loss += loss.item()
        
        # Print statistics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model=brain_llava,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss=val_loss,
                filename=os.path.join(config.CHECKPOINTS_PATH, "brain_llava_full_best.pt")
            )
    
    # Load best model
    brain_llava = load_checkpoint(
        model=brain_llava,
        filename=os.path.join(config.CHECKPOINTS_PATH, "brain_llava_full_best.pt"),
        optimizer=None,
        scheduler=None
    )
    
    return brain_llava



def train():
    """
    Main training function that implements the multi-stage training process.
    """
    # Set random seed for reproducibility
    set_seed(config.SEED)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders()
    
    # Initialize models
    meg_encoder = get_meg_encoder().to(config.DEVICE)
    alignment_module = AlignmentModule().to(config.DEVICE)
    
    # Stage 1: Pretrain MEG encoder
    optimizer = optim.AdamW(
        meg_encoder.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    total_steps = len(train_loader) * config.STAGE1_EPOCHS
    
    if config.LR_SCHEDULER == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.WARMUP_STEPS,
            num_training_steps=total_steps
        )
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.WARMUP_STEPS,
            num_training_steps=total_steps
        )
    
    meg_encoder = train_encoder(
        model=meg_encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    # Stage 2: Train alignment module
    optimizer = optim.AdamW(
        alignment_module.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    total_steps = len(train_loader) * config.STAGE2_EPOCHS
    
    if config.LR_SCHEDULER == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.WARMUP_STEPS,
            num_training_steps=total_steps
        )
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.WARMUP_STEPS,
            num_training_steps=total_steps
        )
    
    alignment_module = train_alignment(
        meg_encoder=meg_encoder,
        alignment_module=alignment_module,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    # Initialize BrainLLaVA model
    brain_llava = BrainLLaVA(
        meg_encoder=meg_encoder,
        alignment_module=alignment_module
    ).to(config.DEVICE)
    
    # Stage 3: Train LLaVA adapter
    # Only train cross-attention layers at this stage
    optimizer = optim.AdamW(
        [p for p in brain_llava.parameters() if p.requires_grad],
        lr=config.LEARNING_RATE / 10,  # Lower learning rate for pretrained model
        weight_decay=config.WEIGHT_DECAY
    )
    
    total_steps = len(train_loader) * config.STAGE3_EPOCHS
    
    if config.LR_SCHEDULER == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.WARMUP_STEPS,
            num_training_steps=total_steps
        )
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.WARMUP_STEPS,
            num_training_steps=total_steps
        )
    
    brain_llava = train_llava_adapter(
        brain_llava=brain_llava,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    # Stage 4: End-to-end fine-tuning
    # Unfreeze all parameters
    brain_llava.unfreeze_llava()
    
    optimizer = optim.AdamW(
        [
            {'params': brain_llava.meg_encoder.parameters(), 'lr': config.LEARNING_RATE / 100},
            {'params': brain_llava.alignment_module.parameters(), 'lr': config.LEARNING_RATE / 100},
            {'params': brain_llava.llava.parameters(), 'lr': config.LEARNING_RATE / 1000}
        ],
        weight_decay=config.WEIGHT_DECAY
    )
    
    total_steps = len(train_loader) * config.STAGE4_EPOCHS
    
    if config.LR_SCHEDULER == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.WARMUP_STEPS,
            num_training_steps=total_steps
        )
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.WARMUP_STEPS,
            num_training_steps=total_steps
        )
    
    brain_llava = train_end_to_end(
        brain_llava=brain_llava,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler
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
    
    return brain_llava