import os
import mne
import numpy as np
import torch
import matplotlib.pyplot as plt
from data_loading import create_data_loaders
import config
import warnings

# 忽略MNE警告
warnings.filterwarnings("ignore", message="This.*does not conform to MNE naming conventions")
warnings.filterwarnings("ignore", message="filter_length.*is longer than the signal")

def test_data_pipeline():
    """测试完整数据管道，包括预处理和批处理"""
    print("测试数据加载和预处理...")
    
    # 临时修改批次大小以加快测试
    original_batch_size = config.BATCH_SIZE
    config.BATCH_SIZE = 4
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders()
    
    print(f"训练加载器批次数: {len(train_loader)}")
    
    # 仅获取少量批次进行测试
    batch_count = min(3, len(train_loader))
    
    for i, batch in enumerate(train_loader):
        if i >= batch_count:
            break
            
        meg_data = batch['meg_data']
        text_data = batch['text_data']['input_ids']
        raw_text = batch['raw_text']
        
        print(f"\n批次 {i+1}:")
        print(f"MEG数据形状: {meg_data.shape}")
        print(f"文本数据形状: {text_data.shape}")
        print(f"样本数: {len(raw_text)}")
        
        # 检查MEG数据统计信息
        print(f"MEG数据统计: min={meg_data.min().item():.4f}, max={meg_data.max().item():.4f}, mean={meg_data.mean().item():.4f}")
        
        # 检查文本ID范围
        if text_data.shape[0] > 0:
            print(f"文本ID范围: min={text_data.min().item()}, max={text_data.max().item()}")
        
        # 可视化第一个样本的MEG数据
        if i == 0:
            sample_data = meg_data[0].numpy()
            plt.figure(figsize=(10, 6))
            # 只显示前10个通道，避免过于拥挤
            for j in range(min(10, sample_data.shape[0])):
                plt.plot(sample_data[j] + j*5)  # 添加偏移以便可视化
            plt.title("MEG数据示例 (前10个通道)")
            plt.savefig("meg_sample_visualization.png")
            print("MEG样本可视化已保存到 meg_sample_visualization.png")
    
    # 恢复原始批次大小
    config.BATCH_SIZE = original_batch_size
    
    print("\n数据管道测试完成")

if __name__ == "__main__":
    test_data_pipeline()