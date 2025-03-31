# 创建test_data_loading.py
import torch
from data_loading import create_data_loaders
import config

def test_data_loading():
    print("测试数据加载...")
    train_loader, val_loader, test_loader = create_data_loaders()
    
    # 获取一个批次
    batch = next(iter(train_loader))
    print(f"批次大小: {len(batch['meg_data'])}")
    print(f"MEG数据形状: {batch['meg_data'].shape}")
    print(f"文本样例: {batch['raw_text'][0]}")
    
    print("数据加载测试完成!")

if __name__ == "__main__":
    test_data_loading()