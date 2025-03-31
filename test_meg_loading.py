import os
import mne
import warnings
import numpy as np
from matplotlib import pyplot as plt

# 忽略MNE关于文件命名约定的警告
warnings.filterwarnings("ignore", message="This.*does not conform to MNE naming conventions")

def test_meg_reading():
    """测试不同方法读取MEG文件"""
    # 测试文件路径
    base_path = r"F:\MachinLearning\v1\data\MASC-MEG"
    sample_file = os.path.join(base_path, "sub-01", "ses-0", "meg", "sub-01_ses-0_task-0_meg.con")
    sample_file = os.path.normpath(sample_file)
    
    print(f"测试读取文件: {sample_file}")
    
    if not os.path.exists(sample_file):
        print(f"错误: 文件不存在!")
        return
    
    # 尝试使用read_raw_kit
    try:
        print("\n尝试使用 mne.io.read_raw_kit...")
        raw_kit = mne.io.read_raw_kit(sample_file, preload=True)
        print("成功! 数据形状:", raw_kit.get_data().shape)
        
        # 绘制几个通道的数据片段
        data = raw_kit.get_data()[:5, :1000]  # 前5个通道，1000个时间点
        plt.figure(figsize=(10, 6))
        for i in range(5):
            plt.plot(data[i] + i*10)  # 添加偏移以便可视化
        plt.title("MEG数据可视化 (使用read_raw_kit)")
        plt.savefig("meg_data_visualization.png")
        print("数据可视化已保存到 meg_data_visualization.png")
        
    except Exception as e:
        print(f"失败: {str(e)}")
    
    # 尝试使用read_raw (自动检测)
    try:
        print("\n尝试使用 mne.io.read_raw...")
        raw_auto = mne.io.read_raw(sample_file, preload=True)
        print("成功! 数据形状:", raw_auto.get_data().shape)
    except Exception as e:
        print(f"失败: {str(e)}")

if __name__ == "__main__":
    test_meg_reading()