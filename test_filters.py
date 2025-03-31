import os
import mne
import numpy as np
import warnings

# 忽略命名约定警告
warnings.filterwarnings("ignore", message="This.*does not conform to MNE naming conventions")

def test_filters():
    """测试MNE滤波器行为"""
    # 生成测试数据
    sfreq = 1000  # 采样率
    t = np.arange(0, 1, 1/sfreq)  # 1秒数据
    n_channels = 10
    data = np.random.randn(n_channels, len(t))  # 随机数据
    
    print("测试短信号的带通滤波器")
    try:
        # 测试正常滤波
        filtered_data = mne.filter.filter_data(
            data, sfreq, l_freq=1, h_freq=40, filter_length='auto'
        )
        print("成功: 使用auto滤波器长度")
    except Exception as e:
        print(f"错误: {e}")
    
    print("\n测试陷波滤波器")
    try:
        # 正确的参数顺序
        notched_data = mne.filter.notch_filter(
            data, sfreq, freqs=60
        )
        print("成功: 正确参数顺序")
    except Exception as e:
        print(f"错误: {e}")
    
    try:
        # 错误的参数顺序
        notched_data = mne.filter.notch_filter(
            data, sfreq=sfreq, freqs=60
        )
        print("成功: 关键字参数也可以工作?")
    except Exception as e:
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    test_filters()