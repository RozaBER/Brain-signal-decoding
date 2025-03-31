"""
Dataset and data loading utilities for MASC-MEG dataset.
This module handles loading, preprocessing, and batching of MEG data and corresponding text.
"""
import warnings
import os
import mne
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
import config

warnings.filterwarnings("ignore", message="filter_length.*is longer than the signal")
warnings.filterwarnings("ignore", message="This.*does not conform to MNE naming conventions")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.LLAVA_MODEL_PATH)

class MASCMEGDataset(Dataset):
    def __init__(self, base_path, subjects=None, sessions=None, tasks=None, transform=None):
        """
        Dataset for MASC-MEG brain signal decoding.
        
        Args:
            base_path: Path to the MASC-MEG dataset
            subjects: List of subject IDs to include, or None for all subjects
            sessions: List of session IDs to include, or None for all sessions
            tasks: List of task IDs to include, or None for all tasks
            transform: Optional transform to apply to MEG signals
        """
        self.base_path = base_path
        self.transform = transform
        
        # Default to all subjects, sessions, and tasks if not specified
        if subjects is None:
            subjects = [f"sub-{i:02d}" for i in range(1, 12)] 
        if sessions is None:
            sessions = ["ses-0", "ses-1"]
        if tasks is None:
            tasks = [f"task-{i}" for i in range(4)] 
            
        self.subjects = subjects
        self.sessions = sessions
        self.tasks = tasks
        
        # Load all available data files
        self.data_files = self._get_data_files()
        
        # Load event information for each file
        self.events_info = self._load_events_info()
        
        # Create samples from events
        self.samples = self._create_samples()
        
        print(f"Loaded {len(self.samples)} samples from MASC-MEG dataset")

    def _standardize_meg_length(self, meg_data, target_length=None):
    
        if target_length is None:
            target_length = 100  # 默认目标长度，可以在config中设置
    
        # 检查数据有效性
        if meg_data is None or meg_data.size == 0:
            print("警告: 收到空MEG数据，返回零矩阵")
            return np.zeros((config.MEG_CHANNELS, target_length))
    
        # 检查是否包含NaN或无穷大
        if np.isnan(meg_data).any() or np.isinf(meg_data).any():
            print("警告: MEG数据包含NaN或无穷大值，进行替换")
            meg_data = np.nan_to_num(meg_data, nan=0.0, posinf=1.0, neginf=-1.0)
    
        # 确保target_length是正整数
        if not isinstance(target_length, int) or target_length <= 0:
            print(f"警告: 无效的目标长度 {target_length}，使用默认值100")
            target_length = 100
    
        current_length = meg_data.shape[1]
        channels = meg_data.shape[0]

        if current_length == target_length:
            return meg_data  # 已经是正确长度
    
        if current_length > target_length:
            # 如果太长，则截取中间部分
            start = (current_length - target_length) // 2
            return meg_data[:, start:start+target_length]
        else:
            try:
                # 如果太短，则用零填充
                padded_data = np.zeros((channels, target_length))
                start = (target_length - current_length) // 2
                padded_data[:, start:start+current_length] = meg_data
                return padded_data
            except Exception as e:
                print(f"填充MEG数据时出错: {e}")
                print(f"MEG数据形状: {meg_data.shape}, 目标长度: {target_length}")
                # 返回一个安全的零矩阵
                return np.zeros((channels, target_length))
    
    def _get_data_files(self):
        """Get all available MEG data files matching the specified subjects, sessions, and tasks."""
        data_files = []

        print(f"正在搜索MEG文件，基础路径: {self.base_path}")
        print(f"搜索受试者: {self.subjects}")
        print(f"搜索会话: {self.sessions}")
        print(f"搜索任务: {self.tasks}")
        
        for subject in self.subjects:
            for session in self.sessions:
                for task in self.tasks:
                    # Check if the file exists
                    meg_file = os.path.join(
                        self.base_path, 
                        subject, 
                        session, 
                        'meg',
                        f"{subject}_{session}_{task}_meg.con"
                    )
                    
                    meg_file = os.path.normpath(meg_file)

                    # 构建事件文件路径
                    events_file = os.path.join(
                        self.base_path, 
                        subject, 
                        session, 
                        'meg',
                        f"{subject}_{session}_{task}_events.tsv"
                    )
                    events_file = os.path.normpath(events_file)
                    
                    if os.path.exists(meg_file) and os.path.exists(events_file):
                        print(f"找到MEG文件: {meg_file}")
                        data_files.append({
                            'subject': subject,
                            'session': session,
                            'task': task,
                            'meg_file': meg_file,
                            'events_file': events_file
                    })
                        
        print(f"总共找到 {len(data_files)} 个MEG文件")
        return data_files
    
    def _load_events_info(self):
        """Load event information (word/phoneme timestamps) for each data file."""
        events_info = {}
        
        for file_info in self.data_files:
            subject = file_info['subject']
            session = file_info['session']
            task = file_info['task']
            
            # Path to events TSV file
            events_file = os.path.join(
                self.base_path, 
                subject, 
                session, 
                'meg',
                f"{subject}_{session}_{task}_events.tsv"
            )
            
            if os.path.exists(events_file):
                # Load events data
                events_df = pd.read_csv(events_file, sep='\t')
                
                # Extract word events (filtering out other types of events if necessary)
                word_events = events_df[events_df['trial_type'].str.contains('word', na=False)]
                
                key = f"{subject}_{session}_{task}"
                events_info[key] = word_events
        
        return events_info
    
    def _create_samples(self):
        """Create samples from MEG data and events information."""
        samples = []
        
        for file_info in self.data_files:
            subject = file_info['subject']
            session = file_info['session']
            task = file_info['task']
            key = f"{subject}_{session}_{task}"
            
            if key in self.events_info:
                word_events = self.events_info[key]
                
                # Group words into sentences based on punctuation or timing
                sentences = []
                current_sentence = []
                current_sentence_meg_start = None
                
                for _, event in word_events.iterrows():
                    word = event['trial_type'].split(':')[1] if ':' in event['trial_type'] else event['trial_type']
                    onset = event['onset']
                    duration = event['duration']
                    
                    if current_sentence_meg_start is None:
                        current_sentence_meg_start = onset
                    
                    current_sentence.append(word)
                    
                    # End of sentence if word ends with punctuation or long pause
                    if word.endswith(('.', '!', '?')) or (
                        len(word_events) > _ + 1 and 
                        word_events.iloc[_ + 1]['onset'] - (onset + duration) > 0.5
                    ):
                        sentence_text = ' '.join(current_sentence)
                        sentence_meg_end = onset + duration
                        
                        samples.append({
                            'subject': subject,
                            'session': session,
                            'task': task,
                            'meg_file': file_info['meg_file'],
                            'meg_start': current_sentence_meg_start,
                            'meg_end': sentence_meg_end,
                            'text': sentence_text
                        })
                        
                        current_sentence = []
                        current_sentence_meg_start = None
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取数据集中的样本"""
        sample = self.samples[idx]
    
        # 加载MEG数据
        try:
            # 使用正确的函数读取.con文件
            raw = mne.io.read_raw_kit(sample['meg_file'], preload=False)
        
            # 提取对应时间窗口的数据
            start_time = sample['meg_start']
            end_time = sample['meg_end']
        
            # 裁剪到特定时间窗口
            raw.crop(tmin=start_time, tmax=end_time)
            meg_data = raw.get_data()
        
            # 应用预处理
            meg_data = self._preprocess_meg(meg_data, raw.info)
        
            # 关键新步骤: 标准化数据长度
            meg_data = self._standardize_meg_length(meg_data, 100)  # 100是目标长度，可调整
        
        except Exception as e:
            print(f"读取MEG文件出错: {sample['meg_file']}")
            print(f"错误信息: {str(e)}")
            # 返回一个零数组作为备选
            meg_data = np.zeros((config.MEG_CHANNELS, 100))  # 使用相同目标长度
    
        # 其余部分保持不变
        if self.transform:
            meg_data = self.transform(meg_data)
    
        text = sample['text']
        tokenized_text = tokenizer(
            text,
            padding="max_length",
            max_length=config.MAX_SEQ_LENGTH,
            truncation=True,
            return_tensors="pt"
        )
    
        return {
            'meg_data': torch.FloatTensor(meg_data),
            'text_data': {
                'input_ids': tokenized_text['input_ids'].squeeze(),
                'attention_mask': tokenized_text['attention_mask'].squeeze()
            },
            'raw_text': text,
            'subject': sample['subject'],
            'task': sample['task']
        }
    
    def _preprocess_meg(self, meg_data, info):
        """预处理MEG数据"""
        
        signal_length = meg_data.shape[1]

        # 针对超短信号简化处理
        if signal_length < 50:  # 对于非常短的信号
            # 跳过滤波，只做归一化
            if config.NORMALIZATION == "global":
                meg_data = (meg_data - np.mean(meg_data)) / (np.std(meg_data) + 1e-8)
            elif config.NORMALIZATION == "channel":
                for i in range(meg_data.shape[0]):
                    channel_data = meg_data[i]
                    if np.std(channel_data) > 1e-8:
                        meg_data[i] = (channel_data - np.mean(channel_data)) / (np.std(channel_data) + 1e-8)
            return meg_data
    
        # 对于正常长度的信号，应用滤波器
        if config.BANDPASS_FILTER and signal_length >= 50:
            try:
                # 使用安全的滤波器长度
                safe_filter_length = min(signal_length // 3, 101)
                safe_filter_length = max(safe_filter_length, 15)  # 至少15个样本
            
                meg_data = mne.filter.filter_data(
                    meg_data, 
                    sfreq=info['sfreq'],
                    l_freq=config.BANDPASS_FILTER[0],
                    h_freq=config.BANDPASS_FILTER[1],
                    filter_length=safe_filter_length,
                    verbose=False
                )
            except Exception as e:
                print(f"带通滤波器错误: {e}")
    
        # 对于长度足够的信号，应用陷波滤波器
        if config.NOTCH_FILTER and signal_length >= 100:  # 只在较长信号上应用陷波滤波
            try:
                meg_data = mne.filter.notch_filter(
                    meg_data,
                    info['sfreq'],  # 位置参数，不是关键字
                    freqs=config.NOTCH_FILTER,
                    verbose=False
                )
            except Exception as e:
                print(f"陷波滤波器错误: {e}")
    
        # 归一化
        if config.NORMALIZATION == "global":
            meg_data = (meg_data - np.mean(meg_data)) / (np.std(meg_data) + 1e-8)
        elif config.NORMALIZATION == "channel":
            for i in range(meg_data.shape[0]):
                channel_data = meg_data[i]
                if np.std(channel_data) > 1e-8:
                    meg_data[i] = (channel_data - np.mean(channel_data)) / (np.std(channel_data) + 1e-8)
    
        return meg_data

def create_data_loaders():
    """Create training, validation, and test data loaders."""
    # Create full dataset
    dataset = MASCMEGDataset(base_path=config.BASE_PATH)
    
    # Calculate split sizes
    train_size = int(len(dataset) * config.TRAIN_VAL_TEST_SPLIT[0])
    val_size = int(len(dataset) * config.TRAIN_VAL_TEST_SPLIT[1])
    test_size = len(dataset) - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.SEED)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader