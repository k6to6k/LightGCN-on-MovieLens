import os
import json

# Remove unnecessary import
import torch
import random
import numpy as np
from datetime import datetime

class Config:
    """
    LightGCN模型的配置类
    包含所有模型、训练、评估相关的超参数
    """
    def __init__(self):
        # 实验控制参数
        self.seed = 42 # 随机种子，确保结果可复现
        self.use_cuda = torch.cuda.is_available() # 是否使用CUDA

        # 数据相关参数
        self.data_path = "./data.pt" # 处理后的数据文件路径
        self.raw_data_path = "./ml-1m" # 原始数据文件路径

        # 模型结构参数
        self.embed_dim = 64 # 嵌入维度，越大表达能力越强但也更容易过拟合
        self.n_layers = 3 # LightGCN 图卷积层数，越多越深表达能力越强但也更容易过拟合，通常2-4层最佳

        # 训练相关参数
        self.lr = 0.001 # 学习率，影响收敛速度和稳定性
        self.weight_decay = 1e-4 # L2正则化系数，驯服模型的关键
        self.batch_size = 4096 # 批次大小，越大训练越稳定但需要更多内存
        self.epochs = 50 # 训练轮数，越大模型拟合能力越强但容易过拟合
        self.early_stop_patience = 10 # 早停轮数，如果验证集性能不再提升，则停止训练
        self.min_delta = 0.001 # 最小变化阈值，用于早停
        self.eval_freq = 5 # 评估频率，每多少轮评估一次

        # 评估相关参数
        self.k = 20 # 推荐列表长度
        self.batch_size_eval = 256 # 评估批次大小，防止内存溢出

        # 模型保存相关参数
        self.checkpoints_dir = "./checkpoints" # 检查点保存目录
        self.save_model_freq = 10 # 每训练多少个epoch保存一次模型
        self.best_model_path = "./best_model.pt" # 最佳模型保存路径

        # 结果保存相关
        self.results_dir = "./results" # 结果保存目录
        self.experiment_name = f"lightgcn_{datetime.now().strftime('%Y%m%d_%H%M%S')}" # 实验名称

    def set_seed(self):
        """
        设置随机种子，确保结果可复现
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.use_cuda:
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            # 下面两行设置可能会降低性能，但可以提高结果的确定性
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def get_device(self):
        """
        获取计算设备
            
        返回:
            torch.device: 计算设备(CPU或GPU)
        """
        return torch.device('cuda' if self.use_cuda else 'cpu')

    def to_dict(self):
        """
        将配置转换为字典
        
        返回:
            dict: 包含所有配置项的字典
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def from_dict(self, config_dict):
        """
        从字典加载配置
        
        参数:
            config_dict: 包含配置项的字典
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def __str__(self):
        """
        将配置转换为格式化字符串
        """
        config_str = "Model Configuration:\n"
        for key, value in sorted(self.to_dict().items()):
            config_str += f"{key}: {value}\n"
        return config_str

def save_config(config, path):
    """
    保存配置到JSON文件
    
    参数:
        config: Config类的实例
        path: 保存路径
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config.to_dict(), f, indent=2)

def load_config(path, config_class=Config):
    """
    从JSON文件加载配置
    
    参数:
        path: 配置文件路径
        config_class: 配置类，默认为基础Config类
        
    返回:
        Config类的实例
    """
    config = config_class()
    with open(path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    return config.from_dict(config_dict)

def print_config(config):
    """
    打印配置信息
    
    参数:
        config: Config类的实例
    """
    print(str(config))


# 复现论文的配置
class PaperReproConfig(Config):
    """
    用于复现LightGCN原始论文结果的配置
    """
    def __init__(self):
        super().__init__()
        # 论文中使用的超参数
        self.embed_dim = 64
        self.n_layers = 3
        self.lr = 0.001
        self.weight_decay = 0.0001  # 论文使用的正则化参数
        self.batch_size = 2048
        self.epochs = 1000  # 论文中训练更多轮次
        self.early_stop_patience = 20


# 一个针对小数据集快速实验的配置
class QuickTestConfig(Config):
    """
    用于快速测试代码功能的配置
    """
    def __init__(self):
        super().__init__()
        # 减小参数加快训练速度
        self.embed_dim = 16
        self.n_layers = 2
        self.batch_size = 1024
        self.epochs = 10
        self.eval_freq = 2
        self.early_stop_patience = 3


# 如果作为主模块运行，则打印默认配置
if __name__ == "__main__":
    config = Config()
    print_config(config)

    # 测试保存和加载功能
    save_path = "./test_config.json"
    save_config(config, save_path)
    loaded_config = load_config(save_path)
    print("\nLoaded configuration:")
    print_config(loaded_config)

    # 如果不需要测试文件，删除它
    if os.path.exists(save_path):
        os.remove(save_path)