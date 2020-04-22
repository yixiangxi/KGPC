import numpy as np
import torch
from torch.utils.data import Dataset
import random
import scipy.sparse as sp

from time import time

# 这段代码定义了两个数据生成器类：
# TrainGenerator用于生成训练样本，TestGenerator用于生成测试样本。
# 这些生成器类实现了torch.utils.data.Dataset的接口，可以被PyTorch的DataLoader用于批量加载数据。
class TrainGenerator(Dataset):
    def __init__(self, args_config, graph):
        """
        训练数据生成器类，用于生成训练样本。

        Args:
            args_config (argparse.Namespace): 参数配置对象
            graph (CKGData): 知识图谱数据对象
        """
        self.args_config = args_config
        self.graph = graph

        self.user_dict = graph.train_user_dict
        self.exist_users = list(graph.exist_users)
        self.low_item_index = graph.item_range[0]
        self.high_item_index = graph.item_range[1]

    def __len__(self):
        """返回数据集长度，即训练样本数量"""
        return self.graph.n_train

    def __getitem__(self, index):
        """获取指定索引的训练样本"""
        out_dict = {}

        user_dict = self.user_dict
        # 随机选择一个用户。
        u_id = random.sample(self.exist_users, 1)[0]
        out_dict["u_id"] = u_id

        # 随机选择一个正样本物品。
        pos_items = user_dict[u_id]
        n_pos_items = len(user_dict[u_id])

        pos_idx = np.random.randint(low=0, high=n_pos_items, size=1)[0]
        pos_i_id = pos_items[pos_idx]

        out_dict["pos_i_id"] = pos_i_id

        neg_i_id = self.get_random_neg(pos_items, [])
        out_dict["neg_i_id"] = neg_i_id

        return out_dict

    def get_random_neg(self, pos_items, selected_items):
        """
        获取随机负样本物品。

        Args:
            pos_items (list): 用户的正样本物品列表
            selected_items (list): 已选的物品列表

        Returns:
            neg_i_id (int): 负样本物品ID
        """
        while True:
            neg_i_id = np.random.randint(
                low=self.low_item_index, high=self.high_item_index, size=1
            )[0]

            if neg_i_id not in pos_items and neg_i_id not in selected_items:
                break
        return neg_i_id


class TestGenerator(Dataset):
    def __init__(self, args_config, graph):
        """
        测试数据生成器类，用于生成测试样本。

        Args:
            args_config (argparse.Namespace): 参数配置对象
            graph (CKGData): 知识图谱数据对象
        """
        self.args_config = args_config
        self.users_to_test = list(graph.test_user_dict.keys())

    def __len__(self):
        """返回数据集长度，即测试用户数量"""
        return len(self.users_to_test)

    def __getitem__(self, index):
        """获取指定索引的测试样本"""
        batch_data = {}

        u_id = self.users_to_test[index]
        batch_data["u_id"] = u_id

        return batch_data
