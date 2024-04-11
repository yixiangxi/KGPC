import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# MF 类：这是主模型类。
# 在初始化时，它接受数据配置和参数配置。它包括用户数量、物品数量、嵌入大小等信息。
class MF(nn.Module):
    def __init__(self, data_config, args_config):
        super(MF, self).__init__()
        self.args_config = args_config  # 存储参数配置
        self.data_config = data_config  # 存储数据配置
        self.n_users = data_config["n_users"]  # 用户数量
        self.n_items = data_config["n_items"]  # 物品数量

        # 加上1表示所有索引值中最后一个将作为填充索引
        self.padding_idx = self.n_users + self.n_items

        self.emb_size = args_config.emb_size  # 嵌入大小
        self.regs = eval(args_config.regs)  # 正则化项

        self.all_embed = self._init_weight()  # 初始化嵌入权重

    def _init_weight(self):
        # 初始化嵌入权重
        all_embed = nn.Embedding(
            num_embeddings=self.n_users + self.n_items + 1,  # 加1用于填充索引
            embedding_dim=self.emb_size,
            padding_idx=self.padding_idx  # 设置填充索引
        )
        if self.args_config.pretrain_r:
            # 如果有预训练权重，加载预训练权重
            pretrained_weights = self.data_config["all_embed"]
            all_embed.weight.data.copy_(pretrained_weights)
        else:
            # 否则使用 Xavier 均匀初始化（填充索引除外）
            nn.init.xavier_uniform_(all_embed.weight.data)

        # 将padding index位置的向量初始化为0？？？存在问题
        with torch.no_grad():
            all_embed.weight[self.padding_idx].fill_(-1)
        return all_embed

    # forward
    # 方法：用于计算前向传播。接受用户索引、正样本索引和负样本索引作为输入，
    # 计算用户、正样本和负样本之间的分数，
    # 然后计算BPR损失和正则化损失。
    def forward(self, user, pos_item, neg_item):
        # 前向传播
        # 获取用户、正样本、负样本的嵌入
        u_e = self.all_embed(user)
        pos_e = self.all_embed(pos_item)
        neg_e = self.all_embed(neg_item)

        # 计算正则化损失
        reg_loss = self._l2_loss(u_e) + self._l2_loss(pos_e) + self._l2_loss(neg_e)
        reg_loss = self.regs * reg_loss

        # 计算用户和正负样本之间的分数
        pos_scores = torch.sum(u_e * pos_e, dim=1)
        neg_scores = torch.sum(u_e * neg_e, dim=1)

        # 计算 BPR 损失
        bpr_loss = torch.log(torch.sigmoid(pos_scores - neg_scores))
        bpr_loss = -torch.mean(bpr_loss)

        return bpr_loss, reg_loss

    # get_reward
    # 方法：用于计算正样本和负样本之间的奖励。
    def get_reward(self, user, pos_item, neg_item):
        # 获取奖励
        u_e = self.all_embed(user)
        pos_e = self.all_embed(pos_item)
        neg_e = self.all_embed(neg_item)

        neg_scores = torch.sum(u_e * neg_e, dim=-1)
        ij = torch.sum(neg_e * pos_e, dim=-1)

        reward = neg_scores + ij

        return reward

    # _l2_loss
    # 方法：用于计算L2正则化损失。
    @staticmethod
    def _l2_loss(t):
        # 计算 L2 正则化损失
        return torch.sum(t ** 2) / 2

    # inference
    # 方法：用于推断用户的兴趣。返回用户和物品之间的分数。
    def inference(self, users):
        # 推断用户兴趣
        user_embed, item_embed = torch.split(
            self.all_embed, [self.n_users, self.n_items], dim=0
        )
        user_embed = user_embed[users]
        prediction = torch.matmul(user_embed, item_embed.t())
        return prediction

    # rank
    # 方法：用于对用户和物品进行排名。
    def rank(self, users, items):
        # 使用嵌入层来获取用户和物品的嵌入
        u_e = self.all_embed(users)
        i_e = self.all_embed(items)

        u_e = u_e.unsqueeze(1)  # 增加维度以匹配物品维度
        ranking = torch.sum(u_e * i_e, dim=-1)  # 计算得分
        return ranking.squeeze()  # 移除维度

    def __str__(self):
        # 返回模型名称和嵌入大小的字符串表示
        return "recommender using BPRMF, embedding size {}".format(
            self.args_config.emb_size
        )
