import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch_geometric as geometric


# GraphConv 模块：该模块定义了一个图卷积层，
# 使用了 PyTorch Geometric 中的 Graph Attention Network（图注意力网络）。
# 它接受输入通道和输出通道，并执行图卷积操作，然后进行了丢弃和归一化处理。
class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    embed CKG and using its embedding to calculate prediction score
    图卷积网络
    嵌入 CKG 并利用其嵌入计算预测得分
    """

    def __init__(self, in_channel, out_channel):
        super(GraphConv, self).__init__()
        self.in_channel = in_channel  # 输入通道数
        self.out_channel = out_channel  # 输出通道数
        # 使用 GATConv 进行图卷积
        self.conv1 = geometric.nn.GATConv(in_channel, out_channel)
        self.dropout = nn.Dropout(p=0.5)  # dropout 层

    def forward(self, x, edge_indices):
        x = self.conv1(x, edge_indices)  # 图卷积操作
        x = self.dropout(x)  # dropout
        x = F.normalize(x)  # 归一化
        return x




# KGAT 类：这是主要的模型类。
# 它在初始化时使用了数据配置和模型参数。
# 它设置了图卷积层，并初始化了嵌入。
class KGAT(nn.Module):
    def __init__(self, data_config, args_config):
        super(KGAT, self).__init__()
        self.args_config = args_config  # 模型参数配置
        self.data_config = data_config  # 数据配置
        self.n_users = data_config["n_users"]  # 用户数量
        self.n_items = data_config["n_items"]  # 物品数量
        self.n_nodes = data_config["n_nodes"]  # 节点总数

        """set input and output channel manually"""
        # 手动设置输入和输出通道数
        input_channel = 64
        output_channel = 64
        self.gcn = GraphConv(input_channel, output_channel)  # 初始化图卷积层

        self.emb_size = args_config.emb_size  # 嵌入大小
        self.regs = eval(args_config.regs)  # 正则化项

        self.all_embed = self._init_weight()  # 初始化嵌入权重

    # _init_weight：该函数使用
    # Xavier
    # 均匀初始化方法初始化了嵌入。在配置中指定的情况下，它还可以加载预训练的嵌入。
    def _init_weight(self):
        all_embed = nn.Parameter(
            torch.FloatTensor(self.n_nodes, self.emb_size), requires_grad=True
        )
        ui = self.n_users + self.n_items

        if self.args_config.pretrain_r:
            nn.init.xavier_uniform_(all_embed)  # 使用 Xavier 均匀初始化
            all_embed.data[:ui] = self.data_config["all_embed"]  # 加载预训练嵌入
        else:
            nn.init.xavier_uniform_(all_embed)  # 使用 Xavier 均匀初始化

        return all_embed

    # build_edge：该函数根据邻接矩阵构建了边。它根据阈值对边进行采样，并相应地构建边。
    def build_edge(self, adj_matrix):
        """build edges based on adj_matrix"""
        sample_edge = self.args_config.edge_threshold
        edge_matrix = adj_matrix

        n_node = edge_matrix.size(0)
        # 构建边
        node_index = (
            torch.arange(n_node, device=edge_matrix.device)
            .unsqueeze(1)
            .repeat(1, sample_edge)
            .flatten()
        )
        neighbor_index = edge_matrix.flatten()
        edges = torch.cat((node_index.unsqueeze(1), neighbor_index.unsqueeze(1)), dim=1)
        return edges

    # forward：该函数接受用户索引、正样本索引、负样本索引和边矩阵作为输入。
    # 它使用初始化的嵌入和图卷积层计算用户、正样本和负样本的嵌入。
    # 然后，它计算了正样本和负样本的得分，并计算了贝叶斯个性化排名（BPR）损失以及正则化损失。
    def forward(self, user, pos_item, neg_item, edges_matrix):
        u_e, pos_e, neg_e = (
            self.all_embed[user],  # 用户嵌入
            self.all_embed[pos_item],  # 正样本嵌入
            self.all_embed[neg_item],  # 负样本嵌入
        )

        edges = self.build_edge(edges_matrix)  # 构建边
        x = self.all_embed
        gcn_embedding = self.gcn(x, edges.t().contiguous())  # 进行图卷积

        u_e_, pos_e_, neg_e_ = (
            gcn_embedding[user],  # 图卷积后的用户嵌入
            gcn_embedding[pos_item],  # 图卷积后的正样本嵌入
            gcn_embedding[neg_item],  # 图卷积后的负样本嵌入
        )

        u_e = torch.cat([u_e, u_e_], dim=1)  # 连接用户嵌入
        pos_e = torch.cat([pos_e, pos_e_], dim=1)  # 连接正样本嵌入
        neg_e = torch.cat([neg_e, neg_e_], dim=1)  # 连接负样本嵌入

        pos_scores = torch.sum(u_e * pos_e, dim=1)  # 计算正样本分数
        neg_scores = torch.sum(u_e * neg_e, dim=1)  # 计算负样本分数

        # 定义目标函数，包括：
        # ... (1) BPR 损失
        bpr_loss = torch.log(torch.sigmoid(pos_scores - neg_scores))  # BPR 损失
        bpr_loss = -torch.mean(bpr_loss)

        # ... (2) 嵌入损失
        reg_loss = self._l2_loss(u_e) + self._l2_loss(pos_e) + self._l2_loss(neg_e)
        reg_loss = self.regs * reg_loss

        loss = bpr_loss + reg_loss  # 总损失

        return loss, bpr_loss, reg_loss

    # get_reward：该函数基于得分计算了正样本和负样本的奖励
    def get_reward(self, users, pos_items, neg_items):
        u_e = self.all_embed[users]
        pos_e = self.all_embed[pos_items]
        neg_e = self.all_embed[neg_items]

        neg_scores = torch.sum(u_e * neg_e, dim=1)
        ij = torch.sum(neg_e * pos_e, dim=1)
        reward = neg_scores + ij

        return reward

    # _l2_loss：该函数计算了
    # L2
    # 正则化损失。
    def _l2_loss(self, t):
        return torch.sum(t ** 2) / 2

    # inference：该函数通过执行用户嵌入和物品嵌入的矩阵乘法来计算用户的预测。
    def inference(self, users):
        num_entity = self.n_nodes - self.n_users - self.n_items
        user_embed, item_embed, _ = torch.split(
            self.all_embed, [self.n_users, self.n_items, num_entity], dim=0
        )

        user_embed = user_embed[users]
        prediction = torch.matmul(user_embed, item_embed.t())
        return prediction

    # rank：该函数计算了用户和物品的排名得分。
    def rank(self, users, items):
        u_e = self.all_embed[users]
        i_e = self.all_embed[items]

        u_e = u_e.unsqueeze(dim=1)
        ranking = torch.sum(u_e * i_e, dim=2)
        ranking = ranking.squeeze()

        return ranking

    def __str__(self):
        return "recommender using KGAT, embedding size {}".format(
            self.args_config.emb_size
        )
