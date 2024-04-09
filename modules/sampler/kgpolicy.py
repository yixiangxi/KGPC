import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric as geometric
import networkx as nx

from tqdm import tqdm


# 基于知识图谱的动态负采样器模块 KGPolicy，
# 其目的是根据知识图谱提供合格的负样本，用于训练推荐系统。

# 图卷积网络模块，用于在知识图谱上进行嵌入学习。
class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    Input: embedding matrix for knowledge graph entity and adjacency matrix
    Output: gcn embedding for kg entity
    图卷积网络
    输入：知识图谱实体的嵌入矩阵和邻接矩阵
    输出： kg实体的 GCN 嵌入
    """

    def __init__(self, in_channel, out_channel, config):
        super(GraphConv, self).__init__()
        self.config = config

        self.conv1 = geometric.nn.SAGEConv(in_channel[0], out_channel[0])
        self.conv2 = geometric.nn.SAGEConv(in_channel[1], out_channel[1])

    def forward(self, x, edge_indices):
        x = self.conv1(x, edge_indices)
        x = F.leaky_relu(x)
        x = F.dropout(x)

        x = self.conv2(x, edge_indices)
        x = F.dropout(x)
        x = F.normalize(x)

        return x


# 负采样器模块 KGPolicy，
# 它用于根据知识图谱提供合格的负样本。
# 它包含一个图卷积网络用于知识图谱嵌入学习，并初始化了实体嵌入矩阵。
class KGPolicy(nn.Module):
    """
    Dynamical negative item sampler based on Knowledge graph
    Input: user, postive item, knowledge graph embedding
    Ouput: qualified negative item
    基于知识图谱的动态负项采样器
    输入：用户、正面条目、知识图谱嵌入
    输出：合格的否定项
    """

    def __init__(self, dis, params, config):
        super(KGPolicy, self).__init__()
        self.params = params
        self.config = config
        self.dis = dis

        in_channel = eval(config.in_channel)
        out_channel = eval(config.out_channel)
        self.gcn = GraphConv(in_channel, out_channel, config)

        self.n_entities = params["n_nodes"]
        self.item_range = params["item_range"]
        self.input_channel = in_channel
        self.entity_embedding = self._initialize_weight(
            self.n_entities, self.input_channel
        )

    def _initialize_weight(self, n_entities, input_channel):
        """entities includes items and other entities in knowledge graph
            实体包括知识图谱中的项目和其他实体
        """
        if self.config.pretrain_s:
            kg_embedding = self.params["kg_embedding"]
            entity_embedding = nn.Parameter(kg_embedding)
        else:
            entity_embedding = nn.Parameter(
                torch.FloatTensor(n_entities, input_channel[0])
            )
            nn.init.xavier_uniform_(entity_embedding)

        if self.config.freeze_s:
            entity_embedding.requires_grad = False

        return entity_embedding

    # 前向传播方法，用于根据知识图谱生成合格的负样本。它执行了若干步的知识图谱采样，以生成一系列负样本。
    def forward(self, data_batch, adj_matrix, edge_matrix):
        users = data_batch["u_id"]
        pos = data_batch["pos_i_id"]

        self.edges = self.build_edge(edge_matrix)

        neg_list = torch.tensor([], dtype=torch.long, device=adj_matrix.device)
        prob_list = torch.tensor([], device=adj_matrix.device)

        k = self.config.k_step
        assert k > 0

        for _ in range(k):
            """sample candidate negative items based on knowledge graph
            基于知识图谱的候选负面项目样本"""

            one_hop, one_hop_logits = self.kg_step(pos, users, adj_matrix, step=1)

            candidate_neg, two_hop_logits = self.kg_step(
                one_hop, users, adj_matrix, step=2
            )
            candidate_neg = self.filter_entity(candidate_neg, self.item_range)
            good_neg, good_logits = self.prune_step(
                self.dis, candidate_neg, users, two_hop_logits
            )
            good_logits = good_logits + one_hop_logits

            neg_list = torch.cat([neg_list, good_neg.unsqueeze(0)])
            prob_list = torch.cat([prob_list, good_logits.unsqueeze(0)])

            pos = good_neg

        return neg_list, prob_list

    # 这是构建边的方法，它根据邻接矩阵构建边。
    def build_edge(self, adj_matrix):
        """build edges based on adj_matrix"""
        sample_edge = self.config.edge_threshold
        edge_matrix = adj_matrix

        n_node = edge_matrix.size(0)
        node_index = (
            torch.arange(n_node, device=edge_matrix.device)
            .unsqueeze(1)
            .repeat(1, sample_edge)
            .flatten()
        )
        neighbor_index = edge_matrix.flatten()
        edges = torch.cat((node_index.unsqueeze(1), neighbor_index.unsqueeze(1)), dim=1)
        return edges

    # 知识图谱采样的步骤，它根据图卷积网络的嵌入生成候选的负样本。
    def kg_step(self, pos, user, adj_matrix, step):
        x = self.entity_embedding
        edges = self.edges

        """knowledge graph embedding using gcn"""
        gcn_embedding = self.gcn(x, edges.t().contiguous())

        """use knowledge embedding to decide candidate negative items
        使用知识嵌入来决定候选负面项目"""
        u_e = gcn_embedding[user]
        u_e = u_e.unsqueeze(dim=2)
        pos_e = gcn_embedding[pos]
        pos_e = pos_e.unsqueeze(dim=1)

        one_hop = adj_matrix[pos]
        i_e = gcn_embedding[one_hop]

        p_entity = F.leaky_relu(pos_e * i_e)
        p = torch.matmul(p_entity, u_e)
        p = p.squeeze()
        logits = F.softmax(p, dim=1)

        """sample negative items based on logits
        基于对数的反向抽样"""
        batch_size = logits.size(0)
        if step == 1:
            nid = torch.argmax(logits, dim=1, keepdim=True)
        else:
            n = self.config.num_sample
            _, indices = torch.sort(logits, descending=True)
            nid = indices[:, :n]
        row_id = torch.arange(batch_size, device=logits.device).unsqueeze(1)

        candidate_neg = one_hop[row_id, nid].squeeze()
        candidate_logits = torch.log(logits[row_id, nid]).squeeze()

        return candidate_neg, candidate_logits

    # 一个修剪步骤，它根据用户与负样本的相似度获取最合适的负样本。
    @staticmethod
    def prune_step(dis, negs, users, logits):
        with torch.no_grad():
            ranking = dis.rank(users, negs)

        """get most qualified negative item based on user-neg similarity"""
        indices = torch.argmax(ranking, dim=1)

        batch_size = negs.size(0)
        row_id = torch.arange(batch_size, device=negs.device).unsqueeze(1)
        indices = indices.unsqueeze(1)

        good_neg = negs[row_id, indices].squeeze()
        goog_logits = logits[row_id, indices].squeeze()

        return good_neg, goog_logits

    # 筛选实体的方法，它确保生成的负样本在指定的实体范围内。
    @staticmethod
    def filter_entity(neg, item_range):
        random_neg = torch.randint(
            int(item_range[0]), int(item_range[1] + 1), neg.size(), device=neg.device
        )
        neg[neg > item_range[1]] = random_neg[neg > item_range[1]]
        neg[neg < item_range[0]] = random_neg[neg < item_range[0]]

        return neg
