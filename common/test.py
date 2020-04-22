import torch
import numpy as np
from tqdm import tqdm

# 这段代码实现了一个测试函数 test_v2，用于评估推荐模型的性能。主要功能包括：

# 获取得分矩阵（get_score）：
#
# 输入推荐模型、用户数、物品数、训练集字典以及当前批次的起始和结束索引。
# 利用模型参数计算用户嵌入和物品嵌入，并计算得分矩阵。
# 对于训练集中的正样本，将其对应的得分设为一个较小的负值，以排除已知的正样本。
# 计算NDCG指标（cal_ndcg）：
#
# 输入top-k推荐列表、测试集中的正样本集合、正样本总数以及k值。
# 计算DCG和IDCG，然后计算NDCG指标。
# 评估函数（test_v2）：
#
# 输入推荐模型、评估的top-k值列表、知识图谱对象以及每个批次的数量。
# 初始化结果字典，包括precision、recall、ndcg和hit_ratio。
# 将用户划分为多个批次，每个批次计算一部分用户的评估指标。
# 对于每个批次中的每个用户：
# 获取得分矩阵，并找到top-k推荐列表。
# 计算precision、recall、ndcg和hit_ratio。
# 将每个批次的指标值进行累加，并计算平均值。
# 该函数主要用于在测试集上评估推荐模型的性能，并返回各个指标的平均值。
def get_score(model, n_users, n_items, train_user_dict, s, t):
    """
    获取推荐模型在指定范围内用户的得分矩阵。

    Args:
        model (MF): 推荐模型对象
        n_users (int): 用户数量
        n_items (int): 物品数量
        train_user_dict (dict): 训练集用户字典，包含每个用户的正样本物品列表
        s (int): 当前批次的起始索引
        t (int): 当前批次的结束索引

    Returns:
        score_matrix (torch.Tensor): 得分矩阵，形状为 (t-s) x n_items
    """

    u_e, i_e = torch.chunk(model.all_embed, 2, dim=0)#删除数据进行的修改，确保矩阵相对应

    #u_e, i_e = torch.split(model.all_embed, [n_users, n_items])  # 切分用户嵌入和物品嵌入
    #u_e, i_e = torch.split(model.all_embed, [n_users + n_items])
    u_e = u_e[s:t, :]  # 获取当前批次的用户嵌入

    score_matrix = torch.matmul(u_e, i_e.t())  # 计算得分矩阵
    for u in range(s, t):
        pos = train_user_dict[u]  # 获取当前用户的正样本物品列表
        idx = pos.index(-1) if -1 in pos else len(pos)  # 找到正样本中最后一个-1的索引，若不存在则为正样本长度
        score_matrix[u - s][pos[:idx] - n_users] = -1e5  # 将正样本对应位置的得分设为一个较小的负值，排除已知的正样本

    return score_matrix


def cal_ndcg(topk, test_set, num_pos, k):
    """
    计算NDCG指标。

    Args:
        topk (list): top-k推荐列表
        test_set (set): 测试集中的正样本物品集合
        num_pos (int): 正样本总数
        k (int): top-k值

    Returns:
        ndcg (float): NDCG指标值
    """
    n = min(num_pos, k)  # 计算NDCG的分母，取k和正样本数中的较小值
    nrange = np.arange(n) + 2  # 生成从2到n的序列
    idcg = np.sum(1 / np.log2(nrange))  # 计算IDCG

    dcg = 0
    for i, s in enumerate(topk):
        if s in test_set:  # 若推荐物品在测试集的正样本中
            dcg += 1 / np.log2(i + 2)  # 累加DCG值

    ndcg = dcg / idcg  # 计算NDCG指标

    return ndcg


def test_v2(model, ks, ckg, n_batchs=4):
    """
    在测试集上评估推荐模型的性能。

    Args:
        model (MF): 推荐模型对象
        ks (str): 评估的top-k值列表，格式为字符串
        ckg (CKGData): 知识图谱数据对象
        n_batchs (int): 批次数量，默认为4

    Returns:
        result (dict): 评估结果字典，包含precision、recall、ndcg和hit_ratio
    """
    ks = eval(ks)  # 将字符串转换为列表
    train_user_dict, test_user_dict = ckg.train_user_dict, ckg.test_user_dict  # 获取训练集和测试集用户字典
    n_users = ckg.n_users  # 获取用户数量
    n_items = ckg.n_items  # 获取物品数量
    n_test_users = len(test_user_dict)  # 获取测试集用户数量

    n_k = len(ks)  # 获取评估的top-k值数量
    result = {
        "precision": np.zeros(n_k),
        "recall": np.zeros(n_k),
        "ndcg": np.zeros(n_k),
        "hit_ratio": np.zeros(n_k),
    }  # 初始化结果字典

    batch_size = n_users // n_batchs  # 计算每个批次的用户数量
    for batch_id in tqdm(range(n_batchs), ascii=True, desc="Evaluate"):
        s = batch_size * batch_id  # 当前批次的起始索引
        t = batch_size * (batch_id + 1)  # 当前批次的结束索引
        if t > n_users:
            t = n_users
        if s == t:
            break

        score_matrix = get_score(model, n_users, n_items, train_user_dict, s, t)  # 获取得分矩阵
        for i, k in enumerate(ks):
            precision, recall, ndcg, hr = 0, 0, 0, 0  # 初始化评估指标值
            _, topk_index = torch.topk(score_matrix, k)  # 获取top-k索引
            topk_index = topk_index.cpu().numpy() + n_users  # 将索引转换为numpy数组，并加上用户数量，得到物品ID

            for u in range(s, t):
                gt_pos = test_user_dict[u]  # 获取当前用户的正样本物品列表
                topk = topk_index[u - s]  # 获取当前用户的top-k推荐列表
                num_pos = len(gt_pos)  # 获取当前用户的正样本数量

                topk_set = set(topk)  # 转换为集合方便计算
                test_set = set(gt_pos)  # 转换为集合方便计算
                num_hit = len(topk_set & test_set)  # 计算推荐列表和测试集正样本的交集数量

                precision += num_hit / k  # 计算precision
                recall += num_hit / num_pos  # 计算recall
                hr += 1 if num_hit > 0 else 0  # 计算hit_ratio

                ndcg += cal_ndcg(topk, test_set, num_pos, k)  # 计算NDCG

            result["precision"][i] += precision / n_test_users  # 累加precision
            result["recall"][i] += recall / n_test_users  # 累加recall
            result["ndcg"][i] += ndcg / n_test_users  # 累加NDCG
            result["hit_ratio"][i] += hr / n_test_users  # 累加hit_ratio

    return result  # 返回评估结果字典
