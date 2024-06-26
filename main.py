import os
import random

import torch
import numpy as np

from time import time
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path
from prettytable import PrettyTable

from common.test import test_v2
from common.utils import early_stopping, print_dict
from common.config import parse_args
from common.dataset import CKGData
from common.dataset.build import build_loader

from modules.sampler import KGPolicy
from modules.recommender import MF

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# 手动清除 GPU 缓存
torch.cuda.empty_cache()


# 于训练基于 Matrix Factorization（MF）和 Knowledge Graph Attention Network（KGAT）的推荐模型的训练代码

# train_one_epoch 函数：用于训练一个 epoch 的推荐模型。
# 它接收训练数据、模型、优化器等作为输入，
# 并在每个批次上执行前向传播、反向传播和优化器更新操作。
def train_one_epoch(
        recommender,
        sampler,
        train_loader,
        recommender_optim,
        sampler_optim,
        adj_matrix,
        edge_matrix,
        train_data,
        cur_epoch,
        avg_reward,
):
    # 初始化损失和奖励
    loss, base_loss, reg_loss = 0, 0, 0
    epoch_reward = 0

    """Train one epoch"""
    # 使用 tqdm 函数迭代训练数据加载器
    tbar = tqdm(train_loader, ascii=True)
    num_batch = len(train_loader)
    for batch_data in tbar:

        tbar.set_description("Epoch {}".format(cur_epoch))

        # 将数据移到 GPU 上（如果可用）
        if torch.cuda.is_available():
            batch_data = {k: v.cuda(non_blocking=True) for k, v in batch_data.items()}

        """Train recommender using negtive item provided by sampler"""
        # 推荐器模型优化器梯度清零
        recommender_optim.zero_grad()

        # 获取批次数据
        neg = batch_data["neg_i_id"]
        pos = batch_data["pos_i_id"]
        users = batch_data["u_id"]

        # 使用采样器获取负样本
        selected_neg_items_list, _ = sampler(batch_data, adj_matrix, edge_matrix)
        selected_neg_items = selected_neg_items_list[-1, :]

        # 将负样本与训练集中的样本对齐
        train_set = train_data[users]
        in_train = torch.sum(
            selected_neg_items.unsqueeze(1) == train_set.long(), dim=1
        ).byte()
        selected_neg_items[in_train] = neg[in_train].to(torch.long)

        # 计算推荐器的损失
        base_loss_batch, reg_loss_batch = recommender(users, pos, selected_neg_items)
        loss_batch = base_loss_batch + reg_loss_batch

        # 反向传播和参数更新
        loss_batch.backward()
        recommender_optim.step()

        """Train sampler network"""
        # 采样器模型优化器梯度清零
        sampler_optim.zero_grad()

        # 使用采样器获取负样本和概率
        selected_neg_items_list, selected_neg_prob_list = sampler(
            batch_data, adj_matrix, edge_matrix
        )

        with torch.no_grad():
            reward_batch = recommender.get_reward(users, pos, selected_neg_items_list)

        epoch_reward += torch.sum(reward_batch)
        reward_batch -= avg_reward

        # 计算累积奖励
        batch_size = reward_batch.size(1)
        n = reward_batch.size(0) - 1
        R = torch.zeros(batch_size, device=reward_batch.device)
        reward = torch.zeros(reward_batch.size(), device=reward_batch.device)

        gamma = args_config.gamma

        for i, r in enumerate(reward_batch.flip(0)):
            R = r + gamma * R
            reward[n - i] = R

        # 计算采样器的损失
        reinforce_loss = -1 * torch.sum(reward_batch * selected_neg_prob_list)
        reinforce_loss.backward()
        sampler_optim.step()

        # 记录损失和奖励
        loss += loss_batch
        reg_loss += reg_loss_batch
        base_loss += base_loss_batch

    avg_reward = epoch_reward / num_batch
    train_res = PrettyTable()
    train_res.field_names = ["Epoch", "Loss", "BPR-Loss", "Regulation", "AVG-Reward"]
    train_res.add_row(
        [cur_epoch, loss.item(), base_loss.item(), reg_loss.item(), avg_reward.item()]
    )
    print(train_res)

    return loss, base_loss, reg_loss, avg_reward


# save_model 函数：用于保存训练好的模型。
def save_model(file_name, model, config):
    if not os.path.isdir(config.out_dir):
        os.mkdir(config.out_dir)

    model_file = Path(config.out_dir + file_name)
    model_file.touch(exist_ok=True)

    print("Saving model...")
    torch.save(model.state_dict(), model_file)


# build_sampler_graph 函数：
# 用于构建采样器的邻接矩阵和边矩阵。
# 在训练过程中，采样器会从知识图谱中采样负样本，这里构建的邻接矩阵和边矩阵用于存储采样器的采样结果。
def build_sampler_graph(n_nodes, edge_threshold, graph):
    adj_matrix = torch.zeros(n_nodes, edge_threshold * 2)
    edge_matrix = torch.zeros(n_nodes, edge_threshold)

    """sample neighbors for each node"""
    for node in tqdm(graph.nodes, ascii=True, desc="Build sampler matrix"):
        neighbors = list(graph.neighbors(node))
        if len(neighbors) >= edge_threshold:
            sampled_edge = random.sample(neighbors, edge_threshold)
            edges = deepcopy(sampled_edge)
        else:
            neg_id = random.sample(
                range(CKG.item_range[0], CKG.item_range[1] + 1),
                edge_threshold - len(neighbors),
            )
            node_id = [node] * (edge_threshold - len(neighbors))
            sampled_edge = neighbors + neg_id
            edges = neighbors + node_id

        """concatenate sampled edge with random edge"""
        sampled_edge += random.sample(
            range(CKG.item_range[0], CKG.item_range[1] + 1), edge_threshold
        )

        adj_matrix[node] = torch.tensor(sampled_edge, dtype=torch.long)
        edge_matrix[node] = torch.tensor(edges, dtype=torch.long)

    if torch.cuda.is_available():
        adj_matrix = adj_matrix.cuda().long()
        edge_matrix = edge_matrix.cuda().long()

    return adj_matrix, edge_matrix


# build_train_data 函数：用于构建训练数据。它将用户-物品交互数据整理成一个张量，用于训练模型。
# 函数将训练数据字典转换为一个可以直接用于训练神经网络的张量形式。
def build_train_data(train_mat):
    num_user = max(train_mat.keys()) + 1
    num_true = max([len(i) for i in train_mat.values()])
    # 计算了训练数据中的最大用户ID和最大正样本数量
    train_data = torch.zeros(num_user, num_true)
    # 一个全零张量，其大小为(num_user, num_true)，其中
    # num_user
    # 是用户的数量，num_true
    # 是训练集中最大用户的正样本数量。

    for i in train_mat.keys():
        true_list = train_mat[i]
        true_list += [-1] * (num_true - len(true_list))
        train_data[i] = torch.tensor(true_list, dtype=torch.long)
    # 函数遍历训练数据字典中的每个用户，将每个用户的正样本列表填充到对应行的张量中。如果某个用户的正样本数量小于
    # num_true，则用 - 1
    # 进行填充，以便保持张量的形状一致。
    return train_data


# train 函数：训练主函数。它构建了推荐模型和采样器，并在训练集上进行模型训练。
# 在每个 epoch 结束后，会在测试集上评估模型的性能，并输出评价指标
def train(train_loader, test_loader, graph, data_config, args_config):
    """build padded training set"""
    train_mat = graph.train_user_dict
    train_data = build_train_data(train_mat)

    if args_config.pretrain_r:
        print(
            "\nLoad model from {}".format(
                args_config.data_path + args_config.model_path
            )
        )

        # 使用了预训练模型  args_config.model_path
        paras = torch.load(args_config.data_path + args_config.model_path)
        # all_embed作用是？
        all_embed = torch.cat((paras["user_para"], paras["item_para"]))

        data_config["all_embed"] = all_embed

    # 使用MF推荐器,
    # 它接受数据配置和参数配置。
    # 它包括用户数量、物品数量、嵌入大小等信息。
    recommender = MF(data_config=data_config, args_config=args_config)

    # 基于知识图谱的动态负采样器模块 KGPolicy，
    sampler = KGPolicy(recommender, data_config, args_config)

    # 这段代码首先检查当前系统是否支持
    # CUDA（即GPU加速），如果支持，则将相关的张量和模型移动到GPU 上进行计算。
    if torch.cuda.is_available():
        train_data = train_data.long().cuda()
        sampler = sampler.cuda()
        recommender = recommender.cuda()

        print("\nSet sampler as: {}".format(str(sampler)))
        print("Set recommender as: {}\n".format(str(recommender)))

    recommender_optimer = torch.optim.Adam(recommender.parameters(), lr=args_config.rlr)
    sampler_optimer = torch.optim.Adam(sampler.parameters(), lr=args_config.slr)
    # 定义了一些变量用于记录训练过程中的指标和状态pre_loger精确率，loss_loger损失率
    # rec_loger召回率，ndcg_loger归一化折损累积增益（NDCG）指标，命中率（hit ratio）指标，cur_best_pre_0目前为止最佳的精确度值。
    # avg_reward平均奖励值
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step, cur_best_pre_0, avg_reward = 0, 0.0, 0
    t0 = time()

    for epoch in range(args_config.epoch):
        if epoch % args_config.adj_epoch == 0:
            """sample adjacency matrix"""
            adj_matrix, edge_matrix = build_sampler_graph(
                data_config["n_nodes"], args_config.edge_threshold, graph.ckg_graph
            )

        cur_epoch = epoch + 1
        loss, base_loss, reg_loss, avg_reward = train_one_epoch(
            recommender,
            sampler,
            train_loader,
            recommender_optimer,
            sampler_optimer,
            adj_matrix,
            edge_matrix,
            train_data,
            cur_epoch,
            avg_reward,
        )

        """Test"""
        if cur_epoch % args_config.show_step == 0:
            with torch.no_grad():
                ret = test_v2(recommender, args_config.Ks, graph)

            loss_loger.append(loss)
            rec_loger.append(ret["recall"])
            pre_loger.append(ret["precision"])
            ndcg_loger.append(ret["ndcg"])
            hit_loger.append(ret["hit_ratio"])
            hit_loger.append(ret["mrr"])

            print_dict(ret)

            cur_best_pre_0, stopping_step, should_stop = early_stopping(
                ret["recall"][0],
                cur_best_pre_0,
                stopping_step,
                expected_order="acc",
                flag_step=args_config.flag_step,
            )

            if should_stop:
                break

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = (
            "Best Iter=[%d]@[%.1f]\n recall=[%s] \n precision=[%s] \n hit=[%s] \n ndcg=[%s]"
            % (
                idx,
                time() - t0,
                "\t".join(["%.5f" % r for r in recs[idx]]),
                "\t".join(["%.5f" % r for r in pres[idx]]),
                "\t".join(["%.5f" % r for r in hit[idx]]),
                "\t".join(["%.5f" % r for r in ndcgs[idx]]),
            )
    )
    print(final_perf)


if __name__ == "__main__":
    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    """initialize args and dataset"""
    # 参数配置设置 更改parser文件
    args_config = parse_args()

    CKG = CKGData(args_config)

    """set the gpu id"""
    if torch.cuda.is_available():
        torch.cuda.set_device(args_config.gpu_id)

    data_config = {
        "n_users": CKG.n_users,
        "n_items": CKG.n_items,
        "n_relations": CKG.n_relations + 2,
        "n_entities": CKG.n_entities,
        "n_nodes": CKG.entity_range[1] + 1,
        "item_range": CKG.item_range,
    }

    print("\ncopying CKG graph for data_loader.. it might take a few minutes")
    # 内存不足无法进行深拷贝会报错
    graph = deepcopy(CKG)
    train_loader, test_loader = build_loader(args_config=args_config, graph=graph)

    train(
        train_loader=train_loader,
        test_loader=test_loader,
        graph=CKG,
        data_config=data_config,
        args_config=args_config,
    )
