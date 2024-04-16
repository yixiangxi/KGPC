import argparse


# data_path: 输入数据路径，默认为 "../Data/"。
# dataset: 选择数据集，默认为 "last-fm"。       修改为yelp2018
# emb_size: 嵌入向量的维度，默认为 64。
# regs: 用户和物品嵌入的正则化系数，默认为 "1e-5"。
# gpu_id: GPU 的 ID，默认为 0。
# k_neg: 列表中负样本的数量，默认为 1。
# slr: 采样器的学习率，默认为 0.0001。
# rlr: 推荐模型的学习率，默认为 0.0001。         修改为0.001
# edge_threshold: 过滤知识图谱的边的阈值，默认为 64。
# num_sample: 从 GCN 中采样的样本数量，默认为 32。
# k_step: 从当前正样本到目标样本的步数，默认为 2。
# in_channel: GCN 的输入通道，默认为 "[64, 32]"。
# out_channel: GCN 的输出通道，默认为 "[32, 64]"。
# pretrain_s: 是否加载预训练的采样器数据，默认为 False。
# batch_size: 训练的批量大小，默认为 1024。
# test_batch_size: 测试的批量大小，默认为 1024。
# num_threads: 线程数，默认为 4。
# epoch: 训练的轮数，默认为 400。  修改为1
# show_step: 测试步数，默认为 3。  修改为1
# adj_epoch: 每 _ 轮构建一次邻接矩阵，默认为 1。
# pretrain_r: 是否使用预训练模型，默认为 True。    修改为false
# freeze_s: 是否冻结推荐模型的参数，默认为 False。
# model_path: 预训练模型的路径，默认为 "model/best_fm.ckpt"。
# out_dir: 模型输出目录，默认为 "./weights/"。
# flag_step: 提前停止的步数，默认为 64。
# gamma: 奖励累积的 gamma 值，默认为 0.99。
# Ks: 评估时的 K 值列表，默认为 "[20, 40, 60, 80, 100]"。

def parse_args():
    parser = argparse.ArgumentParser(description="Run KGPolicy2.")
    # ------------------------- experimental settings specific for data set --------------------------------------------
    parser.add_argument(
        "--data_path", nargs="?", default="../Data/", help="Input data path."
    )
    parser.add_argument(
        "--dataset", nargs="?", default="bridge_all_train2", help="Choose a dataset."
    )
    parser.add_argument("--emb_size", type=int, default=64, help="Embedding size.")
    parser.add_argument(
        "--regs",
        nargs="?",
        default="1e-5",
        help="Regularization for user and item embeddings.",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument(
        "--k_neg", type=int, default=1, help="number of negative items in list"
    )

    # ------------------------- experimental settings specific for recommender -----------------------------------------
    parser.add_argument(
        "--slr", type=float, default=0.0001, help="Learning rate for sampler."
    )
    parser.add_argument(
        "--rlr", type=float, default=0.001, help="Learning rate recommender."
    )

    # ------------------------- experimental settings specific for sampler ---------------------------------------------
    parser.add_argument(
        "--edge_threshold",
        type=int,
        default=64,
        help="edge threshold to filter knowledge graph",
    )
    parser.add_argument(
        "--num_sample", type=int, default=32, help="number fo samples from gcn"
    )
    parser.add_argument(
        "--k_step", type=int, default=2, help="k step from current positive items"
    )
    parser.add_argument(
        "--in_channel", type=str, default="[64, 32]", help="input channels for gcn"
    )
    parser.add_argument(
        "--out_channel", type=str, default="[32, 64]", help="output channels for gcn"
    )
    parser.add_argument(
        "--pretrain_s",
        type=bool,
        default=False,
        help="load pretrained sampler data or not",
    )

    # ------------------------- experimental settings specific for recommender --------------------------------------------
    parser.add_argument(
        "--batch_size", type=int, default=1024, help="batch size for training."
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=1024, help="batch size for test"
    )
    parser.add_argument("--num_threads", type=int, default=4, help="number of threads.")
    parser.add_argument("--epoch", type=int, default=5, help="Number of epoch.")
    parser.add_argument("--show_step", type=int, default=1, help="test step.")
    parser.add_argument(
        "--adj_epoch", type=int, default=1, help="build adj matrix per _ epoch"
    )
    parser.add_argument(
        "--pretrain_r", type=bool, default=False, help="use pretrained model or not"
    )
    parser.add_argument(
        "--freeze_s",
        type=bool,
        default=False,
        help="freeze parameters of recommender or not",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="model/best_fm.ckpt",
        help="path for pretrain model",
    )
    parser.add_argument(
        "--out_dir", type=str, default="./weights/", help="output directory for model"
    )
    parser.add_argument("--flag_step", type=int, default=64, help="early stop steps")
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="gamma for reward accumulation"
    )

    # ------------------------- experimental settings specific f
    # or testing ---------------------------------------------
    parser.add_argument(
        "--Ks", nargs="?", default="[20, 40, 60, 80, 100]", help="evaluate K list"
    )

    return parser.parse_args()
