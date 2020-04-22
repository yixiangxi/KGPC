from torch.utils.data import DataLoader
from common.dataset.dataset import TrainGenerator, TestGenerator

# 定义了一个函数build_loader，用于构建训练和测试数据加载器。
# 该函数接收参数配置对象和知识图谱数据对象，并返回训练数据加载器和测试数据加载器。
def build_loader(args_config, graph):
    """
    构建训练和测试数据加载器。

    Args:
        args_config (argparse.Namespace): 参数配置对象
        graph (CKGData): 知识图谱数据对象

    Returns:
        train_loader (DataLoader): 训练数据加载器
        test_loader (DataLoader): 测试数据加载器
    """
    # 创建训练数据生成器和加载器
    train_generator = TrainGenerator(args_config=args_config, graph=graph)
    train_loader = DataLoader(
        train_generator,
        batch_size=args_config.batch_size,
        shuffle=True,
        num_workers=args_config.num_threads,
    )

    # 创建测试数据生成器和加载器
    test_generator = TestGenerator(args_config=args_config, graph=graph)
    test_loader = DataLoader(
        test_generator,
        batch_size=args_config.test_batch_size,
        shuffle=False,
        num_workers=args_config.num_threads,
    )

    return train_loader, test_loader
