#!/usr/local/bin/bash
__author__ = "xiangwang"
import os
import re
# 一些基本的工具，可以用于处理各种数据处理和机器学习任务中常见的操作，
# 例如文件读取、参数冻结、早停策略等。

def txt2list(file_src):
    """读取文本文件并将其内容转换为列表形式

    Args:
        file_src (str): 要读取的文本文件路径

    Returns:
        list: 包含文本文件内容的列表，每行文本作为列表的一个元素
    """
    orig_file = open(file_src, "r")
    lines = orig_file.readlines()
    return lines


def ensure_dir(dir_path):
    """确保目录存在，若不存在则创建

    Args:
        dir_path (str): 要确保存在的目录路径
    """
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)


def uni2str(unicode_str):
    """将Unicode字符串转换为ASCII编码的字符串，并移除换行符和空格

    Args:
        unicode_str (str): Unicode字符串

    Returns:
        str: ASCII编码的字符串
    """
    return str(unicode_str.encode("ascii", "ignore")).replace("\n", "").strip()


def has_numbers(input_string):
    """检查字符串中是否包含数字

    Args:
        input_string (str): 输入字符串

    Returns:
        bool: 如果字符串中包含数字，则返回True，否则返回False
    """
    return bool(re.search(r"\d", input_string))


def del_multichar(input_string, chars):
    """删除字符串中指定的多个字符

    Args:
        input_string (str): 输入字符串
        chars (str): 要删除的字符列表或字符串

    Returns:
        str: 删除指定字符后的字符串
    """
    for ch in chars:
        input_string = input_string.replace(ch, "")
    return input_string


def merge_two_dicts(x, y):
    """合并两个字典

    Args:
        x (dict): 第一个字典
        y (dict): 第二个字典

    Returns:
        dict: 合并后的字典
    """
    z = x.copy()  # 复制第一个字典的键值对
    z.update(y)  # 将第二个字典的键值对合并到第一个字典中
    return z


def early_stopping(
    log_value, best_value, stopping_step, expected_order="acc", flag_step=100
):
    """早停策略函数

    Args:
        log_value (float): 当前日志值
        best_value (float): 最佳日志值
        stopping_step (int): 停止步数
        expected_order (str, optional): 期望的日志值顺序，可选值为'acc'（递增）和'dec'（递减）。默认为'acc'。
        flag_step (int, optional): 触发早停的步数阈值。默认为100。

    Returns:
        tuple: 包含更新后的最佳日志值、停止步数和一个布尔值，表示是否应该停止训练
    """
    # 早停策略：
    assert expected_order in ["acc", "dec"]

    if (expected_order == "acc" and log_value >= best_value) or (
        expected_order == "dec" and log_value <= best_value
    ):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print(
            "Early stopping is trigger at step: {} log:{}".format(flag_step, log_value)
        )
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop


def freeze(model):
    """冻结模型参数

    Args:
        model (torch.nn.Module): 要冻结参数的模型

    Returns:
        torch.nn.Module: 冻结参数后的模型
    """
    for param in model.parameters():
        param.requires_grad = False
    return model


def unfreeze(model):
    """解冻模型参数

    Args:
        model (torch.nn.Module): 要解冻参数的模型

    Returns:
        torch.nn.Module: 解冻参数后的模型
    """
    for param in model.parameters():
        param.requires_grad = True
    return model


def print_dict(dic):
    """以指定格式打印字典

    Args:
        dic (dict): 要打印的字典

    Example:
        dic = {"a": 1, "b": 2}
        print_dict(dic)
        # 输出：
        # "a": 1
        # "b": 2
    """
    print("\n".join("{:10s}: {}".format(key, values) for key, values in dic.items()))
