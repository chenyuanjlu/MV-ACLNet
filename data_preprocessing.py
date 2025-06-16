import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split


def load_annotations(data_dir):
    # 路径设置
    normal_csv_path = os.path.join(data_dir, 'normal_annotations.csv')
    damaged_csv_path = os.path.join(data_dir, 'damaged_annotations.csv')

    # 文件检查
    if not os.path.exists(normal_csv_path):
        raise FileNotFoundError(f"Normal annotations file not found at {normal_csv_path}")
    if not os.path.exists(damaged_csv_path):
        raise FileNotFoundError(f"Damaged annotations file not found at {damaged_csv_path}")

    # 读取CSV文件
    normal_annotations = pd.read_csv(normal_csv_path)
    damaged_annotations = pd.read_csv(damaged_csv_path)

    # 添加标签
    normal_annotations['label'] = 0
    damaged_annotations['label'] = 1

    # 合并数据
    annotations = pd.concat([normal_annotations, damaged_annotations], ignore_index=True)
    return annotations

def split_dataset(annotations, train_ratio=0.9):
    """
    以 patient_id 为单位划分数据集，确保同一个病人的所有样本在同一个集合中。
    :param annotations: 包含所有样本的 DataFrame
    :param train_ratio: 训练集比例
    :return: train_annotations, val_annotations
    """
    # 按 patient_id 分组，并计算每个病人的标签（取第一个样本的标签）
    patient_groups = annotations.groupby('patient_id')
    patient_ids = list(patient_groups.groups.keys())  # 所有病人的 ID
    patient_labels = [group['label'].iloc[0] for _, group in patient_groups]  # 每个病人的标签

    # 使用 StratifiedShuffleSplit 对 patient_id 进行分层抽样
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=1 - train_ratio, random_state=42)

    # 获取训练集和验证集的 patient_id
    for train_idx, val_idx in stratified_split.split(patient_ids, patient_labels):
        train_patient_ids = [patient_ids[i] for i in train_idx]  # 训练集的 patient_id
        val_patient_ids = [patient_ids[i] for i in val_idx]  # 验证集的 patient_id

    # 根据 patient_id 划分数据集
    train_annotations = annotations[annotations['patient_id'].isin(train_patient_ids)]
    val_annotations = annotations[annotations['patient_id'].isin(val_patient_ids)]

    return train_annotations, val_annotations