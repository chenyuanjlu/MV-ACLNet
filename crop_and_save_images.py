import os
from PIL import Image
import pandas as pd
from tqdm import tqdm
from torchvision.transforms import functional as F  # PyTorch transforms 支持

# 调用您现有的函数
from dataset import KneeInjuryDataset  # 替换为实际文件名或模块路径
from data_preprocessing import load_annotations


def crop_and_save_using_dataset(annotations, data_dir, output_dir):
    """
    使用 `KneeInjuryDataset` 类裁剪图像，并按照原文件结构保存，同时将裁剪后的图像压缩为 224×224 大小。

    :param annotations: pd.DataFrame, 包含图像路径和边界框信息的注释文件
    :param data_dir: str, 原始数据目录
    :param output_dir: str, 裁剪后图像保存的目标目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 创建 Dataset 实例
    dataset = KneeInjuryDataset(annotations, data_dir)

    for idx in tqdm(range(len(dataset)), desc="Processing images"):
        # 获取裁剪后的图像和相关信息
        cropped_image, label, patient_id = dataset[idx]

        # 将图像压缩为 224×224
        resized_image = cropped_image.resize((224, 224), Image.Resampling.LANCZOS)

        # 确定保存路径
        category = 'normal' if label == 0 else 'damaged'
        output_patient_dir = os.path.join(output_dir, category, patient_id)
        os.makedirs(output_patient_dir, exist_ok=True)

        # 获取图像名称
        image_name = annotations.iloc[idx]['image_name']
        output_path = os.path.join(output_patient_dir, image_name)

        # 保存压缩后的图像
        resized_image.save(output_path, format='JPEG')

    print(f"All cropped and resized images have been saved to {output_dir}")


if __name__ == "__main__":
    # 数据目录和注释文件路径
    data_dir = "train_dataset"  # 原始数据目录
    output_dir = "outputs"  # 裁剪后保存目录
    annotations_path = "train_dataset"  # 注释文件路径

    # 加载注释数据
    annotations = load_annotations(data_dir)  # 使用现有的 `load_annotations` 函数加载注释

    # 裁剪图像并保存
    crop_and_save_using_dataset(annotations, data_dir, output_dir)
