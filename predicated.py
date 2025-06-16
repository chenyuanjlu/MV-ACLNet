import os
import torch
import timm
import pandas as pd
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedShuffleSplit

# 加载注释数据
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

# 数据划分函数：返回训练集和测试集，并添加划分信息
def split_dataset(annotations, train_ratio=0.9):
    # 初始化 StratifiedShuffleSplit 进行分层抽样
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=1 - train_ratio, random_state=42)

    # 进行一次划分：训练集和测试集
    for train_index, val_index in stratified_split.split(annotations, annotations['label']):
        train_annotations = annotations.iloc[train_index]
        val_annotations = annotations.iloc[val_index]

    # 标记每个数据点是训练集还是测试集
    annotations['split'] = 'train'
    annotations.loc[val_annotations.index, 'split'] = 'test'

    return annotations

# 数据集类
class KneeInjuryDataset(Dataset):
    def __init__(self, annotations, data_dir, transform=None):
        self.annotations = annotations
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        image_name = row['image_name']
        label = row['label']
        patient_id = row['patient_id']  # 从CSV文件中获取病人ID
        split = row['split']  # 获取该样本属于训练集还是测试集

        # 根据标签确定图像文件夹（正常/损伤）
        category = 'normal' if label == 0 else 'damaged'

        # 病人文件夹路径
        patient_dir = os.path.join(self.data_dir, category, patient_id)

        # 尝试在病人文件夹中直接查找图像
        img_path = os.path.join(patient_dir, image_name)

        if not os.path.exists(img_path):
            # 如果病人文件夹中没有图像，则查找病人文件夹中的子文件夹
            subfolders = [f.path for f in os.scandir(patient_dir) if f.is_dir()]

            img_path = None
            for subfolder in subfolders:
                possible_img_path = os.path.join(subfolder, image_name)
                if os.path.exists(possible_img_path):
                    img_path = possible_img_path
                    break

            if img_path is None:
                raise FileNotFoundError(f"Image {image_name} not found for patient {patient_id}")

        # 打开图像
        image = Image.open(img_path).convert('RGB')

        # 获取边界框信息
        bbox_x = row['bbox_x']
        bbox_y = row['bbox_y']
        bbox_width = row['bbox_width']
        bbox_height = row['bbox_height']

        # 裁剪图像：根据边界框裁剪出损伤区域
        cropped_image = image.crop((bbox_x, bbox_y, bbox_x + bbox_width, bbox_y + bbox_height))

        if self.transform:
            cropped_image = self.transform(cropped_image)

        return cropped_image, label, patient_id, split  # 返回split信息

# 预测得分函数
def predict_scores(model, dataloader, device):
    model.eval()
    patient_scores = defaultdict(list)
    patient_labels = {}
    patient_splits = {}  # 用于保存每个样本的划分信息

    with torch.no_grad():
        for images, labels, patient_ids, splits in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            for pid, prob, label, split in zip(patient_ids, probs.cpu().numpy(), labels.cpu().numpy(), splits):
                pid_str = str(pid)  # 确保pid为字符串
                patient_scores[pid_str].append(prob)
                patient_labels[pid_str] = label
                patient_splits[pid_str] = split  # 保存划分信息

    final_patient_ids = []
    final_predicted_scores = []
    final_true_labels = []
    final_splits = []  # 保存每个样本的划分信息

    for pid in patient_scores:
        scores = patient_scores[pid]
        average_score = sum(scores) / len(scores)

        final_patient_ids.append(pid)
        final_predicted_scores.append(average_score)
        final_true_labels.append(patient_labels[pid])
        final_splits.append(patient_splits[pid])  # 添加划分信息

    return final_patient_ids, final_predicted_scores, final_true_labels, final_splits

# 主程序
if __name__ == "__main__":
    data_dir = 'dataset'
    annotations = load_annotations(data_dir)

    # 使用split_dataset划分数据
    annotations = split_dataset(annotations)

    # 定义数据转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 创建数据集
    dataset = KneeInjuryDataset(annotations, data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载预训练模型
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=2)
    model.load_state_dict(torch.load('outputs/best_knee_injury_model_swin224.pth'))
    model.to(device)

    # 预测
    patient_ids, predicted_scores, true_labels, splits = predict_scores(model, dataloader, device)

    # 确保结果数据一致
    assert len(patient_ids) == len(predicted_scores) == len(true_labels) == len(splits), "数据长度不一致"

    # 将结果保存为CSV文件
    results = pd.DataFrame({
        'Patient_ID': patient_ids,
        'Predicted_Score': predicted_scores,
        'True_Label': true_labels,
        'Split': splits  # 添加划分信息列
    })
    results.to_csv('patient_scores.csv', index=False)
    print("预测得分已保存到patient_scores.csv")
