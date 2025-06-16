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
from scipy.stats import norm
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.model_selection import train_test_split

# 加载注释数据
def load_annotations(data_dir):
    normal_csv_path = os.path.join(data_dir, 'normal_annotations.csv')
    damaged_csv_path = os.path.join(data_dir, 'damaged_annotations.csv')

    if not os.path.exists(normal_csv_path):
        raise FileNotFoundError(f"Normal annotations file not found at {normal_csv_path}")
    if not os.path.exists(damaged_csv_path):
        raise FileNotFoundError(f"Damaged annotations file not found at {damaged_csv_path}")

    normal_annotations = pd.read_csv(normal_csv_path)
    damaged_annotations = pd.read_csv(damaged_csv_path)

    normal_annotations['label'] = 0
    damaged_annotations['label'] = 1

    annotations = pd.concat([normal_annotations, damaged_annotations], ignore_index=True)
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
        patient_id = row['patient_id']

        category = 'normal' if label == 0 else 'damaged'
        patient_dir = os.path.join(self.data_dir, category, patient_id)
        img_path = os.path.join(patient_dir, image_name)

        if not os.path.exists(img_path):
            subfolders = [f.path for f in os.scandir(patient_dir) if f.is_dir()]

            img_path = None
            for subfolder in subfolders:
                possible_img_path = os.path.join(subfolder, image_name)
                if os.path.exists(possible_img_path):
                    img_path = possible_img_path
                    break

            if img_path is None:
                raise FileNotFoundError(f"Image {image_name} not found for patient {patient_id}")

        image = Image.open(img_path).convert('RGB')
        bbox_x, bbox_y, bbox_width, bbox_height = row['bbox_x'], row['bbox_y'], row['bbox_width'], row['bbox_height']
        cropped_image = image.crop((bbox_x, bbox_y, bbox_x + bbox_width, bbox_y + bbox_height))

        if self.transform:
            cropped_image = self.transform(cropped_image)

        return cropped_image, label, patient_id

# 设置全局随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 以 patient_id 为单位进行分层抽样划分数据集
def stratified_split_by_patient(annotations, test_size=0.1, random_state=42, output_dir='split_results'):
    """
    以 patient_id 为单位进行分层抽样划分数据集，并保存划分结果。
    :param annotations: 包含所有样本的 DataFrame
    :param test_size: 验证集比例
    :param random_state: 随机种子
    :param output_dir: 划分结果保存目录
    :return: 划分后的数据集（包含 'subset' 列，标记为 'train' 或 'val'）
    """
    # 检查是否已经存在划分结果
    train_path = os.path.join(output_dir, 'train_annotations_lateral.csv')
    val_path = os.path.join(output_dir, 'val_annotations_lateral.csv')

    if os.path.exists(train_path) and os.path.exists(val_path):
        print("加载已保存的划分结果...")
        train_annotations = pd.read_csv(train_path)
        val_annotations = pd.read_csv(val_path)
        return pd.concat([train_annotations, val_annotations], ignore_index=True)

    # 按 patient_id 分组，并提取每个病人的标签（取第一个样本的标签）
    patient_groups = annotations.groupby('patient_id')
    patient_ids = list(patient_groups.groups.keys())  # 所有病人的 ID
    patient_labels = [group['label'].iloc[0] for _, group in patient_groups]  # 每个病人的标签

    # 使用 train_test_split 对 patient_id 进行分层抽样
    train_patient_ids, val_patient_ids, _, _ = train_test_split(
        patient_ids, patient_labels, test_size=test_size, stratify=patient_labels, random_state=random_state
    )

    # 根据 patient_id 划分数据集
    train_annotations = annotations[annotations['patient_id'].isin(train_patient_ids)]
    val_annotations = annotations[annotations['patient_id'].isin(val_patient_ids)]

    # 添加 'subset' 列，标记为 'train' 或 'val'
    train_annotations['subset'] = 'train'
    val_annotations['subset'] = 'val'

    # 保存划分结果
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    train_annotations.to_csv(train_path, index=False)
    val_annotations.to_csv(val_path, index=False)
    print(f"划分结果已保存到 {output_dir}")

    # 合并训练集和验证集
    return pd.concat([train_annotations, val_annotations], ignore_index=True)

# 预测得分函数
def predict_scores_with_ci(model, dataloader, device):
    model.eval()
    patient_scores = defaultdict(list)
    patient_labels = {}
    all_true_labels = []
    all_pred_probs = []

    with torch.no_grad():
        for images, labels, patient_ids in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            for pid, prob, label in zip(patient_ids, probs.cpu().numpy(), labels.cpu().numpy()):
                pid_str = str(pid)
                patient_scores[pid_str].append(prob)
                patient_labels[pid_str] = label

            all_true_labels.extend(labels.cpu().numpy())
            all_pred_probs.extend(probs.cpu().numpy())

    final_patient_ids = []
    final_predicted_scores = []
    final_true_labels = []
    final_confidence_intervals = []

    for pid in patient_scores:
        scores = patient_scores[pid]
        average_score = np.mean(scores)
        std_error = np.std(scores) / np.sqrt(len(scores))
        ci_margin = std_error * norm.ppf(0.975)  # 95%置信区间

        final_patient_ids.append(pid)
        final_predicted_scores.append(average_score)
        final_true_labels.append(patient_labels[pid])
        final_confidence_intervals.append((average_score - ci_margin, average_score + ci_margin))

    return final_patient_ids, final_predicted_scores, final_true_labels, final_confidence_intervals, all_true_labels, all_pred_probs

# 绘制ROC曲线和PR曲线并保存为PDF
def plot_roc_pr_curve(true_labels, pred_probs, output_pdf_path='roc_pr_curve.pdf'):
    # ROC曲线
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    roc_auc = auc(fpr, tpr)

    # PR曲线
    precision, recall, _ = precision_recall_curve(true_labels, pred_probs)
    pr_auc = average_precision_score(true_labels, pred_probs)

    plt.figure(figsize=(12, 5))

    # ROC曲线
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')

    # PR曲线
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='b', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')

    plt.tight_layout()

    # 保存为PDF
    plt.savefig(output_pdf_path, format='pdf')
    plt.close()

# 计算性能指标及其95%置信区间
def calculate_metrics_with_ci(true_labels, pred_labels, pred_probs, n_bootstrap=1000, output_file='performance_metrics_with_ci.txt'):
    """
    计算模型的性能指标及其95%置信区间，并保存到文件。
    :param true_labels: 真实标签
    :param pred_labels: 预测标签
    :param pred_probs: 预测概率
    :param n_bootstrap: bootstrap 采样次数
    :param output_file: 保存结果的文件名
    """
    # 初始化存储指标的列表
    sensitivity_list = []
    specificity_list = []
    accuracy_list = []
    precision_list = []
    f1_list = []
    auc_list = []

    # 使用 bootstrap 方法计算置信区间
    for _ in range(n_bootstrap):
        # 重采样
        indices = resample(range(len(true_labels)), replace=True)
        resampled_true_labels = np.array(true_labels)[indices]
        resampled_pred_labels = np.array(pred_labels)[indices]
        resampled_pred_probs = np.array(pred_probs)[indices]

        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(resampled_true_labels, resampled_pred_labels).ravel()

        # 计算性能指标
        sensitivity = recall_score(resampled_true_labels, resampled_pred_labels)  # 灵敏度/召回率
        specificity = tn / (tn + fp)  # 特异性
        accuracy = accuracy_score(resampled_true_labels, resampled_pred_labels)  # 准确率
        precision = precision_score(resampled_true_labels, resampled_pred_labels)  # 精确率
        f1 = f1_score(resampled_true_labels, resampled_pred_labels)  # F1分数
        auc = roc_auc_score(resampled_true_labels, resampled_pred_probs)  # AUC

        # 存储指标
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        f1_list.append(f1)
        auc_list.append(auc)

    # 计算每个指标的均值和95%置信区间
    def calculate_mean_and_ci(values):
        mean = np.mean(values)
        ci_lower = np.percentile(values, 2.5)
        ci_upper = np.percentile(values, 97.5)
        return mean, ci_lower, ci_upper

    sensitivity_mean, sensitivity_ci_lower, sensitivity_ci_upper = calculate_mean_and_ci(sensitivity_list)
    specificity_mean, specificity_ci_lower, specificity_ci_upper = calculate_mean_and_ci(specificity_list)
    accuracy_mean, accuracy_ci_lower, accuracy_ci_upper = calculate_mean_and_ci(accuracy_list)
    precision_mean, precision_ci_lower, precision_ci_upper = calculate_mean_and_ci(precision_list)
    f1_mean, f1_ci_lower, f1_ci_upper = calculate_mean_and_ci(f1_list)
    auc_mean, auc_ci_lower, auc_ci_upper = calculate_mean_and_ci(auc_list)

    # 将结果保存到文件
    with open(output_file, 'w') as f:
        f.write(f"Sensitivity/Recall: {sensitivity_mean:.4f} (95% CI: {sensitivity_ci_lower:.4f} - {sensitivity_ci_upper:.4f})\n")
        f.write(f"Specificity: {specificity_mean:.4f} (95% CI: {specificity_ci_lower:.4f} - {specificity_ci_upper:.4f})\n")
        f.write(f"Accuracy: {accuracy_mean:.4f} (95% CI: {accuracy_ci_lower:.4f} - {accuracy_ci_upper:.4f})\n")
        f.write(f"Precision: {precision_mean:.4f} (95% CI: {precision_ci_lower:.4f} - {precision_ci_upper:.4f})\n")
        f.write(f"F1 Score: {f1_mean:.4f} (95% CI: {f1_ci_lower:.4f} - {f1_ci_upper:.4f})\n")
        f.write(f"AUC: {auc_mean:.4f} (95% CI: {auc_ci_lower:.4f} - {auc_ci_upper:.4f})\n")

    print(f"性能指标及其95%置信区间已保存到 {output_file}")

# 主程序
if __name__ == "__main__":
    # 设置全局随机种子
    set_seed(42)

    data_dir = 'train_dataset'
    annotations = load_annotations(data_dir)

    # 以 patient_id 为单位进行分层抽样划分数据集
    annotations = stratified_split_by_patient(annotations, test_size=0.1, random_state=42)

    # 创建数据转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 数据加载
    dataset = KneeInjuryDataset(annotations, data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=2)
    # 加载模型权重
    model.load_state_dict(torch.load('outputs/best_knee_injury_model_swin_both.pth', map_location=device))
    model.to(device)

    patient_ids, predicted_scores, true_labels, confidence_intervals, all_true_labels, all_pred_probs = predict_scores_with_ci(model, dataloader, device)

    # 确保预测数据和病人数据长度一致
    assert len(patient_ids) == len(predicted_scores) == len(true_labels) == len(confidence_intervals), "数据长度不一致"

    # 保存结果到CSV文件
    results = pd.DataFrame({
        'Patient_ID': patient_ids,
        'Predicted_Score': predicted_scores,
        'True_Label': true_labels,
        'Predicted_Label': [1 if score >= 0.5 else 0 for score in predicted_scores],
        'CI_Lower': [ci[0] for ci in confidence_intervals],
        'CI_Upper': [ci[1] for ci in confidence_intervals],
        'Subset': [annotations[annotations['patient_id'] == pid]['subset'].iloc[0] for pid in patient_ids]
    })
    results.to_csv('patient_scores.csv', index=False)
    print("预测得分和置信区间已保存到 patient_scores_lateral.csv")

    # 绘制ROC曲线和PR曲线并保存为PDF
    plot_roc_pr_curve(all_true_labels, all_pred_probs, output_pdf_path='roc_pr_curve.pdf')
    print("ROC曲线和PR曲线已保存为 'roc_pr_curve.pdf'")

    # 计算并保存性能指标及其95%置信区间
    pred_labels = [1 if score >= 0.5 else 0 for score in all_pred_probs]
    calculate_metrics_with_ci(all_true_labels, pred_labels, all_pred_probs, output_file='performance_metrics_with_ci.txt')