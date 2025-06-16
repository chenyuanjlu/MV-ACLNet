import torch
import timm
from torch import nn, optim
from data_preprocessing import load_annotations, split_dataset
from dataset import get_dataloaders
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import numpy as np
from torch.optim import lr_scheduler


def smooth_curve(values, weight=0.8):
    """
    使用指数加权平均法平滑曲线
    :param values: 原始值列表
    :param weight: 平滑权重，值越大平滑程度越高
    :return: 平滑后的值列表
    """
    smoothed_values = [values[0]]  # 保留第一个值不变
    for value in values[1:]:
        smoothed_value = smoothed_values[-1] * weight + (1 - weight) * value
        smoothed_values.append(smoothed_value)
    return smoothed_values


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, output_dir='outputs'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 设定学习率调度器：每 10 轮降低一次学习率
    scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

    best_val_accuracy = 0.0  # 用于追踪验证集上的最佳准确率
    best_labels = []  # 存储效果最好的模型对应的真实标签
    best_probs = []  # 存储效果最好的模型对应的预测概率
    best_threshold = 0.5  # 默认使用 0.5 作为阈值

    # 初始化准确率和损失的初始值
    val_accuracies = []  # 验证集初始准确率
    train_accuracies = []  # 训练集初始准确率
    train_losses = []  # 训练集初始损失
    val_losses = []  # 验证集初始损失

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # 在训练集上添加进度条
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Training]", leave=True)

        for images, labels, _ in train_loader_tqdm:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 计算训练准确率
            _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

            # 更新 tqdm 进度条
            train_loader_tqdm.set_postfix(loss=running_loss / len(train_loader), accuracy=correct_train / total_train)

        epoch_loss = running_loss / len(train_loader)
        train_loss = epoch_loss  # 当前 epoch 的训练损失
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # 验证模型
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_labels = []
        all_probs = []

        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Validation]", leave=True)

        with torch.no_grad():
            for images, labels, _ in val_loader_tqdm:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # 获取每个类别的概率
                probs = torch.softmax(outputs, dim=1)

                # 获取模型预测的类别
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

                _, predicted_val = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted_val == labels).sum().item()

                val_loader_tqdm.set_postfix(loss=val_loss / len(val_loader), accuracy=correct_val / total_val)

        val_loss /= len(val_loader)
        val_accuracy = correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(
            f'Epoch {epoch + 1}/{num_epochs}, '
            f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
            f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        # 如果当前验证准确率更高，保存当前模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_labels = all_labels
            best_probs = all_probs
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_knee_injury_model_swin_both.pth'))
            print(f'Saved best model with accuracy: {best_val_accuracy:.4f}')

        # 每轮结束后更新学习率
        scheduler.step()

    # 计算约登指数和最佳阈值
    fpr, tpr, thresholds = roc_curve(best_labels, best_probs)
    youden_index = tpr - fpr
    best_threshold_idx = np.argmax(youden_index)
    best_threshold = thresholds[best_threshold_idx]

    print(f"Best Threshold (Youden's J): {best_threshold:.4f}")

    # 使用最佳阈值重新计算性能指标
    final_predictions = [1 if prob >= best_threshold else 0 for prob in best_probs]
    accuracy = accuracy_score(best_labels, final_predictions)
    precision = precision_score(best_labels, final_predictions)
    recall = recall_score(best_labels, final_predictions)
    f1 = f1_score(best_labels, final_predictions)

    fpr, tpr, _ = roc_curve(best_labels, best_probs)
    roc_auc = auc(fpr, tpr)

    precision_curve, recall_curve, _ = precision_recall_curve(best_labels, best_probs)
    pr_auc = auc(recall_curve, precision_curve)

    # 绘制 ROC 曲线
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.pdf'))
    plt.close()

    # 绘制 PR 曲线
    plt.figure(figsize=(6, 5))
    plt.plot(recall_curve, precision_curve, color='b', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pr_curve.pdf'))
    plt.close()

    # 绘制训练和验证的准确率曲线（平滑处理）
    plt.figure(figsize=(6, 5))
    smoothed_train_accuracies = smooth_curve(train_accuracies)  # 使用完整的 train_accuracies
    smoothed_val_accuracies = smooth_curve(val_accuracies)  # 使用完整的 val_accuracies
    plt.plot(range(1, num_epochs + 1), smoothed_train_accuracies, color='darkorange', label='Train Accuracy (Smoothed)')
    plt.plot(range(1, num_epochs + 1), smoothed_val_accuracies, color='blue', label='Val Accuracy (Smoothed)')
    plt.title('Smoothed Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_curve_smoothed.pdf'))
    plt.close()

    # 绘制训练和验证的损失曲线（平滑处理）
    plt.figure(figsize=(6, 5))
    smoothed_train_losses = smooth_curve(train_losses)  # 使用完整的 train_losses
    smoothed_val_losses = smooth_curve(val_losses)  # 使用完整的 val_losses
    plt.plot(range(1, num_epochs + 1), smoothed_train_losses, color='darkorange', label='Train Loss (Smoothed)')
    plt.plot(range(1, num_epochs + 1), smoothed_val_losses, color='blue', label='Val Loss (Smoothed)')
    plt.title('Smoothed Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curve_smoothed.pdf'))
    plt.close()

    # 保存性能指标到文本文件
    metrics_file = os.path.join(output_dir, 'metrics_both.txt')
    with open(metrics_file, 'w') as f:
        f.write(f'Accuracy: {accuracy:.4f}\n')
        f.write(f'Precision: {precision:.4f}\n')
        f.write(f'Recall: {recall:.4f}\n')
        f.write(f'F1 Score: {f1:.4f}\n')
        f.write(f'ROC AUC: {roc_auc:.4f}\n')
        f.write(f'PR AUC: {pr_auc:.4f}\n')
        f.write(f'Best Threshold: {best_threshold:.4f}\n')

    print(f'Model evaluation metrics saved to {metrics_file}')


if __name__ == "__main__":
    data_dir = 'train_dataset'
    annotations = load_annotations(data_dir)
    train_annotations, val_annotations = split_dataset(annotations)
    train_loader, val_loader = get_dataloaders(train_annotations, val_annotations, data_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50, output_dir='outputs')