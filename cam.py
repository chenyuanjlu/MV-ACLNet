import os
import matplotlib.pyplot as plt
import timm
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd


# 加载注释文件
def load_annotations(data_dir):
    normal_csv_path = os.path.join(data_dir, "normal_annotations.csv")
    damaged_csv_path = os.path.join(data_dir, "damaged_annotations.csv")

    if not os.path.exists(normal_csv_path):
        raise FileNotFoundError(f"Normal annotations file not found at {normal_csv_path}")
    if not os.path.exists(damaged_csv_path):
        raise FileNotFoundError(f"Damaged annotations file not found at {damaged_csv_path}")

    normal_annotations = pd.read_csv(normal_csv_path)
    damaged_annotations = pd.read_csv(damaged_csv_path)

    normal_annotations["label"] = 0
    damaged_annotations["label"] = 1

    annotations = pd.concat([normal_annotations, damaged_annotations], ignore_index=True)
    return annotations


# 生成并保存热力图
def generate_and_save_cam(image_row, model, output_dir, device, transform):
    # 从注释中提取图像信息和边界框
    image_name = image_row["image_name"]
    label = image_row["label"]
    patient_id = image_row["patient_id"]
    bbox_x = image_row["bbox_x"]
    bbox_y = image_row["bbox_y"]
    bbox_width = image_row["bbox_width"]
    bbox_height = image_row["bbox_height"]

    # 图像路径和输出路径
    category = "normal" if label == 0 else "damaged"
    patient_dir = os.path.join(data_dir, category, patient_id)
    image_path = os.path.join(patient_dir, image_name)
    relative_path = os.path.relpath(image_path, start=data_dir)
    output_image_path = os.path.join(output_dir, relative_path)

    # 检查图像是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # 打开图像并裁剪
    image = Image.open(image_path).convert("RGB")
    cropped_image = image.crop((bbox_x, bbox_y, bbox_x + bbox_width, bbox_y + bbox_height))

    # 转换为张量
    img_tensor = transform(cropped_image).to(device)

    # CAM计算
    with torch.no_grad():
        feature_map = model.forward_features(img_tensor.unsqueeze(0))
    feature_map.requires_grad = True
    output = model.head(feature_map)
    target_class = 1  # 假设目标类是1
    c = output[0, target_class]
    c.backward()
    g = feature_map.grad
    a = g.mean(dim=[1, 2], keepdim=True)
    cam = (a * feature_map).sum(dim=-1)

    cam = torch.nn.functional.relu(cam)
    cam = torch.nn.functional.interpolate(cam.unsqueeze(0), (224, 224), mode="bilinear", align_corners=False).squeeze(0)
    cam_np = cam[0].detach().cpu().numpy()
    cam_np = (cam_np - np.min(cam_np)) / (np.max(cam_np) - np.min(cam_np))

    # 原始裁剪图像
    cropped_np = np.array(cropped_image.resize((224, 224))) / 255.0

    # 叠加热力图
    plt.imshow(cropped_np)
    plt.imshow(cam_np, cmap="jet", alpha=0.5)
    plt.axis("off")

    # 保存热力图
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    plt.savefig(output_image_path)
    plt.close()


# 遍历数据集结构，生成并保存热力图
def process_dataset(annotations, model, output_dir, device, transform):
    for _, row in annotations.iterrows():
        try:
            generate_and_save_cam(row, model, output_dir, device, transform)
        except Exception as e:
            print(f"Error processing {row['image_name']}: {e}")


# 主程序
if __name__ == "__main__":
    # 数据路径
    data_dir = "train_dataset"
    output_dir = "gradcam_outputs"

    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 加载注释文件和模型
    annotations = load_annotations(data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=2)
    model.load_state_dict(torch.load("outputs/best_knee_injury_model_swin224.pth", map_location=device))
    model.to(device)
    model.eval()

    # 图像变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 处理整个数据集
    process_dataset(annotations, model, output_dir, device, transform)
