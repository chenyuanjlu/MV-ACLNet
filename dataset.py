import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class KneeInjuryDataset(Dataset):
    def __init__(self, annotations, data_dir, transform=None):
        self.annotations = annotations  # 注释文件 (DataFrame)
        self.data_dir = data_dir  # 数据目录
        self.transform = transform  # 图像变换

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        image_name = row['image_name']
        label = row['label']
        patient_id = row['patient_id']  # 从CSV文件中获取病人ID

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

        return cropped_image, label, patient_id

def get_dataloaders(train_annotations, val_annotations, data_dir, batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 可以根据需要调整大小
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomGrayscale(p=0.1),
        transforms.ColorJitter(saturation=0.2),
        transforms.ColorJitter(hue=0.1),
        transforms.ToTensor(),
    ])

    # 定义训练和验证集
    train_dataset = KneeInjuryDataset(train_annotations, data_dir, transform=transform)
    val_dataset = KneeInjuryDataset(val_annotations, data_dir, transform=transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
