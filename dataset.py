import os
import random
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image

def assign_labels(root_dir):
    """
    폴더명 -> 라벨 인덱스 맵핑
    """
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    subdirs.sort()
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(subdirs)}
    return class_to_idx

def split_dataset(root_dir, train_ratio=0.8):
    class_to_idx = assign_labels(root_dir)

    train_data = []
    test_data = []

    for cls_name, cls_idx in class_to_idx.items():
        subdir_path = os.path.join(root_dir, cls_name)
        if os.path.isdir(subdir_path):
            images = [os.path.join(subdir_path, file) for file in os.listdir(subdir_path)
                      if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
            random.shuffle(images)
            split_idx = int(len(images) * train_ratio)

            train_data.extend([(img_path, cls_idx) for img_path in images[:split_idx]])
            test_data.extend([(img_path, cls_idx) for img_path in images[split_idx:]])
    return train_data, test_data

class CustomDataset(datasets.VisionDataset):
    def __init__(self, data_list, transform=None):
        super(CustomDataset, self).__init__(root=None, transform=transform)
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

def get_dataloader(root_dir, batch_size, image_size, train_ratio=0.8):
    train_data, test_data = split_dataset(root_dir, train_ratio)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    train_dataset = CustomDataset(train_data, transform=transform)
    test_dataset = CustomDataset(test_data, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_loader, test_loader
