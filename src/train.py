from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from utils import set_seed


@dataclass
class Config:
    data_dir: str = 'data/split'
    model_name: str = 'resnet18'
    num_classes: int = 3
    batch_size: int = 16
    lr: float = 5e-3
    epochs: int = 10
    freeze_layers: int = 5  # Количество слоев для заморозки
    seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_data_loaders(config):
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(f'{config.data_dir}/train', transform=train_transforms)
    val_dataset = datasets.ImageFolder(f'{config.data_dir}/val', transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(f'{config.data_dir}/test', transform=val_test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def train_model(config):
    set_seed(config.seed)
    model = timm.create_model(config.model_name, pretrained=True, num_classes=config.num_classes)

    # Заморозка слоев
    for param in list(model.parameters())[:config.freeze_layers]:
        param.requires_grad = False

    model = model.to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)

    # Обучение 
    for epoch in range(config.epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(config.device), labels.to(config.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{config.epochs}, Loss: {loss.item()}')

    
    torch.save(model.state_dict(), f'models/{config.model_name}.pth')


if __name__ == '__main__':
    config = Config()
    train_loader, val_loader, test_loader = get_data_loaders(config)
    train_model(config)