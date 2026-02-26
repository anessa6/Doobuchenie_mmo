from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms

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

train_dataset = ImageFolder('data/split/train', transform=train_transforms)
val_dataset = ImageFolder('data/split/val', transform=val_test_transforms)
test_dataset = ImageFolder('data/split/test', transform=val_test_transforms)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
print(len(train_dataset), len(val_dataset), len(test_dataset))