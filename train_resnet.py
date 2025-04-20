import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from intel_dataset import IntelDataset
from tqdm import tqdm
import os
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_model(resume_from=None, num_epochs=15):
    cudnn.benchmark = True
    checkpoint_dir = 'checkpoints_resnet'
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])    
    train_dataset = IntelDataset('dataset', split='train', transform=train_transform)
    val_dataset = IntelDataset('dataset', split='test', transform=val_transform)

    batch_size = 16
    num_workers = min(4, os.cpu_count())
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2,
        persistent_workers=True
    )

    steps_per_epoch = len(train_loader)
    print(f"Steps per epoch: {steps_per_epoch} (Dataset size: {len(train_dataset)}, Batch size: {batch_size})")

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, 6)
    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.zeros_(model.fc.bias)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * steps_per_epoch)

    start_epoch = 0
    best_accuracy = 0.0
    if resume_from and os.path.exists(resume_from):
        print(f"Loading checkpoint from {resume_from}")
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint.get('best_accuracy', 0.0)
        print(f"Resuming from epoch {start_epoch}, best_accuracy: {best_accuracy:.2f}%")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for i, (images, labels) in enumerate(pbar):
            if images is None or labels is None:
                continue
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if torch.isnan(loss):
                print(f"NaN loss detected at step {i}, skipping update")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            if i % 50 == 49:
                pbar.set_postfix({'loss': running_loss/50, 'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'})
                running_loss = 0.0

        checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}_train.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'accuracy': 0.0,
            'best_accuracy': best_accuracy
        }, checkpoint_path)
        print(f"Training checkpoint saved: {checkpoint_path}")

        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc='Validation')
            for images, labels in val_pbar:
                if images is None or labels is None:
                    continue
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_pbar.set_postfix({'val_loss': val_loss/(val_pbar.n+1), 'accuracy': f'{100*correct/total:.2f}%'})

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {accuracy:.2f}%")

        checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'accuracy': accuracy,
            'best_accuracy': best_accuracy
        }, checkpoint_path)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'accuracy': accuracy,
                'best_accuracy': best_accuracy
            }, best_model_path)
            print(f"New best model saved! Accuracy: {accuracy:.2f}%")

    print(f"Training completed. Best accuracy: {best_accuracy:.2f}%")

if __name__ == '__main__':
    num_epochs = int(input("Enter the number of epochs: "))
    resume_from = input("Enter the path to resume from (or leave blank to start fresh): ")
    train_model(num_epochs=num_epochs, resume_from=resume_from if resume_from else None)