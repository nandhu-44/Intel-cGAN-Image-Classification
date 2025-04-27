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

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_probs = torch.nn.functional.log_softmax(input, dim=-1)
        n_classes = input.size(-1)
        true_dist = torch.zeros_like(log_probs).fill_(self.smoothing / (n_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))

def train_model(resume_from=None, num_epochs=20):
    cudnn.benchmark = True
    checkpoint_dir = 'checkpoints_vit'
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = IntelDataset('dataset', split='train', transform=train_transform)
    val_dataset = IntelDataset('dataset', split='test', transform=val_transform)

    batch_size = 8
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

    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads.head = nn.Linear(model.heads.head.in_features, 6)
    nn.init.xavier_uniform_(model.heads.head.weight)
    nn.init.zeros_(model.heads.head.bias)
    model = model.to(device)

    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * steps_per_epoch)

    start_epoch = 0
    best_accuracy = 0.0
    best_val_loss = float('inf')
    patience = 5
    epochs_no_improve = 0

    if resume_from and os.path.exists(resume_from):
        print(f"Loading checkpoint from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint.get('best_accuracy', 0.0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
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
            'best_accuracy': best_accuracy,
            'best_val_loss': best_val_loss
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
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {accuracy:.2f}%, Validation Loss: {avg_val_loss:.3f}")

        checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'accuracy': accuracy,
            'best_accuracy': best_accuracy,
            'best_val_loss': best_val_loss
        }, checkpoint_path)
        if avg_val_loss < best_val_loss:
            best_accuracy = accuracy
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'accuracy': accuracy,
                'best_accuracy': best_accuracy,
                'best_val_loss': best_val_loss
            }, best_model_path)
            print(f"New best model saved! Accuracy: {accuracy:.2f}%, Val Loss: {avg_val_loss:.3f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training completed. Best accuracy: {best_accuracy:.2f}%, Best val loss: {best_val_loss:.3f}")

if __name__ == '__main__':
    num_epochs = int(input("Enter the number of epochs: "))
    resume_from = input("Enter the path to resume from (or leave blank to start fresh): ")
    train_model(num_epochs=num_epochs, resume_from=resume_from if resume_from else None)