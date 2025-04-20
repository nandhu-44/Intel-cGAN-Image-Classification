import torch
import torch.nn as nn
import torchvision.transforms as transforms

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Encoder
        self.enc1 = nn.Conv2d(4, 64, 4, stride=2, padding=1)  # 3 channels + 1 condition
        self.enc2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.enc3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        # Decoder
        self.dec1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dec3 = nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x, condition):
        cond = condition.view(-1, 1, 1, 1).expand(-1, 1, x.size(2), x.size(3))
        x = torch.cat([x, cond], dim=1)
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(e1))
        e3 = self.relu(self.enc3(e2))
        d1 = self.relu(self.dec1(e3))
        d2 = self.relu(self.dec2(d1))
        out = self.tanh(self.dec3(d2))
        return out

def get_transforms():
    """Return transforms for image preprocessing."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])