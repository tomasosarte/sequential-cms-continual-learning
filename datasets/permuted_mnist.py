import torch
from torchvision import datasets, transforms

class PermutedMNIST:
    def __init__(self, perm, root="./data"):
        self.perm = perm
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)[perm])
        ])
        self.root = root

    def train_loader(self, batch_size=128):
        return torch.utils.data.DataLoader(
            datasets.MNIST(self.root, train=True, download=True, transform=self.transform),
            batch_size=batch_size, shuffle=True
        )

    def test_loader(self, batch_size=1000):
        return torch.utils.data.DataLoader(
            datasets.MNIST(self.root, train=False, download=True, transform=self.transform),
            batch_size=batch_size, shuffle=False
        )