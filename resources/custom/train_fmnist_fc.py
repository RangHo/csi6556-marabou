import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU name: {torch.cuda.get_device_name(0)}")


class FCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 10),
        )

    def forward(self, x):
        return self.net(x)  # softmax is cringe lol


def train(epochs, tfm):
    train_ds = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=tfm
    )
    train_loader = DataLoader(
        train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True
    )

    model = FCNet().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(
            f"epoch {epoch + 1}/{epochs}: avg_loss={total_loss/len(train_loader):.4f}"
        )

    return model


def evaluate(model, tfm):
    test_ds = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=tfm
    )
    test_loader = DataLoader(
        test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True
    )

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    print(f"test_acc={correct/total:.3f}")


def export(model, path):
    model.cpu().eval()
    dummy = torch.zeros(1, 784)  # batch=1, flat input
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13,
        dynamo=False,
        do_constant_folding=True,
    )

    size_kib = os.path.getsize(path) / 1024
    print(f"wrote {path} ({size_kib:.1f} KiB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--export-path", type=str, default="fmnist_784x16x16x10.onnx")
    args = parser.parse_args()

    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.view(-1)),
        ]
    )

    model = train(args.epochs, tfm)
    evaluate(model, tfm)
    export(model, args.export_path)
