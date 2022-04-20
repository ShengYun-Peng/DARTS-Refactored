from typing import Tuple
import torch
import os
import argparse
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

class Cutout(object):
    """
    randomly zero out a blob in an image
    """
    def __init__(self, length: int):
        self.length = length

    def __call__(self, img: torch.tensor):
        _, H, W = img.shape
        mask = np.ones((H, W), np.float32)
        y = np.random.randint(H)
        x = np.random.randint(W)

        y1 = np.clip(y - self.length // 2, 0, H)
        y2 = np.clip(y + self.length // 2, 0, H)
        x1 = np.clip(x - self.length // 2, 0, W)
        x2 = np.clip(x + self.length // 2, 0, W)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img

def transform_cifar10(args: argparse.Namespace) -> Tuple[transforms.Compose, transforms.Compose]:
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

    return train_transform, valid_transform

def prepare_cifar10(args: argparse.Namespace) -> DataLoader:
    assert os.path.exists(args.data)

    _, test_transform = transform_cifar10(args)
    dataset = CIFAR10(root=args.data, train=False, download=True, transform=test_transform)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8)

    return dataloader
