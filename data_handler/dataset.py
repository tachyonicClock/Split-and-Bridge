import os
from torchvision import datasets, transforms
import torch
import numpy as np

DATASET_DIR = os.environ['DATASETS']

# To incdude a new Dataset, inherit from Dataset and add all the Dataset specific parameters here.
# Goal : Remove any data specific parameters from the rest of the code

def make_dataset(file_path, classes, test_img_per_class = 50):
    data = datasets.ImageFolder(file_path)
    loader = data.loader

    img_cnt = np.zeros(20000)
    img_mask = np.zeros(20000)
    target_map = np.zeros(20000)

    for i in range(len(data.imgs)):
        path, target = data.imgs[i]
        img_cnt[target] += 1

    idx = classes
    img_cnt_args = np.flip(np.argsort(img_cnt), axis=0)[:idx]
    img_mask[img_cnt_args] = 1
    target_map[img_cnt_args] = np.arange(idx)

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    img_cnt = np.zeros(20000)
    for i in range(len(data.imgs)):
        path, target = data.imgs[i]
        if img_mask[target] == 0:
            continue
        if img_cnt[target] < test_img_per_class:
            test_data.append(path)
            test_labels.append(int(target_map[target]))
        else:
            train_data.append(path)
            train_labels.append(int(target_map[target]))

        img_cnt[target] += 1


    train_data = np.stack(train_data, axis=0)
    test_data = np.stack(test_data, axis=0)
    
    return train_data, train_labels, test_data, test_labels, loader

class Dataset():
    '''
    Base class to reprenent a Dataset
    '''

    def __init__(self, classes, name):
        self.classes = classes
        self.name = name
        self.train_data = None
        self.test_data = None


class CIFAR100(Dataset):
    def __init__(self):
        super().__init__(100, "CIFAR100")

        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])

        train_dataset = datasets.CIFAR100(DATASET_DIR, train=True, transform=self.train_transform, download=True)
        self.train_data = train_dataset.data
        self.train_labels = np.array(train_dataset.targets)
        test_dataset = datasets.CIFAR100(DATASET_DIR, train=False, transform=self.test_transform, download=True)
        self.test_data = test_dataset.data
        self.test_labels = np.array(test_dataset.targets)


class CIFAR10(Dataset):
    def __init__(self):
        super().__init__(10, "CIFAR10")

        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])

        train_dataset = datasets.CIFAR10(DATASET_DIR, train=True, transform=self.train_transform, download=True)
        self.train_data = train_dataset.data
        self.train_labels = np.array(train_dataset.targets)
        test_dataset = datasets.CIFAR10(DATASET_DIR, train=False, transform=self.test_transform, download=True)
        self.test_data = test_dataset.data
        self.test_labels = np.array(test_dataset.targets)

class FMNIST(Dataset):
    def __init__(self):
        super().__init__(10, "FMNIST")

        self.train_transform = transforms.Compose([
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda x: x.float()),
        ])

        self.test_transform = transforms.Compose([
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
            transforms.Resize(32),
            transforms.Lambda(lambda x: x.float()),
        ])

        train_dataset = datasets.FashionMNIST(DATASET_DIR, train=True, transform=self.train_transform, download=True)
        self.train_data = train_dataset.data
        self.train_labels = np.array(train_dataset.targets)
        test_dataset = datasets.FashionMNIST(DATASET_DIR, train=False, transform=self.test_transform, download=True)
        self.test_data = test_dataset.data
        self.test_labels = np.array(test_dataset.targets)
        self.loader = lambda x : x

class TinyImagenet(Dataset):
    def __init__(self):
        super().__init__(200, "TinyImagenet")

        train_dir = "C:/Users/admin/Desktop/train"
        val_dir = "C:/Users/admin/Desktop/val"

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        train_data = datasets.ImageFolder(train_dir, transform=self.train_transform)
        test_data = datasets.ImageFolder(val_dir, transform=self.test_transform)
        self.loader = train_data.loader

        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []

        for i in range(len(train_data.imgs)):
            path, target = train_data.imgs[i]
            self.train_data.append(path)
            self.train_labels.append(target)

        for i in range(len(test_data.imgs)):
            path, target = test_data.imgs[i]
            self.test_data.append(path)
            self.test_labels.append(target)

        self.train_data = np.stack(self.train_data, axis=0)
        self.test_data = np.stack(self.test_data, axis=0)


class CORe50_NC(Dataset):
    """Class incremental CORe50 dataset"""
    def __init__(self, mini=False):
        super().__init__(50, "CORe50_NC")
        from avalanche.benchmarks.datasets import CORe50Dataset


        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        train_dataset = CORe50Dataset(DATASET_DIR, train=True, transform=self.train_transform, download=True, mini=mini)
        # train_dataset.
        self.train_data = train_dataset
        self.train_labels = np.array(train_dataset.targets)
        test_dataset = CORe50Dataset(DATASET_DIR, train=False, transform=self.test_transform, download=True, mini=mini)
        self.test_data = test_dataset
        self.test_labels = np.array(test_dataset.targets)
        if mini:
            self.loader = lambda path: train_dataset.loader(str(f"{DATASET_DIR}/core50_32x32/{path}"))
        else:
            self.loader = lambda path: train_dataset.loader(str(f"{DATASET_DIR}/core50_128x128/{path}"))

        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []

        for i in range(len(train_dataset)):
            path = train_dataset.paths[i]
            target = train_dataset.targets[i]
            self.train_data.append(path)
            self.train_labels.append(target)

        for i in range(len(test_dataset)):
            path = test_dataset.paths[i]
            target = test_dataset.targets[i]
            self.test_data.append(path)
            self.test_labels.append(target)

        self.train_data = np.stack(self.train_data, axis=0)
        self.test_data = np.stack(self.test_data, axis=0)
