import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms
from torchvision.transforms import v2


class SignDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        # v2.Compose([
        #     v2.RandomResizedCrop(size=(28, 28), antialias=True),
        #     v2.RandomHorizontalFlip(p=0.5)
        # ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data.iloc[idx, 1:].to_numpy().reshape(1, 28, 28)
        label = self.data.iloc[idx]['label']

        image = torch.tensor(image).float() / 255
        label = torch.tensor(label)

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label


def get_sign_dataloader(
        csv_path_train, csv_path_val, batch_size, shuffle=True, num_workers=1,
    ):

    transform = transforms.Compose([
        transforms.RandomRotation(10, fill=0),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5)
    ])
        
    train_dataset = SignDataset(csv_file=csv_path_train, transform=transform)
    val_dataset = SignDataset(csv_file=csv_path_val)

    loader_args = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers
    }
    return DataLoader(train_dataset, **loader_args), DataLoader(val_dataset, **loader_args)


def get_sign_test_dataloader(
        csv_path_test, batch_size, num_workers=1,
    ):
    test_dataset = SignDataset(csv_file=csv_path_test)

    loader_args = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': num_workers
    }
    return DataLoader(test_dataset, **loader_args)
