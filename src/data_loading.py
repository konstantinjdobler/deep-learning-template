import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms, datasets
from torch.utils.data import ConcatDataset


class BasicDataModule(pl.LightningDataModule):
    def __init__(self, data_dirs: str, batch_size: int, workers: int, image_resizing: int, fast_debug: bool = False):
        super().__init__()
        self.data_dirs = data_dirs
        self.batch_size = batch_size
        self.workers = workers
        self.image_size = image_resizing
        self.fast_debug = fast_debug

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize(self.image_size),
            transforms.RandomCrop(self.image_size),
            transforms.RandomHorizontalFlip()
        ])
        self.dims = (3, self.image_size, self.image_size)

    def setup(self, stage):
        complete_data = ConcatDataset([datasets.ImageFolder(
            data_dir, transform=self.transform) for data_dir in self.data_dirs])
        total_len = len(complete_data)
        train_len, val_len = int(0.8*total_len), int(0.1*total_len)
        test_len = total_len - (train_len + val_len)
        print(
            f"Perform train/val/test split: train {train_len}, val {val_len}, test {test_len}")
        self.train_set, self.val_set, self.test_set = random_split(complete_data,
                                                                   [train_len, val_len, test_len])
        if stage == "fit":
            print("Training with classes", self.train_set.classes)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.workers, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.workers, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.workers, pin_memory=True)
