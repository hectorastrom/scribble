# @Time    : 2026-01-13 08:04
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : pretrain.py

# Pretrain StrokeNet on EMNIST dataset
#
# Two dataset options:
#   - "byclass" (default): 62 classes (a-z, A-Z, 0-9) - proper case/digit separation
#   - "letters" (legacy): 26 merged classes (upper+lower combined) - for backward compat
#
# Pretrain with 62 classes, then finetune discards head and creates
# a new 63+ class head (adds space, future custom strokes).

from datasets import load_dataset
import matplotlib.pyplot as plt
from PIL import Image
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from src.ml.architectures.cnn import StrokeNet
from src.data.utils import NUM_PRETRAIN_CLASSES, build_char_map
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--img-size", type=int, default=28, help="Image size (default 28x28)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="byclass",
        choices=["byclass", "letters"],
        help="EMNIST dataset: 'byclass' (62 classes, default) or 'letters' (26 merged, legacy)"
    )
    parser.add_argument("--max-epochs", type=int, default=20, help="Max epochs to train")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    args = parser.parse_args()

    ###############################
    # Prepare EMNIST dataset
    ###############################
    print(f"Preparing EMNIST {args.dataset} dataset...")
    
    if args.dataset == "byclass":
        # 62 classes (a-z, A-Z, 0-9)
        data_module = EMNISTByClassDataModule(
            batch_size=args.batch_size,
            num_workers=0,
            downsample_size=args.img_size,
            augment=not args.no_augment,
        )
        num_classes = NUM_PRETRAIN_CLASSES  # 62
        wandb_tags = ["pretrain", "byclass-dset"]
    else:
        # Legacy: 26 merged classes (upper+lower combined)
        data_module = EMNISTLettersDataModule(
            batch_size=args.batch_size,
            num_workers=0,
            downsample_size=args.img_size
        )
        num_classes = 27  # 26 letters + space (legacy behavior)
        wandb_tags = ["pretrain", "letters-dset"]
    
    if not args.no_augment and args.dataset == "byclass":
        wandb_tags.append("augmented")
    
    data_module.prepare_data()
    data_module.setup()

    # Debug: display sample images from the dataset
    display_samples(data_module)

    ###############################
    # Training CNN
    ###############################
    train_size = len(data_module.train_data)
    print(f"Commencing training with {train_size} training samples, {num_classes} classes...")
    L.seed_everything(42, workers=True)
    model = StrokeNet(num_classes=num_classes, dropout_p=0.1, finetune=False)
    wandblogger = WandbLogger(project="scribble", tags=wandb_tags)

    ckpt = ModelCheckpoint(
        dirpath=f"checkpoints/emnist_pretrain/{wandblogger.experiment.name}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best",
    )

    trainer = L.Trainer(
        min_epochs=5,
        max_epochs=args.max_epochs,
        accelerator="auto",
        limit_train_batches=1.0, # portion of train data to use
        limit_val_batches=1.0, # portion of test data to use
        logger=wandblogger,
        enable_checkpointing=True,
        callbacks=[ckpt],
        check_val_every_n_epoch=1, # val & checkpoint this frequency
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

    print("\nPretraining complete!")
    print(f"Dataset: {args.dataset} ({num_classes} classes)")
    print(f"Data augmentation: {'enabled' if not args.no_augment else 'disabled'}")
    print(
        f"Pretrained model checkpoints saved to checkpoints/emnist_pretrain/{wandblogger.experiment.name}"
    )


def display_samples(data_module, num_samples: int = 9):
    """
    Display sample images from a data module.
    
    Expects data_module to have:
        - train_data: list of (image, label) tuples where image is numpy array (28x28)
        - id2label: dict mapping label index to character
    
    EMNIST images need 90 deg clockwise rotation and horizontal flip to display correctly.
    """
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        if i >= len(data_module.train_data):
            ax.axis("off")
            continue
        
        img, label_idx = data_module.train_data[i]
        # rotate 90 deg clockwise and flip horizontally (EMNIST orientation fix)
        img = np.rot90(img, k=-1)
        img = np.fliplr(img)
        
        label_char = data_module.id2label.get(label_idx, f"?{label_idx}")
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Label: {label_char}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# IDX file format parsers (MNIST/EMNIST ubyte format)
def read_idx_images(path: str) -> np.ndarray:
    """
    Parse IDX3 format ubyte file containing images.
    
    Returns:
        np.ndarray of shape (num_images, rows, cols) with uint8 pixel values
    """
    with open(path, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic}, expected 2051 for images")
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num_images, rows, cols)


def read_idx_labels(path: str) -> np.ndarray:
    """
    Parse IDX1 format ubyte file containing labels.
    
    Returns:
        np.ndarray of shape (num_labels,) with uint8 label values
    """
    with open(path, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic}, expected 2049 for labels")
        num_labels = int.from_bytes(f.read(4), 'big')
        return np.frombuffer(f.read(), dtype=np.uint8)


def remap_emnist_byclass_label(emnist_label: int) -> int:
    """
    Remap EMNIST ByClass label to our internal index scheme.
    
    EMNIST ByClass mapping (from emnist-byclass-mapping.txt):
        0-9   -> ASCII 48-57  -> digits '0'-'9'
        10-35 -> ASCII 65-90  -> uppercase 'A'-'Z'
        36-61 -> ASCII 97-122 -> lowercase 'a'-'z'
    
    Our internal scheme:
        0-25:  lowercase a-z
        26-51: uppercase A-Z
        52-61: digits 0-9
    
    Returns:
        Remapped label index in our scheme
    """
    if 0 <= emnist_label <= 9:
        # EMNIST digits 0-9 -> our indices 52-61
        return emnist_label + 52
    elif 10 <= emnist_label <= 35:
        # EMNIST uppercase A-Z (10-35) -> our indices 26-51
        return emnist_label - 10 + 26
    elif 36 <= emnist_label <= 61:
        # EMNIST lowercase a-z (36-61) -> our indices 0-25
        return emnist_label - 36
    else:
        raise ValueError(f"Invalid EMNIST ByClass label: {emnist_label}")


class EMNISTByClassDataModule(L.LightningDataModule):
    """
    DataModule for EMNIST ByClass dataset (62 classes: a-z, A-Z, 0-9).
    
    Reads from local ubyte files and remaps labels to our internal scheme:
        a-z: 0-25, A-Z: 26-51, 0-9: 52-61
    
    This is the recommended dataset for pretraining as it properly separates
    uppercase/lowercase letters and includes digits.
    """
    def __init__(
        self,
        data_dir: str = "./emnist_data/by_class",
        batch_size: int = 64,
        num_workers: int = 0,
        downsample_size: int = 28,
        augment: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.downsample_size = downsample_size
        self.augment = augment
        self.to_tensor = transforms.ToTensor()
        
        # data augmentation for training applied in collate
        self.train_transform = transforms.Compose([
            transforms.RandomRotation(degrees=15, fill=0),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # up to 10% shift
                scale=(0.9, 1.1),      # up to 10% scale
                fill=0,
            ),
            transforms.RandomResizedCrop(
                size=downsample_size,
                scale=(0.85, 1.0),
                ratio=(0.9, 1.1),
            ),
        ]) if augment else None
        
        # will be set in setup()
        # train_data/val_data: list of (image, label) tuples where image is numpy (28x28)
        self.train_data = None
        self.val_data = None
        self.id2label = None
    
    def prepare_data(self):
        """Verify that ubyte files exist."""
        required_files = [
            "emnist-byclass-train-images-idx3-ubyte",
            "emnist-byclass-train-labels-idx1-ubyte",
            "emnist-byclass-test-images-idx3-ubyte",
            "emnist-byclass-test-labels-idx1-ubyte",
        ]
        for fname in required_files:
            path = os.path.join(self.data_dir, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"EMNIST ByClass file not found: {path}\n"
                    f"Please download the dataset from https://www.nist.gov/itl/products-and-services/emnist-dataset"
                )
    
    def setup(self, stage=None):
        """Load and preprocess the dataset."""
        # load raw data from ubyte files
        train_images_raw = read_idx_images(
            os.path.join(self.data_dir, "emnist-byclass-train-images-idx3-ubyte")
        )
        train_labels_raw = read_idx_labels(
            os.path.join(self.data_dir, "emnist-byclass-train-labels-idx1-ubyte")
        )
        val_images_raw = read_idx_images(
            os.path.join(self.data_dir, "emnist-byclass-test-images-idx3-ubyte")
        )
        val_labels_raw = read_idx_labels(
            os.path.join(self.data_dir, "emnist-byclass-test-labels-idx1-ubyte")
        )
        
        print(f"Loaded {len(train_images_raw)} train and {len(val_images_raw)} val images")
        
        # remap labels to our internal scheme and store as (image, label) tuples
        train_labels = [remap_emnist_byclass_label(l) for l in train_labels_raw]
        val_labels = [remap_emnist_byclass_label(l) for l in val_labels_raw]
        self.train_data = list(zip(train_images_raw, train_labels))
        self.val_data = list(zip(val_images_raw, val_labels))
        
        # build id2label mapping using our char_map (for 62 pretrain classes, no space)
        char_map = build_char_map()
        self.id2label = {i: char_map[i] for i in range(NUM_PRETRAIN_CLASSES)}
        
        # print label distribution summary
        unique, counts = np.unique(train_labels, return_counts=True)
        print(f"Label distribution: {len(unique)} unique classes")
        print(f"  Samples per class: min={counts.min()}, max={counts.max()}, mean={counts.mean():.0f}")
    
    def _transform_image(self, img: np.ndarray) -> torch.Tensor:
        """
        Transform a single image: rotate, flip, resize, convert to tensor.
        
        EMNIST images need 90 deg clockwise rotation and horizontal flip
        to match standard orientation.
        """
        # rotate 90 deg clockwise and flip horizontally
        img = np.rot90(img, k=-1)
        img = np.fliplr(img)
        
        # convert to PIL for resize if needed
        pil_img = Image.fromarray(img)
        if self.downsample_size != 28:
            pil_img = pil_img.resize((self.downsample_size, self.downsample_size))
        
        return self.to_tensor(pil_img)
    
    def _collate_fn(self, batch):
        """Collate function that transforms images (no augmentation)."""
        images, labels = zip(*batch)
        xs = [self._transform_image(img) for img in images]
        x = torch.stack(xs, dim=0)
        y = torch.tensor(labels, dtype=torch.long)
        return x, y
    
    def _train_collate_fn(self, batch):
        """Collate function with augmentation for training batches."""
        images, labels = zip(*batch)
        xs = [self._transform_image(img) for img in images]
        if self.train_transform:
            xs = [self.train_transform(img) for img in xs]
        x = torch.stack(xs, dim=0)
        y = torch.tensor(labels, dtype=torch.long)
        return x, y
    
    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,  # bad on mps
            collate_fn=self._train_collate_fn,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=self._collate_fn,
        )
    
    def test_dataloader(self):
        return self.val_dataloader()



class EMNISTLettersDataModule(L.LightningDataModule):
    """
    DataModule for EMNIST Letters dataset (26 classes: a-z, for both uppercase
    and lowercase drawn characters).
    
    Not recommended for new work - use EMNISTByClassDataModule instead.
    """
    def __init__(self, cache_dir="./emnist_data", batch_size=64, num_workers=0, downsample_size=28):
        super().__init__()
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.downsample_size = downsample_size
        self.to_tensor = transforms.ToTensor()

        # will be set in setup()
        # train_data/val_data: list of (image, label) tuples where image is numpy (28x28)
        self.train_data = None
        self.val_data = None
        self.id2label = None

    def prepare_data(self):
        load_dataset("tanganke/emnist_letters", cache_dir=self.cache_dir)

    def setup(self, stage=None):
        dataset = load_dataset("tanganke/emnist_letters", cache_dir=self.cache_dir)
        
        # convert HuggingFace format to standard (image_numpy, label) tuples
        # HuggingFace stores PIL images, we convert to numpy for consistency
        self.train_data = [
            (np.array(ex["image"]), int(ex["label"]))
            for ex in dataset["train"]
        ]
        self.val_data = [
            (np.array(ex["image"]), int(ex["label"]))
            for ex in dataset["test"]
        ]
        
        # build id2label as dict for consistency with ByClass module
        label_names = dataset["train"].features["label"].names
        self.id2label = {i: name for i, name in enumerate(label_names)}
        
        print(f"Loaded {len(self.train_data)} train and {len(self.val_data)} val images")

    def _transform_image(self, img: np.ndarray) -> torch.Tensor:
        """Transform image: rotate, flip, resize, convert to tensor."""
        # rotate 90 deg clockwise and flip horizontally
        img = np.rot90(img, k=-1)
        img = np.fliplr(img)
        
        pil_img = Image.fromarray(img)
        if self.downsample_size != 28:
            pil_img = pil_img.resize((self.downsample_size, self.downsample_size))
        
        return self.to_tensor(pil_img)

    def _collate_fn(self, batch):
        """Collate function that transforms images."""
        images, labels = zip(*batch)
        xs = [self._transform_image(img) for img in images]
        x = torch.stack(xs, dim=0)
        y = torch.tensor(labels, dtype=torch.long)
        return x, y

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,  # bad on mps
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self):
        return self.val_dataloader()


if __name__ == "__main__":
    main()
