# @Time    : 2026-01-20 10:19
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : pretrain.py

# Generic pretraining script for any model variant: pretrains on locally
# collected data

from src.data.utils import build_char_map, build_inverse_char_map, char_to_folder_name, load_chars, RAW_MOUSE_DATA_DIR
from src.ml.architectures.cnn import StrokeNet
from src.ml.architectures.lstm import StrokeLSTM
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import torch as t
import argparse
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, PackedSequence
from torch.utils.data import Dataset, DataLoader, random_split
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lstm", help="Model to use (lstm only right now)")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs to train for")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    args = parser.parse_args()

    ###############################
    # Prepare dataset
    ###############################
    data_module = RawMouseDataModule(
        batch_size=args.batch_size,
        T_max=200,
        num_workers=0,
        seed=args.seed
    )
    data_module.setup()

    ###############################
    # Training
    ###############################
    L.seed_everything(args.seed, workers=True)
    if args.model == "lstm":
        model = StrokeLSTM(num_classes=53, hidden_size=64, num_layers=2)
    else:
        raise ValueError(f"Model {args.model} not supported")

    print(
        f"Commencing pretraining with "
        f"{len(data_module.train_dataset)} train / "
        f"{len(data_module.val_dataset)} val / "
        f"{len(data_module.test_dataset)} test "
        f"(total {len(data_module.train_dataset)+len(data_module.val_dataset)+len(data_module.test_dataset)})"
    )

    wandblogger = WandbLogger(project="str(mouse)", tags=["pretrain", "uppercase", args.model])
    ckpt = ModelCheckpoint(
        dirpath=f"checkpoints/{args.model}/pretrain/{wandblogger.experiment.name}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best",
    )

    trainer = L.Trainer(
        min_epochs=5,
        max_epochs=args.epochs,
        accelerator="auto",
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        logger=wandblogger,
        enable_checkpointing=True,
        callbacks=[ckpt],
        check_val_every_n_epoch=1,
    )
    trainer.fit(model, datamodule=data_module)

    trainer.test(
        ckpt_path="best", datamodule=data_module
    )  # auto selects test dataloader


class VariableSequenceDataset(Dataset):
    """
    Stores variable length sequences as list of tensors, so each item is 
    (x: (T, D) tensor, length: int, y : int)
    """
    def __init__(self, xs, lengths, ys):
        super().__init__()
        self.xs = xs
        self.lengths = lengths
        self.ys = ys
        
    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        return self.xs[idx], self.lengths[idx], self.ys[idx]


class RawMouseDataModule(L.LightningDataModule):

    def __init__(
        self,
        cache_dir=RAW_MOUSE_DATA_DIR,
        max_trials=-1,
        T_max=200,  # 200*10ms sample = 2s per char
        batch_size=128,
        num_workers=0,
        seed=42
    ):
        """
        T_max is the max length of any training sample in samples (typically,
        samples are seperated by 10ms)
        """
        super().__init__()
        self.cache_dir = cache_dir
        self.T_max = T_max
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        # use shared char map from utils
        self.char_map = build_char_map()

        # find character with min trials if max_trials <=0
        self.max_trials = max_trials
        min_trial_char = None
        if self.max_trials <= 0:
            self.max_trials = float("inf")
            for char in build_inverse_char_map().keys():
                char_name = char_to_folder_name(char)
                char_dir = os.path.join(self.cache_dir, char_name)
                num_trials = len(os.listdir(char_dir))
                if num_trials < self.max_trials:
                    self.max_trials = num_trials
                    min_trial_char = char
            print(
                f"Using min trials across characters: {self.max_trials} (from '{min_trial_char}')"
            )

    def setup(self, stage=None):
        """
        Prepare dataset:
        1. Load data I've labelled so far
        2. Create PackedSequence datasets for train/val/test
        """
        # Load collected data by reading raw_mouse_data/<char>/*.csv files
        xs, lengths, ys = [], [], []
        
        truncate_count = 0

        for char_idx, char_value in self.char_map.items():
            folder_name = char_to_folder_name(char_value)

            _, trials = load_chars(
                f"{self.cache_dir}/{folder_name}",
                max_trials=self.max_trials,
                samples_last=False,  # (T, 2)
                return_tensor=True,
                silent=True,
            )

            for x in trials:
                if len(x.shape) != 2:
                    raise ValueError(f"Expected (T, D) trial tensor, but got shape {tuple(x.shape)}")

                T = int(x.shape[0])
                # truncate large trials
                if T > self.T_max:
                    x = x[:self.T_max, :]
                    T = self.T_max
                    truncate_count += 1

                xs.append(x)
                lengths.append(T)
                ys.append(char_idx)

        dataset = VariableSequenceDataset(xs, lengths, ys)
        print(f"Trunctaed {truncate_count} trials to length T={self.T_max}")

        # Split into train/val/test (70/10/20)
        total_size = len(dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, 
            [train_size, val_size, test_size],
            generator=t.Generator().manual_seed(self.seed)
        )

    @staticmethod
    def collate_packed(batch) -> tuple[PackedSequence, t.Tensor]:
        """
        Collates a batch into a PackedSequence, sorted desc by sequence length
        batch: list of (x: (T, D), length: int, y: int)

        collate into:
            packed_x: PackedSequence 
                .data is (sum(lengths), D) concatenated tensor
                .batch_sizes is a LongTensor describing length of each sequence
            y_sorted: (B, )
        """
        # splat operator pairs idx i w/ idx i of a two distinct items, so all xs,
        # lens, and ys are grouped
        xs, lengths, ys = zip(*batch) 

        lengths = t.tensor(lengths, dtype=t.long) # int64
        ys = t.tensor(ys, dtype=t.long)  # int64

        # PAD TO LENGTH OF BATCH
        # xs: iterator length B of (T, D) tensors
        # pad_sequence takes a list of tensors (T, *) & pads along a new dim
        # so that they're all equal length (B, T_max, *)
        padded = pad_sequence(xs, batch_first=True, padding_value=0) # (B, T, D)

        lengths_sorted, sort_idx = lengths.sort(descending=True)
        padded_sorted = padded[sort_idx]
        ys_sorted = ys[sort_idx]

        # CONSTRUCT PACKED_SEQUENCES
        padded_x = pack_padded_sequence(
            padded_sorted,
            lengths_sorted,
            batch_first=True,
            enforce_sorted=True
        )

        return padded_x, ys_sorted

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=self.collate_packed,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=self.collate_packed
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_packed,
        )


if __name__ == "__main__":
    main()
