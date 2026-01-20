# @Time    : 2026-01-17 19:17
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : lstm.py

# Defines an LSTM model for basic testing of a time series approach to the task

import torch as t
import torch.nn as nn
from torch import optim
import lightning as L
from torch.nn.utils.rnn import PackedSequence

class StrokeLSTM(L.LightningModule):
    """
    LSTM for classifying mouse *velocity* data into letters (currently 53, but
    most up to date in data.utils.build_char_map())
    
    Input: (B, T, 2) PackedSequence tensor
    Output: logits over num_classes
    
    There is no seperation of pretraining and finetuning for this module, as I
    lack access to an abundant pretraining dataset for this time series task. 
    """
    def __init__(
        self, 
        num_classes: int = 53, 
        hidden_size=64, # control model size
        num_layers=2 # size of hidden layer stack
    ):
        super().__init__()
        self.save_hyperparameters()

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(2,
                            hidden_size,
                            num_layers,
                            bias=True,
                            batch_first=True)
        self.fcc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: PackedSequence):
        output, (h_n, c_n) = self.lstm(x)
        # output is PackedSequence with all timesteps outputs
        # h_n is (num_layers, B, hidden_size) tensor
        # we only want the last layer's hidden state for stroke classification
        last_hidden = h_n[-1] # (B, hidden_size)
        return self.fcc(last_hidden)

    def training_step(self, batch, batch_idx):
        x, labels = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, labels)
        preds = t.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, labels = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, labels)
        preds = t.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)

        return optimizer
