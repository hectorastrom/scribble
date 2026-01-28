# @Time    : 2026-01-12 21:18
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : cnn.py

import torch as t
from torch import optim
import torch.nn as nn
import lightning as L

class StrokeNet(L.LightningModule):
    """
    Lightning CNN for classifying mouse strokes into characters.

    Input: Bx1x28x28 tensor
    Output: logits over num_classes

    Training approach:
        - Pretrain: 62 classes (EMNIST ByClass: a-z, A-Z, 0-9)
        - Finetune: Discard head, create new 63+ class head (adds space, future custom strokes)

    Index scheme for finetune (63 base classes):
        a-z: 0-25, A-Z: 26-51, 0-9: 52-61, space: 62
    """
    def __init__(self, num_classes: int = 63, dropout_p: float = 0.10, finetune=False, class_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        
        # class_weights for handling imbalanced datasets (set after init or via set_class_weights)
        self.class_weights = class_weights

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.MaxPool2d(2, 2)  # 28 -> 14

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.MaxPool2d(2, 2)  # 14 -> 7

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.down3 = nn.MaxPool2d(2, 2)  # 7 -> 3

        # regularization after convs
        self.dropout2d = nn.Dropout2d(p=dropout_p)

        # transforms to 1x1 image
        self.gap = nn.AdaptiveAvgPool2d((1, 1)) # 3 -> 1

        # Small classifier head
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        # didn't add residual connections: only really relevant for very deep nets
        x = self.block1(x)
        x = self.down1(x)

        x = self.block2(x)
        x = self.down2(x)

        x = self.block3(x)
        x = self.down3(x)

        x = self.dropout2d(x)
        x = self.gap(x)  # (B, 128, 1, 1)
        x = t.flatten(x, 1)  # (B, 128)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        # use class weights if provided (for imbalanced datasets)
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss = nn.CrossEntropyLoss(weight=weights)(logits, labels)
        else:
            loss = nn.CrossEntropyLoss()(logits, labels)
        self.log('train_loss', loss)
        return loss
    
    def set_class_weights(self, class_weights):
        """Set class weights for handling imbalanced datasets."""
        self.class_weights = class_weights

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = nn.CrossEntropyLoss()(logits, labels)
        preds = t.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = nn.CrossEntropyLoss()(logits, labels)
        preds = t.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def replace_classifier(self, num_classes: int):
        """
        Replace classifier head with a new randomly initialized head for
        finetuning 
        """
        self.hparams.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(128, num_classes),
        )

    def configure_optimizers(self):
        """
        Configure optimizer for training and finetuning.
        
        Training: AdamW with lr=1e-3, weight_decay=1e-4
        Finetuning: Adam with lr=1e-4 only on block3 and classifier (new head)
        """
        if self.hparams.finetune:
            print("Configuring optimizer for finetuning...")
            params_to_optimize = list(self.block3.parameters()) + list(self.classifier.parameters())
            optimizer = optim.Adam(params_to_optimize, lr=1e-4)
            return optimizer
        else:
            print("Configuring optimizer for pretraining...")
            optimizer = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)

        return optimizer
