import torch
from torch import nn
import pytorch_lightning as pl

class Embedding(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            # 3 x 32 x 32 (3072)
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1), # 64 x 32 x 32
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2), # 64 x 16 x 16
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=5, padding=2), # 16 x 16 x 16
            nn.MaxPool2d(2), # 16 x 8 x 8
            nn.Flatten()
        )

        self.loss = nn.TripletMarginLoss()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        anchor, positive, negative = batch

        anchor_embeddings = self(anchor)
        positive_embeddings = self(positive)
        negative_embeddings = self(negative)

        loss = self.loss(anchor_embeddings, positive_embeddings, negative_embeddings)
        self.log('train/loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss
