import torch
from torch import nn
import pytorch_lightning as pl

class Embedding(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            # Input: 3 x 32 x 32

            # Conv Layer 1: 3 -> 16, keeps spatial size with padding=1, then downsample to 16x16
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: 16 x 16 x 16

            # Conv Layer 2: 16 -> 32, spatial size remains 16x16, then downsample to 8x8
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: 32 x 8 x 8

            # Conv Layer 3: 32 -> 64, spatial size remains 8x8, then downsample to 4x4
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: 64 x 4 x 4

            # Conv Layer 4: 64 -> 32, keep spatial size at 4x4
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Conv Layer 5: 32 -> 16, keep spatial size at 4x4, then downsample to 2x2
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: 16 x 2 x 2

            # Flatten: 16 * 2 * 2 = 64 features (which is <= 100)
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
