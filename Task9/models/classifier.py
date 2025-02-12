import pytorch_lightning as pl
from torch import nn

from Task9.models import Embedding


class Classifier(pl.LightningModule):
    def __init__(self, embedding: Embedding, embedding_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()

        self.embedding = embedding
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        embedding = self.embedding(x)
        output = self.mlp(embedding)
        return output

    def training_step(self, batch, batch_idx):
        img, labels = batch
        logits = self(img)
        loss = self.loss(logits, labels)
        self.log('train/loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        img, labels = batch
        logits = self(img)
        loss = self.loss(logits, labels)
        self.log('test/loss', loss)
