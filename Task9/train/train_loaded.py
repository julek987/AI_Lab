import pytorch_lightning as pl

from Task9.data import CifarDataModule
from Task9.models import load_pretrained_embedding
from Task9.models.classifier import Classifier

embedding = load_pretrained_embedding()

# if frozen
# embedding.requires_grad_(False)

model = Classifier(embedding, 64, 256, 10)
data_module = CifarDataModule()

trainer = pl.Trainer(max_epochs=7)
trainer.fit(model, data_module)
trainer.test(model, data_module)
