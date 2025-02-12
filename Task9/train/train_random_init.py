import pytorch_lightning as pl

from Task9.data import CifarDataModule
from Task9.models import Embedding
from Task9.models.classifier import Classifier

model = Classifier(Embedding(), 64, 256, 10)
data_module = CifarDataModule()

trainer = pl.Trainer(max_epochs=7)
trainer.fit(model, data_module)
trainer.test(model, data_module)
