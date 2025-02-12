import pytorch_lightning as pl
import torch

from Task9.data import CifarTripletDataModule
from Task9.models import Embedding, EMBEDDING_WEIGHTS

model = Embedding()
data_module = CifarTripletDataModule()

trainer = pl.Trainer(max_epochs=7)
trainer.fit(model, data_module)

torch.save(model.state_dict(), EMBEDDING_WEIGHTS)
