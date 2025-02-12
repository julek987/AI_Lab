import pytorch_lightning as pl

from Task9.data import CifarTripletDataModule
from Task9.models import Classifier, Embedding

model = Classifier(Embedding(), 1024, 256, 10)
data_module = CifarTripletDataModule()

trainer = pl.Trainer(max_epochs=7)
trainer.fit(model, data_module)
trainer.test(model, data_module)
