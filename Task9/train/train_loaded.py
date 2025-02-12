import pytorch_lightning as pl

from Task9.data import CifarTripletDataModule
from Task9.models import load_pretrained_embedding, Classifier

embedding = load_pretrained_embedding()

# jeżeli frozen
# embedding.requires_grad_(False)

model = Classifier(embedding, 1024, 256, 10)
data_module = CifarTripletDataModule()

trainer = pl.Trainer(max_epochs=7)
trainer.fit(model, data_module)
trainer.test(model, data_module)
