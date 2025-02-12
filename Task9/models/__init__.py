import pathlib

_MODELS_DIR = pathlib.Path(__file__).parent / 'saved_models'
_MODELS_DIR.mkdir(exist_ok=True)

EMBEDDING_WEIGHTS = _MODELS_DIR / 'embedding.pt'

from .embedding import Embedding
from .classifier import Classifier


def load_pretrained_embedding() -> Embedding:
    import torch
    model = Embedding()
    model.load_state_dict(torch.load(EMBEDDING_WEIGHTS, weights_only=True))
    return model
