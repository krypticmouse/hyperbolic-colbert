import os
import torch
import sys

from colbert.utils.utils import torch_load_dnn

from transformers import AutoTokenizer
from colbert.modeling.hf_colbert import class_factory
from colbert.infra.config import ColBERTConfig
from colbert.parameters import DEVICE
from hypere.layers import PoincareProjection


class BaseColBERT(torch.nn.Module):
    """
    Shallow module that wraps the ColBERT parameters, custom configuration, and underlying tokenizer.
    This class provides direct instantiation and saving of the model/colbert_config/tokenizer package.

    Like HF, evaluation mode is the default.
    """

    def __init__(self, name_or_path, colbert_config=None):
        super().__init__()

        self.colbert_config = ColBERTConfig.from_existing(ColBERTConfig.load_from_checkpoint(name_or_path), colbert_config)
        self.name = self.colbert_config.model_name or name_or_path

        # don't assume and raise an error when needed
        assert self.name is not None
        HF_ColBERT = class_factory(self.name)
        
        self.model = HF_ColBERT.from_pretrained(name_or_path, colbert_config=self.colbert_config)
        
        # freeze the model
        for param in self.model.parameters():
            param.requires_grad = False

        self.projection = PoincareProjection(colbert_config.dim, colbert_config.projection_dim)
        try:
            self.projection = self.projection.load_pretrained(name_or_path)
        except FileNotFoundError as e:
            print(f"Could not load projection from {name_or_path}. Initializing from scratch.")

        self.model.to(DEVICE)
        self.projection = self.projection.to(DEVICE)
        self.raw_tokenizer = AutoTokenizer.from_pretrained(name_or_path)

        self.eval()

    @property
    def device(self):
        return self.model.device

    @property
    def bert(self):
        return self.model.LM

    @property
    def linear(self):
        return self.model.linear

    @property
    def score_scaler(self):
        return self.model.score_scaler

    def save(self, path):
        self.model.save_pretrained(path)
        self.projection.save_pretrained(path)

        self.raw_tokenizer.save_pretrained(path)
        self.colbert_config.save_for_checkpoint(path)
