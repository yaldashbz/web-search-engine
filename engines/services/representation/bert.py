import itertools
import os
import pickle
from typing import List

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from engines.services.data_collection.utils import get_contents, TOKENS_KEY
from engines.services.preprocess import PreProcessor
from engines.services.representation.base import BaseRepresentation
from engines.services.utils import check_mkdir


class BertRepresentation(BaseRepresentation):
    _FILE = 'bert_embeddings.json'
    _MODEL = 'distilbert.pkl'

    def __init__(
            self, data,
            load: bool = False,
            tokens_key: str = TOKENS_KEY,
            root: str = 'models',
            folder: str = 'bert'
    ):
        super().__init__(data, tokens_key)
        if load and not os.path.exists(os.path.join(root, folder, self._FILE)):
            raise ValueError
        check_mkdir(root)
        path = os.path.join(root, folder)
        check_mkdir(path)
        self.model = self._get_model(load, os.path.join(path, self._MODEL))
        self._to_cuda()

        if not load and data:
            contents = self.prepare_data()
            self.df = self._get_embeddings(contents)
            self._save_embeddings(path)
        else:
            self.df = self._load_embeddings(path)
        self.embeddings = np.asarray(self.df.values.tolist()).astype('float32')

    @classmethod
    def _get_model(cls, load: bool, model_path: str):
        if not load:
            model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
            pickle.dump(model.to('cpu'), open(model_path, 'wb'))
            return model
        else:
            return pickle.load(open(model_path, 'rb'))

    def _load_embeddings(self, path):
        return pd.read_json(os.path.join(path, self._FILE))

    def _save_embeddings(self, path):
        self.df.to_json(os.path.join(path, self._FILE))

    def prepare_data(self) -> List:
        return get_contents(self.data, key=self.tokens_key)

    def represent(self) -> pd.DataFrame:
        return self.df

    @classmethod
    def process_query(cls, query: str):
        preprocessor = PreProcessor()
        tokens = preprocessor.process(query)
        tokens = itertools.chain(*tokens)
        query = ' '.join(tokens)
        return [query]

    def embed(self, query: str):
        query = self.process_query(query)
        return self.model.encode(
            query,
            show_progress_bar=True,
            normalize_embeddings=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        ).reshape(-1)

    def _to_cuda(self):
        if torch.cuda.is_available():
            self.model = self.model.to(torch.device('cuda'))

    def _get_embeddings(self, contents):
        embeddings = self.model.encode(
            contents,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        return pd.DataFrame(embeddings)
