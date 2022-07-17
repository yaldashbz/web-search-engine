import itertools
import os
from typing import List

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from engines.services.data_collection.utils import get_contents, TOKENS_KEY
from engines.services.preprocess import PreProcessor
from engines.services.representation.base import BaseRepresentation


class BertRepresentation(BaseRepresentation):
    _PATH = '../embeddings'
    _FILE = 'bert_embeddings.json'

    def __init__(self, data, load: bool = False, tokens_key: str = TOKENS_KEY):
        super().__init__(data, tokens_key)

        if load and not os.path.exists(os.path.join(self._PATH, self._FILE)):
            raise ValueError

        self.model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        self._to_cuda()

        if not load and data:
            contents = self.prepare_data()
            self.df = self._get_embeddings(contents)
            self._save_embeddings()
        else:
            self.df = self._load_embeddings()
        self.embeddings = np.asarray(self.df.values.tolist()).astype('float32')

    def _load_embeddings(self):
        return pd.read_json(os.path.join(self._PATH, self._FILE))

    def _save_embeddings(self):
        if not os.path.exists(self._PATH):
            os.mkdir(self._PATH)
        self.df.to_json(os.path.join(self._PATH, self._FILE))

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
            query, show_progress_bar=True, normalize_embeddings=True
        )

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
