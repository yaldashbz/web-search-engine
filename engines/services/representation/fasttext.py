import os
import re
from itertools import chain

import numpy as np
import pandas as pd
from gensim.models.fasttext import FastText
from tqdm import tqdm

from engines.services.data_collection.utils import get_doc_words, TOKENS_KEY
from engines.services.representation import BaseRepresentation, PreProcessor


class FasttextRepresentation(BaseRepresentation):
    _EPOCHS = 6
    _MODEL_FILE = 'fasttext.model'

    def __init__(
            self, data,
            train: bool = True,
            load: bool = False,
            min_count: int = 4,
            tokens_key: str = TOKENS_KEY,
            root: str = 'models',
            folder: str = 'fasttext'
    ):
        assert load != train

        super().__init__(data, tokens_key)
        self.mkdir(root)
        self.mkdir(os.path.join(root, folder))

        model_file = os.path.join(root, self._MODEL_FILE)
        if not (train or os.path.exists(model_file)):
            raise ValueError

        self.fasttext = self._get_fasttext(min_count, model_file, load)
        if data and train and not load:
            self._train()
            self._save_model(model_file)
        self.doc_embedding_avg = self._get_doc_embedding_avg()

    @classmethod
    def _get_fasttext(cls, min_count: int, path: str, load: bool):
        return FastText(
            sg=1, window=10, min_count=min_count,
            negative=15, min_n=2, max_n=5
        ) if not load else FastText.load(path)

    def _train(self):
        tokens = [get_doc_words(doc, key=self.tokens_key) for doc in self.data]
        self.fasttext.build_vocab(tokens)
        self.fasttext.train(
            tokens,
            epochs=self._EPOCHS,
            total_examples=self.fasttext.corpus_count,
            total_words=self.fasttext.corpus_total_words
        )

    def _save_model(self, path):
        self.fasttext.save(path)

    def _get_doc_embedding_avg(self):
        docs_avg = dict()
        for index, doc in tqdm(enumerate(self.data)):
            words = get_doc_words(doc, key=self.tokens_key)
            docs_avg[index] = np.mean([
                self.fasttext.wv[word] for word in words if re.fullmatch('\\w+', word)
            ], axis=0)
        return docs_avg

    def represent(self) -> pd.DataFrame:
        return pd.DataFrame(
            data=self.doc_embedding_avg.values(),
            index=self.doc_embedding_avg.keys()
        )

    @classmethod
    def process_query(cls, query: str):
        preprocessor = PreProcessor()
        tokens = preprocessor.process(query)
        return list(chain(*tokens))

    def embed(self, query: str):
        tokens = self.process_query(query)
        return np.mean([
            self.fasttext.wv[token] for token in tokens
        ], axis=0)
