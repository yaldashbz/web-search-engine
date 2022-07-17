import os
import pickle
import re
from itertools import chain
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from engines.services.data_collection.utils import get_contents, TOKENS_KEY
from engines.services.preprocess import PreProcessor
from engines.services.representation.base import BaseRepresentation


class TFIDFRepresentation(BaseRepresentation):
    def __init__(self, data, tokens_key: str = TOKENS_KEY):
        super().__init__(data, tokens_key)
        contents = self.prepare_data()
        self.tfidf = TfidfVectorizer(use_idf=True, norm='l2', analyzer='word')
        self.matrix = self.tfidf.fit_transform(contents)
        self.vocab = self.tfidf.get_feature_names_out()

    def prepare_data(self) -> List:
        return get_contents(self.data, key=self.tokens_key)

    def represent(self) -> pd.DataFrame:
        return pd.DataFrame(data=self.matrix.toarray(), columns=self.vocab)

    @classmethod
    def process_query(cls, query: str):
        preprocessor = PreProcessor()
        query = re.sub('\\W+', ' ', query).strip()
        return preprocessor.process(query)

    def embed(self, query: str):
        tokens = self.process_query(query)
        n = len(self.vocab)
        vector = np.zeros(n)
        for token in chain(*tokens):
            try:
                index = self.tfidf.vocabulary_[token]
                vector[index] = 1
            except ValueError:
                pass
        return vector
