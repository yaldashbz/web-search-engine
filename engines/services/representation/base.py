from abc import abstractmethod, ABC
from typing import List

import pandas as pd

from engines.services.data_collection.utils import TOKENS_KEY


class BaseRepresentation(ABC):
    def __init__(self, data, tokens_key: str = TOKENS_KEY):
        self.data = data
        self.tokens_key = tokens_key

    def prepare_data(self) -> List:
        return self.data

    @abstractmethod
    def represent(self) -> pd.DataFrame:
        raise NotImplementedError

    @classmethod
    def process_query(cls, query: str):
        raise NotImplementedError

    def embed(self, query: str):
        raise NotImplementedError
