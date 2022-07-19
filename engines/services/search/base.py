import os
from abc import ABC, abstractmethod
from typing import Optional

from engines.services.data_collection.utils import TOKENS_KEY
from engines.services.search.utils import DataOut
from engines.services.preprocess import PreProcessor


class BaseSearcher(ABC):
    def __init__(self, data, tokens_key: str = TOKENS_KEY):
        assert len(data) > 0
        assert tokens_key in data[0].keys()

        self.data = data
        self.tokens_key = tokens_key
        self.pre_processor = PreProcessor()

    @abstractmethod
    def search(self, query, k: int = 10) -> Optional[DataOut]:
        pass
