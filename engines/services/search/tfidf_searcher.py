from typing import Optional

import numpy as np

from engines.services.data_collection.utils import TOKENS_KEY
from engines.services.representation import TFIDFRepresentation
from engines.services.search.base import BaseSearcher
from engines.services.search.utils import DataOut
from engines.services.utils import cosine_sim


class TFIDFSearcher(BaseSearcher):
    def __init__(self, data, tokens_key: str = TOKENS_KEY):
        super().__init__(data, tokens_key)
        self.representation = TFIDFRepresentation(data, tokens_key=tokens_key)

    def search(self, query, k: int = 10) -> Optional[DataOut]:
        scores = list()
        vector = self.representation.embed(query)
        for doc in self.representation.matrix.A:
            scores.append(cosine_sim(vector, doc))

        return DataOut(self._get_results(scores, k))

    def _get_results(self, scores, k):
        out = np.array(scores).argsort()[-k:][::-1]
        return [dict(
            index=index,
            url=self.data[index]['url'],
            score=scores[index]
        ) for index in out]
