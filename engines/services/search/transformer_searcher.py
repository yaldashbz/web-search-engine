from typing import Optional

import faiss
import numpy as np

from engines.services.data_collection.utils import TOKENS_KEY
from engines.services.representation import BertRepresentation
from engines.services.search.base import BaseSearcher
from engines.services.search.utils import DataOut


class TransformerSearcher(BaseSearcher):
    def __init__(
            self, data,
            load: bool = False,
            tokens_key: str = TOKENS_KEY
    ):
        super().__init__(data, tokens_key)
        self.representation = BertRepresentation(
            data=data, load=load, tokens_key=tokens_key
        )
        self.index = self._get_index(self.representation.embeddings)

    def _get_index(self, embeddings):
        index = faiss.IndexIDMap(faiss.IndexFlatL2(embeddings.shape[1]))
        index.add_with_ids(embeddings, np.array(range(len(self.data))))
        return index

    def search(self, query, k: int = 10) -> Optional[DataOut]:
        vector = self.representation.embed(query)
        distances, indexes = self.index.search(np.array(vector).astype('float32'), k=k)
        return DataOut(self._get_results(distances, indexes))

    def _get_results(self, distances, indexes):
        indexes = indexes.flatten().tolist()
        distances = distances.flatten().tolist()
        return [dict(
            url=self.data[index]['url'],
            score=1 - distances[i] / 2
        ) for i, index in enumerate(indexes)]
