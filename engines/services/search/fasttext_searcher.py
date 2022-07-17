from typing import Optional

from engines.services.data_collection.utils import TOKENS_KEY
from engines.services.representation import FasttextRepresentation
from engines.services.search.base import BaseSearcher
from engines.services.search.utils import DataOut
from engines.services.utils import cosine_sim


class FasttextSearcher(BaseSearcher):
    def __init__(
            self, data,
            train: bool = True,
            load: bool = False,
            min_count: int = 1,
            tokens_key: str = TOKENS_KEY,
            root: str = 'models',
            folder: str = 'fasttext'
    ):
        super().__init__(data, tokens_key)
        self.representation = FasttextRepresentation(
            data, train=train, load=load,
            min_count=min_count, tokens_key=tokens_key,
            root=root, folder=folder
        )

    def search(self, query, k: int = 10) -> Optional[DataOut]:
        query_embedding_avg = self.representation.embed(query)
        similarities = self._get_similarities(query_embedding_avg, k)
        return DataOut(self._get_result(similarities))

    def _get_similarities(self, query_embedding, k):
        similarities = dict()
        for index, embedding in self.representation.doc_embedding_avg.items():
            similarities[index] = cosine_sim(embedding, query_embedding)
        return sorted(similarities.items(), key=lambda x: x[1])[::-1][:k]

    def _get_result(self, similarities):
        return [dict(
            url=self.data[index]['url'],
            score=score
        ) for index, score in similarities]
