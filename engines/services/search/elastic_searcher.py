from copy import deepcopy
from typing import Optional

from elasticsearch import Elasticsearch
from tqdm import tqdm

from WebSearchEngine.settings import CLOUD_ID, ELASTIC_USER, ELASTIC_PASSWORD
from engines.services.data_collection.utils import get_content
from engines.services.search import BaseSearcher, DataOut


class ElasticSearcher(BaseSearcher):
    _INDEX = 'pages'

    def __init__(self, data, tokens_key: str = 'tokens'):
        super().__init__(data, tokens_key)
        self.client = Elasticsearch(
            cloud_id=CLOUD_ID,
            basic_auth=(ELASTIC_USER, ELASTIC_PASSWORD)
        )

    def index_data(self, tokens_key: str):
        data = deepcopy(self.data)
        for doc in data:
            doc['content'] = get_content(doc[tokens_key])
            doc.pop('cleaned_tokens')
            doc.pop('tokens')
            doc.pop('keywords')
        for i, doc in tqdm(enumerate(data)):
            self.client.index(index=self._INDEX, id=i, document=doc)

    def delete_index(self):
        self.client.indices.delete(index=self._INDEX, ignore=[400, 404])

    def search(self, query, k: int = 10, _from: int = 0) -> Optional[DataOut]:
        body = {
            'from': 0,
            'size': k,
            'query': {
                'match': {
                    'content': query
                }
            }
        }
        out = self.client.search(index=self._INDEX, body=body)
        return DataOut(self._get_result(out))

    @classmethod
    def _get_result(cls, out):
        res = out['hits']['hits']
        return [dict(
            index=doc['_index'],
            score=doc['_score'],
            content=doc['_source']['sentence']
        ) for doc in res]
