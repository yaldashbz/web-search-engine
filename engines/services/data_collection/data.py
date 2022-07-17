import json
from dataclasses import dataclass
from typing import List, Tuple

from engines.services.data_collection.utils import get_keywords
from engines.services.preprocess import PreProcessor


@dataclass
class Document:
    url: str
    tokens: List[List[str]]
    keywords: List[Tuple]
    content: str
    category: str

    def __init__(self, url, content, category):
        self.url = url
        self.content = content
        preprocessor = PreProcessor()
        self.tokens = preprocessor.tokenize(content)
        keyword_tokens = preprocessor.normalize(self.tokens, stopwords_removal=True)
        self.keywords = get_keywords(keyword_tokens)
        self.category = category

    def __hash__(self):
        return hash(f'{self.url} - {self.content}')

    @classmethod
    def _convert(cls, data: List) -> List:
        return [{
            'url': doc.url,
            'tokens': doc.tokens,
            'keywords': doc.keywords,
            'category': doc.category
        } for doc in data]

    @classmethod
    def _cleanup(cls, data: List) -> List:
        return [doc for doc in data if doc.tokens]

    @classmethod
    def save(cls, data: List, path: str):
        data = cls._cleanup(data)
        json.dump(cls._convert(data), open(path, 'w+'))
