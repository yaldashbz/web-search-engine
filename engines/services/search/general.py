import json

from engines.services.search import (
    TFIDFSearcher,
    BooleanSearcher,
    TransformerSearcher,
    FasttextSearcher,
    BaseSearcher
)


def get_searcher(method: str):
    clean_path = 'clean_data.json'
    data = json.load(open(clean_path, 'r'))
    searcher = None
    if method == 'tf-idf':
        searcher = TFIDFSearcher(data=data, tokens_key='cleaned_tokens')
    if method == 'boolean':
        searcher = BooleanSearcher(data=data, build=False, tokens_key='cleaned_tokens')
    if method == 'bert':
        searcher = TransformerSearcher(data=data, load=False, tokens_key='cleaned_tokens')
    if method == 'fasttext':
        searcher = FasttextSearcher(data=data, train=True, load=False, min_count=4, tokens_key='cleaned_tokens')
    return searcher


def get_result(searcher: BaseSearcher, query: str, k: int):
    return searcher.search(query, k)


def search(query: str, method: str, k: int):
    searcher = get_searcher(method)
    return get_result(searcher, query, k)
