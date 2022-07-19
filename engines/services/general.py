from WebSearchEngine.asgi import (
    tfidf_searcher, boolean_searcher, bert_searcher, fasttext_searcher
)

_searchers = {
    'tf-idf': tfidf_searcher,
    'boolean': boolean_searcher,
    'bert': bert_searcher,
    'fasttext': fasttext_searcher
}


def search(query: str, method: str, k: int):
    searcher = _searchers[method]
    return searcher.search(query, k)
