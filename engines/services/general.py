from WebSearchEngine.asgi import (
    tfidf_searcher, boolean_searcher, bert_searcher, fasttext_searcher, fasttext_cluster
)

_searchers = {
    'tf-idf': tfidf_searcher,
    'boolean': boolean_searcher,
    'bert': bert_searcher,
    'fasttext': fasttext_searcher
}

_clusters = {
    'fasttext': fasttext_cluster
}


def search(query: str, method: str, k: int):
    searcher = _searchers[method]
    return searcher.search(query, k)


def cluster(query: str, method: str, k: int):
    _cluster = _clusters[method]
    cluster_id = _cluster.cluster(query)
    similar = _cluster.get_clustered_data(cluster_id)
    return cluster_id, similar[:k]


def rss(method: str):
    return _clusters[method].rss_evaluate()


def silhouette(method: str):
    return _clusters[method].silhouette_evaluate()
