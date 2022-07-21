from WebSearchEngine.asgi import (
    tfidf_searcher, boolean_searcher,
    bert_searcher, fasttext_searcher,
    fasttext_cluster, naive_classifier,
    bert_classifier, data
)
from engines.services.link_analyser import ContentLinkAnalyser

_searchers = {
    'tf-idf': tfidf_searcher,
    'boolean': boolean_searcher,
    'bert': bert_searcher,
    'fasttext': fasttext_searcher
}

_clusters = {
    'fasttext': fasttext_cluster
}

_classifiers = {
    'naive': naive_classifier,
    'bert': bert_classifier
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


def classify(query: str, method: str):
    classifier = _classifiers[method]
    label = classifier.classify(query)
    return label


def f1_score(method: str):
    return _classifiers[method].f1_score()


def accuracy(method: str):
    return _classifiers[method].f1_score()


def confusion_matrix(method: str):
    return _classifiers[method].confusion_matrix(plot=False)


def link_analysis(
        doc_indices,
        method: str = 'word',
        sent_num: int = 5,
        min_similar: int = 5
):
    filtered = [data[i] for i in doc_indices]
    dataset = [doc['tokens'] for doc in filtered]
    cleaned_dataset = [doc['cleaned_tokens'] for doc in filtered]
    word_analyser = ContentLinkAnalyser(
        dataset=dataset,
        cleaned_dataset=cleaned_dataset,
        method=method,
        weighted=True,
        sent_num=sent_num,
        min_similar=min_similar
    )
    pagerank = word_analyser.apply_pagerank(clean_output=False)
    hub, authority = word_analyser.apply_hits(clean_output=False)
    return pagerank, hub, authority
