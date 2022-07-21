"""
ASGI config for WebSearchEngine project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.0/howto/deployment/asgi/
"""
import json
import os

from django.core.asgi import get_asgi_application

from WebSearchEngine.settings import DEBUG
from engines.services.classify import NaiveBayesClassifier, TransformerClassifier
from engines.services.cluster import ContentKMeanCluster
from engines.services.search import (
    BooleanSearcher, TFIDFSearcher, TransformerSearcher, FasttextSearcher, FasttextRepresentation
)
from engines.services.search.elastic_searcher import ElasticSearcher

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'WebSearchEngine.settings')

application = get_asgi_application()

clean_path = 'clean_data.json'
data = json.load(open(clean_path, 'r'))
_kwargs = dict(
    tokens_key='cleaned_tokens',
    data=data
)
tfidf_searcher = TFIDFSearcher(**_kwargs)
boolean_searcher = BooleanSearcher(build=False, **_kwargs)
bert_searcher = TransformerSearcher(load=True, **_kwargs)
elastic_searcher = ElasticSearcher(**_kwargs)
fasttext_repr = FasttextRepresentation(
    train=False, load=True, min_count=4, **_kwargs
)
fasttext_searcher = FasttextSearcher(
    data=data, representation=fasttext_repr
)
fasttext_cluster = ContentKMeanCluster(
    data=data, load_cluster=True, representation=fasttext_repr
)

_bert_repr_kwargs = dict(
    root='models', folder='classifiers',
    load=True, tokens_key='cleaned_tokens'
)
naive_classifier = NaiveBayesClassifier(
    data=data[:1000],
    load_clf=True,
    method='bert',
    **_bert_repr_kwargs
)
bert_classifier = TransformerClassifier(
    data=data[:1000],
    load=True,
    tokens_key='cleaned_tokens'
)

# if DEBUG:
#     naive_classifier.build(save=False)
#     bert_classifier.test()
