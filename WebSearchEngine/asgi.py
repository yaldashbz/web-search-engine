"""
ASGI config for WebSearchEngine project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.0/howto/deployment/asgi/
"""
import json
import os

from django.core.asgi import get_asgi_application

from engines.services.cluster import ContentKMeanCluster
from engines.services.search import (
    BooleanSearcher, TFIDFSearcher, TransformerSearcher, FasttextSearcher, FasttextRepresentation
)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'WebSearchEngine.settings')

application = get_asgi_application()

clean_path = 'clean_data.json'
data = json.load(open(clean_path, 'r'))
_args = dict(
    tokens_key='cleaned_tokens',
    data=data
)
tfidf_searcher = TFIDFSearcher(**_args)
boolean_searcher = BooleanSearcher(build=False, **_args)
bert_searcher = TransformerSearcher(load=True, **_args)
fasttext_repr = FasttextRepresentation(
    train=False, load=True, min_count=4, **_args
)
fasttext_searcher = FasttextSearcher(
    data=data, representation=fasttext_repr
)
fasttext_cluster = ContentKMeanCluster(
    data=data, load_cluster=True, representation=fasttext_repr
)
