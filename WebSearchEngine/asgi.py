"""
ASGI config for WebSearchEngine project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.0/howto/deployment/asgi/
"""
import json
import os

from django.core.asgi import get_asgi_application

from engines.services.search import (
    BooleanSearcher, TFIDFSearcher, TransformerSearcher, FasttextSearcher
)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'WebSearchEngine.settings')

application = get_asgi_application()

clean_path = 'clean_data.json'
tokens_key = 'cleaned_tokens'
data = json.load(open(clean_path, 'r'))
tfidf_searcher = TFIDFSearcher(data=data, tokens_key=tokens_key)
boolean_searcher = BooleanSearcher(data=data, build=False, tokens_key=tokens_key)
bert_searcher = TransformerSearcher(data=data, load=True, tokens_key=tokens_key)
fasttext_searcher = FasttextSearcher(
    data=data, train=False, load=True, min_count=4, tokens_key=tokens_key
)
