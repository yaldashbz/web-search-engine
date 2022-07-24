import json

from django.core.management import BaseCommand

from engines.services.search.elastic_searcher import ElasticSearcher


class Command(BaseCommand):
    def handle(self, *args, **options):
        clean_path = 'clean_data.json'
        data = json.load(open(clean_path, 'r'))
        _kwargs = dict(
            tokens_key='cleaned_tokens',
            data=data
        )
        elastic_searcher = ElasticSearcher(**_kwargs)
        elastic_searcher.index_data('tokens')
