from django.urls import re_path

from engines.views.cluster_view import ClusterViewSet
from engines.views.search_view import SearchViewSet

urlpatterns = [
    re_path(r'^cluster/(?P<method>fasttext)$', ClusterViewSet.as_view({
        'get': 'cluster'
    })),
    re_path(r'^cluster/score/(?P<method>fasttext)$', ClusterViewSet.as_view({
        'get': 'score'
    })),
    re_path('search/(?P<method>tf-idf|bert|fasttext|boolean)', SearchViewSet.as_view({
        'get': 'search'
    })),
]
