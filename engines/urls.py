from django.urls import re_path, path

from engines.views.classifier_view import ClassifyViewSet
from engines.views.cluster_view import ClusterViewSet
from engines.views.search_view import SearchViewSet

urlpatterns = [
    re_path(r'^classify/(?P<method>naive|bert)$', ClassifyViewSet.as_view({
        'get': 'classify'
    })),
    re_path(r'^classify/score/(?P<method>naive|bert)$', ClassifyViewSet.as_view({
        'get': 'score'
    })),
    re_path(r'^cluster/(?P<method>fasttext)$', ClusterViewSet.as_view({
        'get': 'cluster'
    })),
    re_path(r'^cluster/score/(?P<method>fasttext)$', ClusterViewSet.as_view({
        'get': 'score'
    })),
    re_path('search/(?P<method>tf-idf|bert|fasttext|boolean)', SearchViewSet.as_view({
        'get': 'search'
    })),
    path('search/link-analysis', SearchViewSet.as_view({
        'get': 'link_analysis'
    }))
]
