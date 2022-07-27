from django.urls import re_path, path

from engines.views import utility_view
from engines.views.classifier_view import ClassifyViewSet
from engines.views.cluster_view import ClusterViewSet
from engines.views.search_view import SearchViewSet

urlpatterns = [
    # (?P<method>naive|bert)$
    re_path(r'^classify/', ClassifyViewSet.as_view({
        'get': 'classify'
    }), name='classify'),
    # (?P<method>naive|bert)$
    re_path(r'^classify/score/', ClassifyViewSet.as_view({
        'get': 'score'
    }), name='score_f'),
    # (?P<method>fasttext)$
    re_path(r'^cluster/', ClusterViewSet.as_view({
        'get': 'cluster'
    }), name='cluster'),
    # (?P<method>fasttext)$
    re_path(r'^cluster/score/', ClusterViewSet.as_view({
        'get': 'score'
    }), name='score_t'),
    # (?P<method>tf-idf|bert|fasttext|boolean|elastic)
    re_path('search/', SearchViewSet.as_view({
        'get': 'search'
    }), name='search'),
    path('search/link-analysis', SearchViewSet.as_view({
        'get': 'link_analysis'
    }), name='link-analysis'),
    re_path(r'searchpage/', utility_view.search_page_view, name='search_page'),
    re_path(r'clusterpage/', utility_view.cluster_page_view, name='cluster_page'),
    re_path(r'classifypage/', utility_view.calssifier_page_view, name='classify_page'),
    # re_path(r'home/', utility_view.home_view),
]
