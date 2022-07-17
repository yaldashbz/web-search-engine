from django.conf.urls import include
from django.urls import path

from engines.views.search_view import SearchViewSet

urlpatterns = [
    path('search/<str:method>/', SearchViewSet.as_view({
        'get': 'search'
    })),
]
