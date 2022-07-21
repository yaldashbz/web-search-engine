from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.status import HTTP_200_OK
from rest_framework.viewsets import GenericViewSet

from engines.services.general import search, link_analysis


class SearchViewSet(GenericViewSet):
    # authentication_classes = [JSONWebTokenAuthentication]
    # permission_classes = [IsAuthenticated]

    @action(methods=['GET'], detail=False)
    def search(self, request, method):
        query = request.query_params.get('query')
        if query:
            result = search(query, method, k=10)
            return Response(data={
                'result': result
            }, status=HTTP_200_OK)
        return Response(status=HTTP_200_OK)

    @action(methods=['GET'], detail=False, url_path='link-analysis')
    def link_analysis(self, request):
        doc_indices = list(map(int, request.GET.getlist('doc')))
        print(doc_indices)
        if doc_indices:
            pagerank, hub, authority = link_analysis(doc_indices)
            return Response(data={
                'pagerank': pagerank,
                'hub': hub,
                'authority': authority
            })
        return Response(status=HTTP_200_OK)
