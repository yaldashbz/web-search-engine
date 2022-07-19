from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.status import HTTP_200_OK
from rest_framework.viewsets import GenericViewSet

from engines.services.general import cluster, rss, silhouette


class ClusterViewSet(GenericViewSet):

    @action(methods=['GET'], detail=False)
    def cluster(self, request, method):
        query = request.query_params.get('query')
        if query:
            cluster_id, result = cluster(query, method, k=10)
            return Response(data={
                'cluster_id': cluster_id,
                'result': result
            }, status=HTTP_200_OK)
        return Response(status=HTTP_200_OK)

    @action(methods=['GET'], detail=False)
    def score(self, _, method):
        print(method)
        return Response(data={
            'rss': rss(method),
            'silhouette': silhouette(method)
        }, status=HTTP_200_OK)
