from django.shortcuts import render
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.status import HTTP_200_OK
from rest_framework.viewsets import GenericViewSet

from engines.services.general import search, link_analysis


class SearchViewSet(GenericViewSet):

    @action(methods=['GET'], detail=False)
    def search(self, request):
        query = request.query_params.get('query')
        method = request.query_params.get('method')
        use_qe = request.query_params.get('use_qe') == 'true'
        if query:
            result = search(query, method, k=10, use_qe=use_qe)
            # return Response(data={
            #     'result': result
            # }, status=HTTP_200_OK)
            return render(request, 'show_results.html', {'result': result})
        return Response(status=HTTP_200_OK)

    @action(methods=['GET'], detail=False, url_path='link-analysis')
    def link_analysis(self, request):
        doc_indices = list(map(int, request.GET.getlist('doc')))
        if doc_indices:
            pagerank, hub, authority = link_analysis(doc_indices)
            # return Response(data={
            #     'pagerank': pagerank,
            #     'hub': hub,
            #     'authority': authority
            # })
            return render(request, 'link_analysis.html',
                          {
                              'pagerank': pagerank,
                               'hub': hub,
                               'authority': authority
                          })
        return Response(status=HTTP_200_OK)
