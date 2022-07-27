from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.status import HTTP_200_OK
from rest_framework.viewsets import GenericViewSet

from engines.services.general import classify, f1_score, accuracy, confusion_matrix


class ClassifyViewSet(GenericViewSet):

    @action(methods=['GET'], detail=False)
    def classify(self, request):
        method = request.query_params.get('method')
        query = request.query_params.get('query')
        if query:
            label = classify(query, method)
            return Response(data={
                'label': label
            }, status=HTTP_200_OK)
        return Response(status=HTTP_200_OK)

    # for test
    @action(methods=['GET'], detail=False)
    def score(self, request):
        method = request.query_params.get('method')
        return Response(data={
            'f1_score': f1_score(method),
            'accuracy': accuracy(method),
            'confusion_matrix': confusion_matrix(method)
        }, status=HTTP_200_OK)
