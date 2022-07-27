from django.shortcuts import render
from django.template.loader import get_template


def search_page_view(request):
    return render(request, 'query_page.html')


def cluster_page_view(request):
    return render(request, 'cluster_page.html')


def calssifier_page_view(request):
    return render(request, 'classifier_page.html')
