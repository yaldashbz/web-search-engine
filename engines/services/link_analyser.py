from itertools import chain

import networkx as nx
import numpy as np

from methods.graph_builder import GraphBuilder, TFIDFGraphBuilder

_builder = {
    'word': GraphBuilder,
    'tf-idf': TFIDFGraphBuilder
}


class ContentLinkAnalyser:
    def __init__(
            self,
            dataset,
            cleaned_dataset,
            method: str = 'word',
            weighted: bool = True,
            sent_num: int = 3,
            min_similar: float = 5
    ):
        self.dataset = dataset
        self.cleaned_dataset = cleaned_dataset
        self.sent_num = sent_num
        self.builder = _builder[method](
            dataset=list(chain(*cleaned_dataset)),
            sent_num=sent_num,
            min_similar=min_similar
        )
        self.graph = self.builder.build_weighted() if weighted else self.builder.build()
        self.pagerank = nx.pagerank(self.graph)
        self.hubs, self.authorities = nx.hits(self.graph)

    def _get_most_relevant(self, rank: int):
        return ' '.join(list(chain(*list(chain(
            *self.dataset))[rank * self.sent_num: rank * self.sent_num + self.sent_num])))

    def _get_most_cleaned_relevant(self, rank: int):
        return ' '.join(self.builder.paragraphs[rank])

    def apply_pagerank(self, clean_output: bool = True):
        rank = np.argmax(list(self.pagerank.values()))
        if clean_output:
            return self._get_most_cleaned_relevant(rank)
        return self._get_most_relevant(rank)

    def apply_hits(self, clean_output: bool = True):
        h_rank = np.argmax(list(self.hubs.values()))
        a_rank = np.argmax(list(self.authorities.values()))
        if clean_output:
            return (
                self._get_most_cleaned_relevant(h_rank),
                self._get_most_cleaned_relevant(a_rank),
            )
        return (
            self._get_most_relevant(h_rank),
            self._get_most_relevant(a_rank),
        )
