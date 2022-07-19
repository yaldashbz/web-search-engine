import json
import os
import re
from itertools import chain
from random import shuffle
from typing import Optional

import numpy as np

from engines.services.data_collection.utils import TOKENS_KEY
from engines.services.search.base import BaseSearcher
from engines.services.search.utils import create_boolean_matrix, DataOut
from engines.services.utils import check_mkdir

NOT = 'not'
AND = 'and'
OR = 'or'


class BooleanSearcher(BaseSearcher):
    _MATRIX_NAME = 'boolean_matrix.npz'
    _HEADER_NAME = 'boolean_header.json'

    def __init__(
            self, data,
            build: bool = True,
            tokens_key: str = TOKENS_KEY,
            root: str = 'matrices'
    ):
        super().__init__(data, tokens_key)

        matrix_path = os.path.join(root, self._MATRIX_NAME)
        header_path = os.path.join(root, self._HEADER_NAME)

        if not (build or os.path.exists(matrix_path) or os.path.exists(header_path)):
            raise ValueError

        check_mkdir(path=root)
        self.matrix, self.header = self._get_matrix(build, matrix_path, header_path)

    def _get_matrix(self, build, matrix_path, header_path):
        return create_boolean_matrix(
            self.data, matrix_path, header_path, self.tokens_key
        ) if build else (
            np.load(matrix_path)['matrix'],
            json.load(open(header_path, 'r'))
        )

    @property
    def _all_words(self):
        """words as columns"""
        return self.header['columns']

    @property
    def _all_urls(self):
        """urls as rows"""
        return self.header['rows']

    def _get_column(self, word):
        try:
            token = ''.join(chain(*self.pre_processor.process(word[1])))
            index = self._all_words[token]
            matrix = self.matrix[:, index]
            if word[0] == NOT:
                matrix = ~matrix
            return matrix
        except KeyError:
            return np.zeros(len(self._all_urls), dtype=bool)

    @classmethod
    def _operate(cls, op1, op2, operator):
        if operator == AND:
            return op1 & op2

        if operator == OR:
            return op1 | op2

    @classmethod
    def _handle_not(cls, tokens):
        new_tokens = list()

        i = 0
        n = len(tokens)
        while i < n:
            token = tokens[i]
            if i + 1 < n and token in [AND, OR]:
                new_tokens.append(token)
            elif i + 1 < n and token == NOT:
                new_tokens.append((NOT, tokens[i + 1]))
                i += 1
            else:
                new_tokens.append(('', token))
            i += 1
        return new_tokens

    def process_query(self, query):
        query = re.sub('\\W+', ' ', query).strip().lower()
        words, operators = list(), list()
        tokens = self._handle_not(query.split())

        is_word = False
        for i, token in enumerate(tokens):
            if isinstance(token, str):
                operators.append(token)
                is_word = False
            else:
                if is_word:
                    operators.append(AND)
                words.append(token)
                is_word = True

        return words, operators

    def search(self, query, k: int = 10) -> Optional[DataOut]:
        words, operators = self.process_query(query)
        assert len(words) == len(operators) + 1
        n = len(words)
        if n == 0:
            return None
        if n < 2:
            return DataOut(self._get_results(self._get_column(words[0]), k))

        op1, op2 = self._get_column(words[0]), self._get_column(words[1])
        operator = operators[0]
        result = self._operate(op1, op2, operator)

        for i, token in enumerate(words[2:]):
            op2 = self._get_column(token)
            result = self._operate(result, op2, operators[i + 1])

        return DataOut(self._get_results(result, k))

    def _get_results_urls(self, indexes):
        result_urls = list()
        for url, i in self._all_urls.items():
            if i in indexes:
                result_urls.append(url)
        return result_urls

    def _get_results(self, column, k):
        indexes = column.nonzero()[0]
        urls = self._get_results_urls(indexes)
        shuffle(urls)
        urls = urls[:k] if k < len(urls) else urls
        results = [dict(url=url) for url in urls]
        return results
