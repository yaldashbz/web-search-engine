import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from engines.services.data_collection.utils import get_doc_words


class DataOut(list):
    def to_df(self):
        return pd.DataFrame(self)


def get_dict(big_list):
    return {word: i for i, word in enumerate(big_list)}


def get_all_urls_and_words(data, tokens_key):
    all_urls = list()
    all_words = list()
    for doc in data:
        all_urls.append(doc['url'])
        all_words += get_doc_words(doc, key=tokens_key)
    all_urls = list(set(all_urls))
    all_words = list(set(all_words))
    return get_dict(all_urls), get_dict(all_words)


def create_boolean_matrix(data, matrix_path: str, header_path: str, tokens_key: str):
    all_urls, all_words = get_all_urls_and_words(data, tokens_key)
    matrix = np.zeros(shape=(len(all_urls), len(all_words)), dtype=bool)
    header = {'rows': all_urls, 'columns': all_words}
    for doc in tqdm(data):
        url = doc['url']
        words = get_doc_words(doc, key=tokens_key)
        for word in words:
            matrix[all_urls[url]][all_words[word]] = True

    json.dump(header, open(header_path, 'w+'))
    np.savez_compressed(matrix_path, matrix=matrix)

    return matrix, header
