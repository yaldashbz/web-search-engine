import json
import os

from engines.services.data_collection.utils import get_doc_words
from engines.services.representation import FasttextRepresentation
from engines.services.utils import cosine_sim, check_mkdir


class FasttextQueryExpansion:
    _FILE = 'all_words_emb_fasttext.json'

    def __init__(
            self, data, load: bool, representation: FasttextRepresentation,
            root: str = 'models',
            folder: str = 'QE_fasttext'
    ):
        check_mkdir(root)
        self.path = os.path.join(root, folder)
        check_mkdir(self.path)
        self.model = representation.fasttext.wv
        self.word2vec = self._get_word2vec(load, data)

    def _build_word2vec(self, data, save: bool = True):
        words = list()
        for doc in data:
            words += get_doc_words(doc, 'cleaned_tokens')
        word2vec = {word: list(self.model[word].astype('double')) for word in set(words)}
        if save:
            json.dump(word2vec, open(self.words_emb_path, 'w'))
        return word2vec

    @property
    def words_emb_path(self):
        return os.path.join(self.path, self._FILE)

    def _get_word2vec(self, load: bool, data):
        return self._build_word2vec(data) if not load \
            else json.load(open(self.words_emb_path, 'r'))

    def expand_query(self, query_words, cosine_threshold=0.9):
        most_similar_query = ''
        for q_word in query_words:
            max_cosine_similarity = 0
            most_similar_word = ''
            for k, v in self.word2vec.items():
                cs = cosine_sim(v, self.model[q_word])
                if max_cosine_similarity < cs < cosine_threshold and k != q_word:
                    max_cosine_similarity = cs
                    most_similar_word = k
            most_similar_query = most_similar_query + ' ' + most_similar_word
        print(most_similar_query, '***')
        return most_similar_query.strip()
