from nltk import WordNetLemmatizer


class Lemmatizer:
    def __init__(self):
        self.backbone = WordNetLemmatizer()

    def lemmatize(self, content):
        return [[self.backbone.lemmatize(word) for word in sent] for sent in content]


class POSTagLemmatizer(Lemmatizer):
    """data lemmatizer with pos-tag"""

    def lemmatize(self, content):
        return [[(self.backbone.lemmatize(word), pos) for word, pos in sent]
                for sent in content]
