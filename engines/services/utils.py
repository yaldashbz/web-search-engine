from typing import List

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import silhouette_samples


class WebDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


def plot_silhouette(df, n, labels, score):
    fig, ax = plt.subplots(1)
    fig.set_size_inches(8, 6)
    ax.set_xlim([-0.2, 1])
    ax.set_ylim([0, len(df) + (n + 1) * 10])

    ax.axvline(x=score, color="red", linestyle="--")
    ax.set_yticks([])
    ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.title(("Silhouette analysis for K = %d" % n), fontsize=10, fontweight='bold')

    y_lower = 10
    sample_silhouette_values = silhouette_samples(df, labels)
    for i in range(n):
        values = sample_silhouette_values[labels == i]
        values.sort()
        size = values.shape[0]
        y_upper = y_lower + size

        color = cm.nipy_spectral(float(i) / n)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, values, facecolor=color,
                         edgecolor=color, alpha=0.7)

        ax.text(-0.05, y_lower + 0.5 * size, str(i))
        y_lower = y_upper + 10
    plt.show()


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_docs_with_urls(data, urls):
    return [doc for doc in data if doc['url'] in urls]


def count_same_words(first: List[str], second: List[str]):
    return len(set(first).intersection(set(second)))
