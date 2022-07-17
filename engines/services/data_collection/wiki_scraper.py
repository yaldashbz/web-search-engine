from typing import List, Tuple, Set, Optional

import wikipedia
from tqdm import tqdm
from wikipedia import WikipediaException

from engines.services.data_collection.base_scraper import BaseWebScraper
from engines.services.data_collection.data import Document
from engines.services.data_collection.utils import CATEGORIES

wikipedia.set_lang('en')
wikipedia.set_rate_limiting(True)


class WikiScraper(BaseWebScraper):
    def __init__(self):
        self.categories = CATEGORIES

    def scrape(
            self,
            url: str = None,
            max_depth: int = 3,
            use_linked: bool = False
    ) -> List[Document]:

        return self._linked_scrape(max_depth) if use_linked else self._scrape(max_depth)

    def _scrape(self, max_depth) -> List[Document]:
        titles = self._get_titles(list(), self.categories, max_depth)
        return self._get_data(titles)

    def _linked_scrape(self, max_depth) -> List[Document]:
        titles = self._get_titles(list(), self.categories, max_depth)
        linked_titles, base_data = self._get_linked_titles(titles)
        linked_data = self._get_data(linked_titles)
        return list(set(base_data).union(linked_data))

    @classmethod
    def _get_category(cls, title: str) -> Optional[str]:
        for category in CATEGORIES:
            if title.lower() in category.lower() or category.lower() in title.lower():
                return category

    @classmethod
    def _get_titles(cls, titles, subjects, max_depth) -> Set[str]:
        if max_depth == 0:
            return set(titles)

        for subject in tqdm(subjects):
            titles += wikipedia.search(subject)

        new_subjects = titles.copy()
        return cls._get_titles(titles, new_subjects, max_depth - 1)

    @classmethod
    def _get_linked_titles(cls, titles) -> Tuple[Set[str], List[Document]]:
        links = list()
        base_data = list()
        for title in titles:
            try:
                page = wikipedia.page(title)
                links += page.links
                base_data.append(
                    Document(url=page.url, content=page.content, category=cls._get_category(title)))
            except WikipediaException:
                continue
        return set(links), base_data

    @classmethod
    def _get_data(cls, titles) -> List[Document]:
        data = list()
        for title in tqdm(titles):
            try:
                page = wikipedia.page(title)
                data.append(
                    Document(url=page.url, content=page.content, category=cls._get_category(title)))
            except WikipediaException:
                continue
        return data
