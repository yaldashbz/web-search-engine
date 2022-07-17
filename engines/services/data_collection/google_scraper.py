import re
from typing import List, Optional
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
from tqdm import tqdm

from engines.services.data_collection.base_scraper import BaseWebScraper
from engines.services.data_collection.data import Document
from engines.services.data_collection.utils import CATEGORIES


class SearchResult:
    def __init__(self, url, title, content):
        self.url = url
        self.title = title
        self.content = content

    def __hash__(self):
        return hash(self.url)

    def __str__(self):
        return f'{self.title} - "{self.content[:50]}"'

    def __repr__(self):
        return self.__str__()

    def get_dict(self):
        return self.__dict__


class GoogleScraper(BaseWebScraper):
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:79.0) Gecko/20100101 Firefox/79.0',
            'Referer': 'https://www.google.com/'
        }
        self.google_headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:79.0) Gecko/20100101 Firefox/79.0',
            'Host': 'www.google.com',
            'Referer': 'https://www.google.com/'
        }

    def scrape(self, **kwargs) -> List[Document]:
        result = list()
        for category in CATEGORIES:
            result += self.search(category)
        return result

    def _get_source(self, url: str, is_google=False) -> requests.Response:
        headers = self.google_headers if is_google else self.headers
        return requests.get(url, headers=headers, timeout=10, allow_redirects=False)

    def search(self, query: str) -> List[Document]:
        result = set()
        for i in tqdm(range(0, 100, 10)):
            response = self._get_source(f'https://www.google.com/search?q={quote(query)}&start={i}')
            if response.status_code != 200:
                continue

            soup = BeautifulSoup(response.text, 'html.parser')
            result_containers = soup.find_all('div', class_='g')
            for container in result_containers:
                try:
                    # title = container.find('h3').text
                    url = container.find('a').attrs['href']
                    content = self.get_content(url)
                    if content is None:
                        continue
                    result.add(Document(**{
                        'url': url,
                        'content': content,
                        'category': query
                    }))
                except Exception as e:
                    print(e)
        return list(result)

    def get_content(self, url: str) -> Optional[str]:
        try:
            response = self._get_source(url)
        except Exception as e:
            print(e)
            return
        if response.status_code != 200:
            raise KeyError
        return self.text_from_html(response.text)

    @classmethod
    def tag_visible(cls, element):
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            return False
        if isinstance(element, Comment):
            return False
        return True

    def text_from_html(self, body):
        soup = BeautifulSoup(body, 'html.parser')
        texts = soup.findAll(text=True)
        visible_texts = filter(self.tag_visible, texts)
        return re.sub(' +', ' ', u" ".join(t.strip() for t in visible_texts)).strip()
