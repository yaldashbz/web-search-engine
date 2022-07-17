from data_collection import WikiScraper, GoogleScraper
from data_collection.data import Document

_scrapers = {
    'wiki': WikiScraper,
    'google': GoogleScraper
}


def run(scraper, data_path, **kwargs):
    data = scraper().scrape(**kwargs)
    Document.save(data, data_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Retrieve data from wikipedia.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', '-p', type=str, default='data.json',
                        help='Path to the file containing the quantities.')
    parser.add_argument('--depth', '-d', type=int, default=1,
                        help='Max depth for wiki scraper')
    parser.add_argument('--scraper', '-s', type=str, default='google',
                        help='Chosen scraper')

    args = parser.parse_args()
    run(
        scraper=_scrapers[args.scraper],
        data_path=args.path,
        max_depth=args.depth
    )
