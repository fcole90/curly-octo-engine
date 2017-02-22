from bs4 import BeautifulSoup
from tools.downloader import Downloader


def get_soup(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'}):
    """
    Obtain a beautiful soup object

    Parameters
    ----------
    url: str
    headers: str

    Returns
    -------

    BeautifulSoup

    """

    req = Downloader.fault_tolerant_connection(url, url, headers=headers)

    return BeautifulSoup(req.text, "lxml")


def search(query, base_url):
    """
    Search a query on base_url

    Parameters
    ----------
    query: str

    Returns
    -------
    BeautifulSoup

    """
    query = query.split()
    query = '+'.join(query)
    header = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'}
    url = base_url + query

    return get_soup(url, header)

