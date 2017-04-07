from bs4 import BeautifulSoup
from tools.downloader import Downloader
import re
import sys


def get_soup(url, headers):
    """
    Obtain a beautiful soup

    Parameters
    ----------
    url: str
    headers: str

    Returns
    -------

    BeautifulSoup

    """

    req = Downloader.fault_tolerant_connection(url, "Link Extractor", headers=headers)

    return BeautifulSoup(req.text, "lxml")


def search(query):
    """
    Search a query on google images
    Parameters
    ----------
    query: str

    Returns
    -------
    list of str:
        a list of 20 urls to some thumbs from google

    """
    query = query.split()
    query = '+'.join(query)
    baseurl = "https://www.google.com/search?tbm=isch&q="
    header = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'}
    url = baseurl + query

    soup = get_soup(url, header)

    return [a['src'] for a in soup.find_all("img", {"src": re.compile("gstatic.com")})]

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Query needed")
    else:
        images = search(sys.argv[1])
        for img_url in images:
            print(img_url)
