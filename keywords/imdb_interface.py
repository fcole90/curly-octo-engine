import requests
from tools import url_retriever as ur
from tools import generic_helpers as helpers

class BadResponseError(Exception):
    def __init__(self, value, query=None):
        self.value = str(value)
        self.query = query

    def __str__(self):
        message = ("IMDB Error" +
            " - " + self.value)
        if self.query:
            message += " - Query: '" + str(self.query) + "'"
        return message

    def get_response(self):
        return self.value


    def get_query(self):
        return self.query

def get_imdb_id(title, year=None):
    query_title = '+'.join(title.split())
    url = "http://www.omdbapi.com/?t=" + query_title
    if year:
        url += "&y=" + year
    url += "&type=movie&r=json"

    response = requests.get(url)

    if not response.ok:
        raise BadResponseError(response.content, url)

    response_dict = response.json()

    if not response_dict['Response'] == 'True':
        raise BadResponseError(response.content, url)

    return response_dict['imdbID']


def get_title_and_year(movielens_title):
    if '(' in movielens_title:
        title_list = movielens_title.split('(') # "Toy Story (1995)" -> ["Toy Story", "1995)"]
        title = helpers.clean_string(title_list[0])
        year = helpers.clean_number(title_list[1]) # Get the clean year
        return (title, year)
    else:
        return (movielens_title, None)

def get_imdb_keywords(imdb_id):
    soup = ur.get_soup("http://www.imdb.com/title/" + imdb_id + "/keywords")
    keywords = [el.get('data-item-keyword') for el in soup.find_all('td',class_="soda sodavote")]
    return keywords

def get_imdb_from_movielens(movielens_title):
    title_and_year = get_title_and_year(movielens_title)
    return get_imdb_id(title_and_year[0], title_and_year[1])

def get_keywords_from_movielens(movielens_title):
    return get_imdb_keywords(get_imdb_from_movielens(movielens_title))

def save_imdb_id(movielens_id, imdb_id):
    pass

def save_keywords(movielens_id, keywords):
    pass