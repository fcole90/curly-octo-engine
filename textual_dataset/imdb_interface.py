import requests
import time
import threading
from tools.thread_manager import ThreadManager
from tools.thread_manager import Operation
from tools import url_retriever as ur
from tools import generic_helpers as helpers


__delay__ = 1


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
    return get_imdb_data(title, year)['imdbID']


def get_imdb_data(title, year=None):
    query_title = '+'.join(helpers.clean_string(title).split())
    url = "http://www.omdbapi.com/?t=" + query_title
    if year:
        url += "&y=" + year
    url += "&type=movie&r=json"

    response = requests.get(url)

    if not response.ok:
        raise BadResponseError(response.content, url)

    response_dict = response.json()

    if not response_dict['Response'] == 'True':
        if ',' in title:
            return get_imdb_id(helpers.reform_title(title), year)
        else:
            raise BadResponseError(response.content, url)

    return response_dict


def get_title_and_year(movielens_title):
    if '(' in movielens_title:
        title_list = movielens_title.split('(')  # "Toy Story (1995)" -> ["Toy Story", "1995)"]
        title = title_list[0]
        year = helpers.clean_number(title_list[-1])  # Get the clean year
        return title, year
    else:
        return movielens_title, None


def get_imdb_keywords(imdb_id):
    soup = ur.get_soup("http://www.imdb.com/title/" + imdb_id + "/textual_dataset")
    keywords = [el.get('data-item-keyword') for el in soup.find_all('td', class_="soda sodavote")]
    return keywords


def get_imdb_from_movielens(movielens_title):
    title_and_year = get_title_and_year(movielens_title)
    return get_imdb_id(title_and_year[0], title_and_year[1])


def get_imdb_data_from_movielens(movielens_title):
    title_and_year = get_title_and_year(movielens_title)
    return get_imdb_data(title_and_year[0], title_and_year[1])


def get_keywords_from_movielens(movielens_title):
    return get_imdb_keywords(get_imdb_from_movielens(movielens_title))


def save_imdb_id(movielens_id, imdb_id, imdb_file, imdb_title):
    line = ('::'.join([movielens_id, imdb_id, imdb_title])) + '\n'
    imdb_file.write(line)


def save_keywords(movielens_id, keywords, keywords_file):
    line = '::'.join([movielens_id, '|'.join(keywords)]) + '\n'
    keywords_file.write(line)


def save_failures(movielens_id, error_dict, error_file):
    line = movielens_id + '::' + error_dict + '\n'
    error_file.write(line)


def save_movie_script(movielens_id, is_full, script, file):
    text = '::'.join(['@newmovie', str(movielens_id), str(is_full), script])
    file.write(text)

def wrapper_imdb_retrieve_data(args):
    movie_data = args['movie_data']
    identifier = '[ID: ' + movie_data['id'] + ' - ' + movie_data['name'] + '] '
    warn = '/!\ '

    print(identifier + "Starting to retrieve.. ")
    try:
        imdb_data = get_imdb_data_from_movielens(args['movie_data']['name'])
        imdb_id = imdb_data['imdbID']
        imdb_title = imdb_data['Title']
        keywords = get_imdb_keywords(imdb_id)
        with args['data_file_lock']:
            save_imdb_id(movie_data['id'], imdb_id, args['imdb_file'], imdb_title)
            save_keywords(movie_data['id'], keywords, args['keywords_file'])
    except Exception as e:
        raise e
        with args['error_file_lock']:
            save_failures(args['movie_data']['id'], str(e), args['failures_keywords_file'])

    print(identifier + "Retrivial complete..")


def get_all_keywords_and_imdb_id_from_movielens(movie_data, imdb_file, keywords_file, failures_keywords_file):
    manager = ThreadManager()
    wrapper_function = wrapper_imdb_retrieve_data
    data_file_lock = threading.Lock()
    error_file_lock = threading.Lock()

    operations_list = []

    for movie in movie_data:
        args = {'movie_data': movie,
                'imdb_file': imdb_file,
                'keywords_file': keywords_file,
                'failures_keywords_file': failures_keywords_file,
                'data_file_lock': data_file_lock,
                'error_file_lock': error_file_lock}
        operations_list.append(Operation(manager, wrapper_function, args))

    manager.run_all(operations_list)

    start_time = time.time()
    while not manager.all_ops_finished():
        if start_time - time.time() > (8*60*60):
            print("Timeout reached waiting all threads to end..")
            break
        time.sleep(__delay__)


def get_script_url(movielens_title):
    url_title = get_title_and_year(movielens_title)[0]  # Get only title
    url_title = '-'.join(url_title.split())  # Reformat the title: Toy Story -> Toy-Story
    return url_title


def get_imsdb_script_full_url(movielens_title):
    url_title = get_script_url(movielens_title)
    full_url = "http://www.imsdb.com/scripts/" + url_title + ".html"
    return full_url


def get_imsdb_script_text(movielens_title):
    url = get_imsdb_script_full_url(movielens_title)
    soup = ur.get_soup(url)

    return soup.find_all('td', class_='scrtext')[0].get_text()


def is_full_script(text):
    if len(text) < 200:
        return False
    else:
        return True


def wrapper_imsdb_retrieve_script(args):
    movie_data = args['movie_data']
    identifier = '[ID: ' + movie_data['id'] + ' - ' + movie_data['name'] + '] '
    warn = '/!\ '

    print(identifier + "Starting to retrieve.. ")
    try:
        script = get_imsdb_script_text(movie_data['name'])
        with args['data_file_lock']:
            save_movie_script(movie_data['id'], is_full_script(script), script, args['scripts_file'])
    except Exception as e:
        with args['error_file_lock']:
            save_failures(args['movie_data']['id'], str(e), args['failures_file'])

    print(identifier + "Retrivial complete..")

def get_all_scripts_from_movielens(movie_data, scripts_file, failures_file):
    manager = ThreadManager()
    wrapper_function = wrapper_imsdb_retrieve_script
    data_file_lock = threading.Lock()
    error_file_lock = threading.Lock()

    operations_list = []

    for movie in movie_data:
        args = {'movie_data': movie,
                'scripts_file': scripts_file,
                'failures_file': failures_file,
                'data_file_lock': data_file_lock,
                'error_file_lock': error_file_lock}
        operations_list.append(Operation(manager, wrapper_function, args))

    manager.run_all(operations_list)

    start_time = time.time()
    while not manager.all_ops_finished():
        if start_time - time.time() > (8*60*60):
            print("Timeout reached waiting all threads to end..")
            break
        time.sleep(__delay__)