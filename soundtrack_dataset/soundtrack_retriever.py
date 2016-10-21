import json
import os
import time
from tools import url_retriever as ur
from tools.thread_manager import ThreadManager
from tools.thread_manager import Operation
from soundtrack_dataset import spotify_data_retriever as spot
import re

__delay__ = 1


class ElementNotFoundError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

def new_song(spotify_id, track_name, features_list):
    return dict(id=spotify_id, name=track_name, features=features_list)


def new_soundtrack_data(movie_id, movie_name, song_list):
    return dict(id=movie_id, name=movie_name, soundtrack=song_list)


def search_album_link(movie_name, remove_symbols=True):
    """
    Obtain the album link from the movie name.

    Parameters
    ----------
    movie_name: str
        the name of the movie

    Returns
    -------
    str
        link to the album page

    """
    if remove_symbols:
        movie_name = movie_name.split("(")[0]
        replacements = {
            "&": " ",
            "#": " ",
            ":": " ",
            "-": " ",
            "'": " ",
            ".": " ",
            ",": " "

        }
        movie_name = "".join([replacements.get(char, char) for char in movie_name])

        movie_name = movie_name.replace("...", " ")

    base_url = "http://www.soundtrack.net"
    query_url = base_url + "/search/index.php?q="
    soup = ur.search(movie_name, query_url)

    albums = [a['href'] for a in soup.find_all("a", {"href": re.compile("album/")})]
    if len(albums) == 0:
        raise ElementNotFoundError(movie_name)
    if albums[0].startswith("http"):
        return albums[0]
    return base_url + str(albums[0])


def get_track_data(movie_name):
    """
    Retrieves the soundtrack features for a given movie.

    Parameters
    ----------
    movie_name: str

    Returns
    -------
    list of dict
        list of the features of each track

    """
    track_list = search_soundtrack(movie_name)
    features_list = []

    for track in track_list:
        query_response = spot.query_track(track)
        track_id = spot.get_first_track_id(query_response)
        track_features = spot.request_audio_features(track_id)

        features_list.append(track_features)

    return features_list


def get_tracks_from_url(album_url):
    """
    Get a list of tracks from the album url.

    Parameters
    ----------
    album_url: str
        link to the the album page

    Returns
    -------
    list of str
        the list of the tracks of the soundtrack of the movie
    """

    soup = ur.get_soup(album_url)
    return [td.contents[0] for td in soup.find_all("td", {"class": "trackname"})]


def search_soundtrack(movie_name):
    """
    Get a list of tracks from the movie name.

    Parameters
    ----------
    movie_name: str
        the name of the movie

    Returns
    -------
    list of str
        the list of the tracks of the soundtrack of the movie
    """

    return get_tracks_from_url(search_album_link(movie_name))


def wrapper_get_soundtrack_from_movie(movie):
    print("Starting soundtrack retrieval for " + movie["name"] + "...")
    track_list = search_soundtrack(movie["name"])
    print("Retrieval complete for " + movie["name"])
    data = new_soundtrack_data(movie["id"], movie["name"], track_list)
    print(str(data))
    soundtracks_file = open(os.path.join("resources", "backup", "soundtracks", movie["id"]+'.json'), 'w', encoding="latin-1")
    soundtracks_json = json.dumps(data)
    soundtracks_file.write(soundtracks_json)
    soundtracks_file.close()
    return data


def get_soundtracks_from_movielens(movielens):
    manager = ThreadManager()
    wrapper_function = wrapper_get_soundtrack_from_movie

    operations_list = []

    for movie in movielens:
        operations_list.append(Operation(manager, wrapper_function, movie, movie['id']))

    manager.run_all(operations_list)

    while not manager.all_ops_finished():
        time.sleep(__delay__)

    completed = manager.get_completed_ops()
    failed = manager.get_failures()

    soundtracks = []
    errors = []
    for op in completed:
        soundtracks.append(op.get_return_value())
    for op in failed:
        error = op.get_error()
        err_dict = {
            "id": op.get_id(),
            "value": error.value,
            "class": str(error.__class__),
            "traceback": str(error.__traceback__)
        }
        errors.append(err_dict)
    return soundtracks, errors


def save_soundtrack_data(soundtracks, file_path="soundtracks.json"):
    soundtracks_file = open(file_path, 'w', encoding="latin-1")
    json.dump(soundtracks, soundtracks_file)
    soundtracks_file.close()


def load_soundtrack_data(file_path):
    soundtracks_file = open(file_path, 'r', encoding="latin-1")
    soundtracks = json.load(soundtracks_file)
    soundtracks_file.close()
    return soundtracks
