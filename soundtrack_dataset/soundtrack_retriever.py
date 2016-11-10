import json
import os
import time
from tools import generic_helpers as helpers
from tools import url_retriever as ur
from tools.thread_manager import ThreadManager
from tools.thread_manager import Operation
import threading
import traceback
from soundtrack_dataset import spotify_data_retriever as spot
import random
import re

__delay__ = 1


class ElementNotFoundError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def new_song(spotify_id, track_name, features_list):
    return dict(id=spotify_id, name=track_name, features=features_list)


def new_movie_soundtrack_data(movie_id, movie_name, song_list):
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
        movie_name = helpers.clean_string(movie_name)

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
    data = new_movie_soundtrack_data(movie["id"], movie["name"], track_list)
    print(str(data))
    soundtracks_file = open(os.path.join("resources", "backup", "soundtracks", movie["id"] + '.json'), 'w',
                            encoding="latin-1")
    json.dump(data, soundtracks_file)
    soundtracks_file.close()
    return data


def wrapper_get_features_from_movie_data(args):
    movie_data = args['movie_data']
    secret = args['secret']
    secret_lock = args['secret_lock']

    identifier = '[ID: ' + movie_data['id'] + ' - ' + movie_data['name'] + '] '
    warn = '/!\ '

    with secret_lock:
        if secret.is_expired():
            print("Spotify auth expired: trying to refresh..")
            secret.request_authorization()
            print("Spotify authorisation complete!")

    try:
        print(identifier + "Starting features retrieval..")
        data = get_spotify_features_from_movie_data(movie_data, secret)
        print(identifier + "Retrieval complete")
        soundtracks_file = open(
            os.path.join("resources", "backup", "soundtrack-features", movie_data["id"] + '-features.json'), 'w',
            encoding="latin-1")
        json.dump(data, soundtracks_file)
        soundtracks_file.close()
    except spot.ReachedAPILimitError as e:
        if 'delay' in args:
            args['delay'] *= 2
        else:
            args['delay'] = random.randint(15, 45)  # seconds
        print(warn + identifier + str(e))
        print(warn + identifier + "Retrying in " + str(args['delay']) + " seconds..")
        time.sleep(args['delay'])
        data = wrapper_get_features_from_movie_data(args)
    except spot.ReachedAPILimitError as e:
        raise e
    except Exception as e:
        print(warn + identifier + "Unhandled exception occurred.." + str(e))
        if 'unhandled' in args:
            print(identifier + "Permanent failure..")
            raise e
        else:
            args['unhandled'] = True
            delay = random.randint(5*60, 10*60)
            min = int(delay / 60)
            sec = delay % 60
            message = ""
            message += warn + identifier + str(e.__class__) + "\n"
            message += warn + identifier + str(e) + "\n"
            message += warn + identifier + "Stack Trace - " + str(e) + "\n"
            message += "####################################" + "\n"
            message += traceback.format_exc() + "\n"
            message += "####################################" + "\n"
            print(message)
            print(warn + identifier + "Retrying in " + str(min) + " minutes and " + str(sec) + " seconds..")
            time.sleep(delay)
            data = wrapper_get_features_from_movie_data(args)

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


def get_spotify_features_from_movie_data(movie_data, secret=None):
    if not secret:
        secret = spot.SecretData()
        response = secret.request_authorization()
    movie_data_with_features = movie_data
    soundtrack_with_features = []

    for track_name in movie_data["soundtrack"]:
        track_name_clean = helpers.clean_string(track_name)
        if track_name_clean.isspace():
            raise ValueError("Empty track name after cleaning, was: " + track_name)
        query_response = spot.query_track(track_name_clean, secret)
        queried_track_list = query_response['tracks']['items']
        if len(queried_track_list) == 0:
            missing = new_song("", "", "")
            missing['missing'] = track_name_clean
            soundtrack_with_features.append(missing)
        else:
            spot_id = spot.get_first_track_id(query_response)
            spot_name = spot.get_first_track_name(query_response)
            try:
                features = spot.request_audio_features(spot_id, secret)
            except spot.AnalysisNotFoundError as e:
                features = {"missing": str(e)}
            track = new_song(spot_id, spot_name, features)
            soundtrack_with_features.append(track)

    movie_data_with_features['soundtrack'] = soundtrack_with_features
    return movie_data_with_features


def get_spotify_features_for_whole_dataset(movie_data):
    manager = ThreadManager()
    wrapper_function = wrapper_get_features_from_movie_data
    secret_lock = threading.Lock()
    secret = spot.SecretData()

    operations_list = []

    for movie in movie_data:
        args = dict(secret=secret, secret_lock=secret_lock, movie_data=movie)
        operations_list.append(Operation(manager, wrapper_function, args))

    secret.request_authorization()
    manager.run_all(operations_list, Operation(manager, (lambda x: time.sleep(x)), 10, 999999))

    while not manager.all_ops_finished():
        time.sleep(__delay__)

    completed = manager.get_completed_ops()
    failed = manager.get_failures()

    movie_data_with_features = []
    errors = []
    for op in completed:
        movie_data_with_features.append(op.get_return_value())
    for op in failed:
        error = op.get_error()
        err_dict = {
            "id": op.get_id(),
            "value": error.value,
            "class": str(error.__class__),
            "traceback": str(error.__traceback__)
        }
        errors.append(err_dict)
    return movie_data_with_features, errors


def save_soundtrack_data(soundtracks, file_path="soundtracks.json"):
    soundtracks_file = open(file_path, 'w', encoding="latin-1")
    json.dump(soundtracks, soundtracks_file)
    soundtracks_file.close()


def load_soundtrack_data(file_path):
    soundtracks_file = open(file_path, 'r', encoding="latin-1")
    soundtracks = json.load(soundtracks_file)
    soundtracks_file.close()
    return soundtracks
