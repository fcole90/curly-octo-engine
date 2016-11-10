import requests
import base64
import time
import json

class BadResponseError(Exception):
    def __init__(self, value, query=None):
        self.value = value
        self.query = query

    def __str__(self):
        message = ("Spotify Error " + str(self.value['error']['status']) +
            " - " + self.value['error']['message'])
        if self.query:
            message += " - Query: '" + str(self.query) + "'"
        return message

    def get_response(self):
        return self.value
    # b'{\n  "error" : {\n    "status" : 404,\n    "message" : "analysis not found"\n  }\n}'
    # b'{\n  "error" : {\n    "status" : 400,\n    "message" : "No search query"\n  }\n}'
    # b'{\n  "error": {\n    "status": 429,\n    "message": "API rate limit exceeded"\n  }\n}'

    def get_query(self):
        return self.query


class ReachedAPILimitError(BadResponseError):
    def __init__(self, value):
        self.value = value
        self.query = None


class AnalysisNotFoundError(BadResponseError):
    def __init__(self, value, query):
        self.value = value
        self.query = query


class NoSearchQueryError(BadResponseError):
    def __init__(self, value, query):
        self.value = value
        self.query = query


class SecretData:
    """
    Definitions for client secret data
    """

    def __init__(self):
        self.__secret_oauth__ = ""
        self.__expires_in__ = 0
        self.__auth_time__ = 0
        self.__safe_time_limit__ = 60 #seconds

    def is_expired(self):
        if time.time() - (self.__auth_time__ + self.__safe_time_limit__) > self.__expires_in__:
            return True
        else:
            False


    def get_oauth(self):
        return self.__secret_oauth__

    def set_oauth(self, oauth_string):
        self.__secret_oauth__ = oauth_string

    def request_authorization(self, autoset=True):
        """
        Requests an authorization and optionally
        sets a value to OAuth on success.

        Parameters
        ----------
        autoset: bool
            optional, if true sets the OAuth value on success

        Returns
        -------
        dict
            json response

        """
        # set up data
        encoded_string = base64.b64encode(b"8ba9f109a5804da896a4881feaf09776:b083e54626e24930a4020ad1bc05d9f0")
        url = "https://accounts.spotify.com/api/token"
        body = {
            "grant_type": "client_credentials"
        }
        header = {
            "Authorization": b"Basic " + encoded_string
        }

        # make the actual request
        response = requests.post(url, body, headers=header)
        response_dict = response.json()

        if response.ok:
            print("Spotify authorization succedeed")
            print("Valid for " + str(response_dict["expires_in"]))
            print("OAuth: " + response_dict["access_token"])
            if autoset:
                self.set_oauth(response_dict["access_token"])
                self.__auth_time__ = time.time()
                self.__expires_in__ = int(response_dict["expires_in"])

        else:
            print("Spotify authorization failed")
            print("Error: " + response.content)

        return response_dict


def request_audio_features(track_id, secret):
    """
    Get the audio features of a track.

    Parameters
    ----------
    track_id: str
    secret: SecretData

    Returns
    -------
    dict
        json

    """
    url = "https://api.spotify.com/v1/audio-features/" + track_id
    header = {
        "Accept": "application/json",
        "Authorization": "Bearer " + secret.get_oauth()
    }

    response = requests.get(url, headers=header)
    response_dict = response.json()

    if not response.ok:
        if response_dict['error']['status'] == 404:
            raise AnalysisNotFoundError(response_dict, url)
        elif response_dict['error']['status'] == 400:
            raise NoSearchQueryError(response_dict, url)
        elif response_dict['error']['status'] == 429:
            raise ReachedAPILimitError(response_dict)
        else:
            raise BadResponseError(response_dict, url)

    return response_dict


def query_track(track_name, secret):
    """
    Query a track by its name.

    Parameters
    ----------
    track_name: str
    secret: SecretData

    Returns
    -------
    dict
        json

    """
    if track_name.isspace():
        raise ValueError("Empty track name")
    query = '+'.join(track_name.split())
    url = "https://api.spotify.com/v1/search?q=" + query + "&type=track"
    header = {
        "Accept": "application/json",
        "Authorization": "Bearer " + secret.get_oauth()
    }

    if query is "":
        raise ValueError("Empty query after split of " + track_name)

    response = requests.get(url, headers=header)
    response_dict = response.json()

    if not response.ok:
        if response_dict['error']['status'] == 404:
            raise AnalysisNotFoundError(response_dict, url)
        elif response_dict['error']['status'] == 400:
            raise NoSearchQueryError(response_dict, url)
        elif response_dict['error']['status'] == 429:
            raise ReachedAPILimitError(response_dict)
        else:
            raise BadResponseError(response_dict, url)

    return response_dict


def get_first_track_id(json_response):
    """
    Get the id of the fist track found.

    Parameters
    ----------
    json_response: dict
        a json response

    Returns
    -------
    str
        the id of the first track

    """
    return json_response['tracks']['items'][0]['id']

def get_first_track_name(json_response):
    return json_response['tracks']['items'][0]['name']

