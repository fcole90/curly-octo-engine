import requests
import base64


class BadResponseError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class SecretData:
    """
    Definitions for client secret data
    """

    def __init__(self):
        self.__secret_oauth__ = ""

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
        oauth_code = response.json()["access_token"]

        if response.ok:
            print("Spotify authorization succedeed")
            print("Valid for ")
            print("OAuth: " + oauth_code)
            if autoset:
                self.set_oauth(oauth_code)

        else:
            print("Spotify authorization failed")
            print("Error: " + response.content)

        return response.json()


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

    if not response.ok:
        raise BadResponseError(response.content)

    return response.json()


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
    query = '+'.join(track_name.split())
    url = "https://api.spotify.com/v1/search?q=" + query + "&type=track"
    header = {
        "Accept": "application/json",
        "Authorization": "Bearer " + secret.get_oauth()
    }

    response = requests.get(url, headers=header)

    if not response.ok:
        raise BadResponseError(response.content)

    return response.json()


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

