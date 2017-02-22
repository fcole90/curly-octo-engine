import random as rd
import statistics as st

import tools.movielens_helpers as ml_helpers
from image_dataset import color_dataset as colors

"""
This script is an helper to have all the data in the right place in the simulation.
"""


def get_color_data() -> dict:
    """
    Returns the color-dataset as a dict of flat lists of channel values.

    Each entry is in the following form:
    {id: [r1, g1, b1, r2, g2, b2, ... ,bn]}

    The keys are the movie id from the movielens dataset.

    Returns
    -------
    dict
        dataset dictionary


    """
    color_dict = colors.load_as_dict_of_lists(normalize=True)
    color_dict_keys = color_dict.keys()
    return {key: [value for color in color_dict[key] for value in color] for key in color_dict_keys}



def __user_data_as_color_average__() -> dict:
    """
    Returns a dict of user data as averaged colors.

    The color average is obtained averaging each column of color data.
    Each color data row, is a color data relative to a movie rated by
    the user.

    Returns
    -------


    """
    ml_ratings = ml_helpers.load_ml_ratings()
    ml_color_data = get_color_data()

    user_color_data = {user_id: ml_color_data[movie_id]
                       for user_id in ml_ratings.keys()
                       for movie_id in ml_ratings[user_id].keys()}

    return {i: [st.mean(color_channel_list)
                for color_channel_list in zip(user_color_data[i])]
            for i in user_color_data.keys()}


def get_user_data() -> dict:
    return __user_data_as_color_average__()


def get_train_data() -> list:
    pass # todo


def get_input_data() -> list:
    pass # todo


__color_data__ = get_color_data()
__user_data__ = None # todo


def next_batch(n=100) -> tuple:
    """
    Returns a random sample of n x and y data points.

    Parameters
    ----------
    n: int
        amount of batch

    Returns
    -------
    tuple of list
        a list of xs, a list of ys
    """
    xs = get_input_data()
    ys = get_train_data()

    index_list = rd.sample(range(len(xs)), n)

    return ([xs[i] for i in index_list],
            [ys[i] for i in index_list])

