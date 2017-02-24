from itertools import permutations
import math
import random as rd
import statistics as st

import tools.movielens_helpers as ml_helpers
from image_dataset import color_dataset as colors

"""
This script is an helper to have all the data in the right place in the simulation.
"""
__USE_ALL_DATA_PERMUTATIONS__ = False
__ratings_scale__ = 5
__PALETTE_SIZE__ = 6
__COLOR_DATA_ENTRY_SIZE = 5


def splitted_list(my_list, n) -> list:
    return [my_list[i:i+n] for i in range(0, len(my_list), n)]


def __init_color_data__() -> dict:
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


__color_data__ = __init_color_data__()


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
    ml_color_data = __init_color_data__()

    user_color_data = {user_id: ml_color_data[movie_id]
                       for user_id in ml_ratings.keys()
                       for movie_id in ml_ratings[user_id].keys()}

    return {i: [st.mean(color_channel_list)
                for color_channel_list in zip(user_color_data[i])]
            for i in user_color_data.keys()}

def __user_data_as_color_weighted_average__() -> dict:
    """
    Returns a dict of user data as weighted average.

    The color average is obtained averaging each column of color data.
    Each color data row, is a color data relative to a movie rated by
    the user.

    The weights are:
     - 1.00 for 5
     - 0.4142 for 4
     - 0.1892 for 3
     - 0.00 for 1 and 2

    Returns
    -------


    """
    weights = [.0, .0, .1892, .4142, 1.0]
    ml_ratings = ml_helpers.load_ml_ratings()
    ml_color_data = __init_color_data__()

    user_color_data = {user_id: [color * weights[ml_ratings[user_id][movie_id]-1]
                                 for color in ml_color_data[movie_id]]
                       for user_id in ml_ratings.keys()
                       for movie_id in ml_ratings[user_id].keys()}

    return {i: [st.mean(color_channel_list)
                for color_channel_list in zip(user_color_data[i])]
            for i in user_color_data.keys()}


def __init_user_data__() -> dict:
    return __user_data_as_color_weighted_average__()


__user_data__ = __init_user_data__()


def __init_conversion_dict_data_to_train__() -> dict:
    """
    Gets a dictionary from the user id and movie id to the train index.

    Usage: convert_dict[user_id][movie_id] -> [train_index]

    Returns
    -------
    dict
        conversion dictionary

    """
    ml_ratings = ml_helpers.load_ml_ratings()
    train_index = 0
    convert_dict = dict()
    for user_id in ml_ratings.keys():
        convert_dict[user_id] = dict()
        for movie_id in ml_ratings[user_id].keys():
            convert_dict[user_id][movie_id] = [train_index]
            train_index += 1
    return convert_dict


__convert_dict__ = __init_conversion_dict_data_to_train__()


def __init_input_data__(using_all_permutations=False) -> list:
    input_data = list()
    list_index = 0
    for user_id in __convert_dict__.keys():
        for movie_id in __convert_dict__[user_id].keys():
            if not using_all_permutations:
                input_data.append(__user_data__[user_id] + __color_data__[movie_id])
            else:
                # Each convert dict entry is now a list
                __convert_dict__[user_id][movie_id] = list()

                user_entry = __user_data__[user_id]
                color_data_entry = __color_data__[movie_id]

                # Split the data in color channels
                user_color_list = splitted_list(user_entry, 3)
                color_data_color_list = splitted_list(color_data_entry, 3)

                # Make every permutation
                user_color_list_perm = permutations(user_color_list)
                color_data_color_list_perm = permutations(color_data_color_list)

                # Flatten
                user_color_list_perm = [[color_channel for color in perm
                                        for color_channel in color]
                                        for perm in user_color_list_perm]
                color_data_color_list_perm = [[color_channel for color in perm
                                        for color_channel in color]
                                        for perm in color_data_color_list_perm]

                for user_perm in user_color_list_perm[:10]:
                    for color_perm in color_data_color_list_perm[:10]:
                        input_data.append(user_perm + color_perm)
                        __convert_dict__[user_id][movie_id].append(list_index)
                        list_index += 1
    return input_data


__input_data__ = __init_input_data__(__USE_ALL_DATA_PERMUTATIONS__)


def __init_train_data__(using_all_permutations=False) -> list:
    """
    Return the input data as a one hot ratings list.

    E.g. rating 5 becomes [0,0,0,0,1]

    Returns
    -------
    list
        input data as one hot ratings list
    """
    train_data = list()
    one_hot_list_default = [0]*__ratings_scale__
    ml_ratings = ml_helpers.load_ml_ratings()
    for user_id in ml_ratings.keys():
        for movie_id in ml_ratings[user_id].keys():
            rating = ml_ratings[user_id][movie_id]
            rating_in_one_hot = one_hot_list_default[:]
            rating_in_one_hot[rating - 1] = 1
            if not using_all_permutations:
                train_data.append(rating_in_one_hot)
            else:
                # permutations_size = math.factorial(__PALETTE_SIZE__) \
                #                     * math.factorial(__COLOR_DATA_ENTRY_SIZE)
                permutations_size = 100
                for _ in range(permutations_size):
                    train_data.append(rating_in_one_hot)
    return train_data


__train_data__ = __init_train_data__(__USE_ALL_DATA_PERMUTATIONS__)


def __input_data_train__():
    input_data_size = len(__input_data__)
    input_data_train = __input_data__[:input_data_size // 3 * 2]
    return input_data_train


def __input_data_test__():
    input_data_size = len(__input_data__)
    input_data_test = __input_data__[input_data_size // 3 * 2:]
    return input_data_test


def __train_data_train__():
    train_data_size = len(__train_data__)
    train_data_train = __train_data__[:train_data_size // 3 * 2]
    return train_data_train


def __train_data_test__():
    train_data_size = len(__train_data__)
    train_data_test = __train_data__[train_data_size // 3 * 2:]
    return train_data_test


x_y_couples = list(zip(__input_data__, __train_data__))
rd.shuffle(x_y_couples)


def __x_y_couples_train__():
    x_y_couples_size = len(x_y_couples)
    x_y_couples_train = x_y_couples[:x_y_couples_size // 3 * 2]
    return x_y_couples_train


def __x_y_couples_test__():
    x_y_couples_size = len(x_y_couples)
    x_y_couples_test = x_y_couples[x_y_couples_size // 3 * 2:]
    return x_y_couples_test


def get_only_part(n, x_y_couples_list):
    return [entry[n] for entry in x_y_couples_list]


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

    xs = __input_data_train__()
    ys = __train_data_train__()
    index_list = rd.sample(range(len(xs)), n)

    return ([xs[i] for i in index_list],
            [ys[i] for i in index_list])

def next_batch_x_y_couples(n=100) -> tuple:
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

    index_list = rd.sample(range(len(x_y_couples)), n)

    return ([x_y_couples[i][0] for i in index_list],
            [x_y_couples[i][1] for i in index_list])

