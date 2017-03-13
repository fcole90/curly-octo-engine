from itertools import permutations
import math
import random as rd
import statistics as st

import numpy as np
from sklearn.cluster import KMeans

import tools.generic_helpers as gh
import tools.movielens_helpers as ml_helpers
from image_dataset import color_dataset as colors

"""
This script is an helper to have all the data in the right place in the simulation.
"""

__CACHE_PREFIX__ = "visual_simulation_setup"

__ratings_scale__ = 5
__PALETTE_SIZE__ = 6
__COLOR_DATA_ENTRY_SIZE = 5
__LIMIT_PERM_USER__ = 30
__LIMIT_PERM_COLOR__ = 30

class Setup:

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        labels_data_type: str
            default 'decimal'
            Values: {decimal, one_hot}

        limit_memory_usage: bool
            default True

        test_mode: bool
            default False
            Doesn't load data automatically.

        use_cache: bool
            default True

        user_data_function: str
            default 'weight_average'
            Values: {average, weight_average, clusters}

        """

        # Check keyword arguments
        self.__LIMIT_MEMORY_USAGE__ = self.get_flag_value(kwargs, "limit_memory_usage", default=True)
        self.__TEST_MODE__ = self.get_flag_value(kwargs, "test_mode", default=False)
        self.__USE_CACHE__ = self.get_flag_value(kwargs, "use_cache", default=True)
        self.__LABELS_DATA_TYPE__ = self.get_flag_value(kwargs, "labels_data_type", default="decimal")

        if self.__TEST_MODE__:
            return

        self.load_color_data()
        self.load_user_data(loading_fun=self.get_flag_value(kwargs,
                                                            "user_data_function",
                                                            default='weight_average'))
        self.create_conversion_dict_data_keys_to_list_index()
        self.create_input_data()
        self.load_labels_data()

        input_label_couples = list(zip(self.input_data, self.labels_data))
        rd.shuffle(input_label_couples)

        if self.__LIMIT_MEMORY_USAGE__:
            del self.color_data
            del self.user_data
            del self.input_data
            del self.labels_data
            del self.convert_dict

        self.dataset_size = len(input_label_couples)

        training_set_stop = (self.dataset_size // 3) * 2
        test_set_stop = training_set_stop + ((self.dataset_size - training_set_stop) // 3 * 2)

        self.dataset = {
            "training": np.array(input_label_couples[:training_set_stop]),
            "test": np.array(input_label_couples[training_set_stop:test_set_stop]),
            "validation": np.array(input_label_couples[test_set_stop:])
        }

        if self.__LIMIT_MEMORY_USAGE__:
            del input_label_couples

    def cache_format_file_name(self, data_name, identifier=""):
        return "_".join([__CACHE_PREFIX__, data_name + identifier])

    def can_use_cache(self, cache_name):
        return gh.cache_file_exists(cache_name) or\
                        self.__USE_CACHE__ is False

    def get_flag_value(self, kwargs, name, default):
        if name in kwargs:
            return kwargs[name]
        else:
            return default

    def load_color_data(self):
        """
        Loads the color-dataset as a dict of flat lists of channel values.

        Each entry is in the following form:
        {id: [r1, g1, b1, r2, g2, b2, ... ,bn]}

        The keys are the movie id from the movielens dataset.

        Returns
        -------
        dict
            dataset dictionary


        """
        cache_name = self.cache_format_file_name("color_data")
        if self.can_use_cache(cache_name):
            self.color_data = gh.load_object_from_cache(cache_name)
        else:
            color_dict = colors.load_as_dict_of_lists(normalize=True)
            color_dict_keys = color_dict.keys()
            self.color_data = {key: [value for color in color_dict[key]
                                     for value in color]
                               for key in color_dict_keys}
            gh.save_object_to_cache(self.color_data, cache_name)

    def load_user_data_as_color_average(self, color_data=None):
        """
        Loads a dict of user data as averaged colors.

        The color average is obtained averaging each column of color data.
        Each color data row, is a color data relative to a movie rated by
        the user.
        """
        if not color_data:
            color_data = self.color_data

        cache_name = self.cache_format_file_name("user_data_avg")
        if self.can_use_cache(cache_name):
            self.user_data = gh.load_object_from_cache(cache_name)
        else:
            ml_ratings = ml_helpers.load_ml_ratings()
            ml_color_data = color_data

            user_color_data = {user_id: [ml_color_data[movie_id]
                                         for movie_id in ml_ratings[user_id].keys()]
                               for user_id in ml_ratings.keys()}

            self.user_data = {i: [st.mean(color_channel_list)
                                  for color_channel_list in zip(*user_color_data[i])]
                              for i in user_color_data.keys()}
            gh.save_object_to_cache(self.user_data, cache_name)

    def load_user_data_as_color_weighted_average(self, color_data=None):
        """
        Loads a dict of user data as weighted average.

        The color average is obtained averaging each column of color data.
        Each color data row, is a color data relative to a movie rated by
        the user.

        The weights are:
         - 1.00 for 5
         - 0.4142 for 4
         - 0.1892 for 3
         - 0.00 for 1 and 2

        The weights of (3, 4, 5) are roots (4, 2, 0) of 2, minus one.
        """
        if not color_data:
            color_data = self.color_data

        cache_name = self.cache_format_file_name("user_data_weight_avg")
        if self.can_use_cache(cache_name):
            self.user_data = gh.load_object_from_cache(cache_name)
        else:

            weights = [.000001, .00001, .1892, .4142, 1.0]
            ml_ratings = ml_helpers.load_ml_ratings()
            ml_color_data = color_data

            """
            user_color_data[user_id] := couple
            couple := (list_of_movies, list_of_weights)
            list_of_movies := [movie1, movie2, ...]
            list_of_weights := [w1, w2, ...]
            movie_ = [r1, g1, b1, r2, g2, b2, ...] (weighted to the movie weight)
            w_ = (weight of the movie)
            """
            user_color_data = {user_id: ([[color * weights[ml_ratings[user_id][movie_id] - 1]
                                         for color in ml_color_data[movie_id]]
                                         for movie_id in ml_ratings[user_id].keys()],
                                         [weights[ml_ratings[user_id][movie_id] - 1]
                                          for movie_id in ml_ratings[user_id].keys()])
                               for user_id in ml_ratings.keys()}


            self.user_data = {i: [sum(color_channel_list) / sum(user_color_data[i][1])
                                  for color_channel_list in zip(*user_color_data[i][0])]
                              for i in user_color_data.keys()}
            gh.save_object_to_cache(self.user_data, cache_name)

    def load_user_data_as_color_clusters(self, color_data=None):
        """
        Loads a dict of user data as color cluster centroids.

        The user is a list of the centroids of the clusters of her
        favourite movies.

        The movies are considered favourite if they have been rated
        above a certain value.

        The movies are represented as colours, so the users are
        represented on the same space.

        """
        cache_name = self.cache_format_file_name("user_data_clusters")
        if self.can_use_cache(cache_name):
            self.user_data = gh.load_object_from_cache(cache_name)
        else:

            if not color_data:
                color_data = self.color_data

            ml_ratings = ml_helpers.load_ml_ratings()
            min_rate = 3
            clusters_number = 6

            kmeans = KMeans(n_clusters=clusters_number)

            user_data = dict()
            for user_id in ml_ratings.keys():
                user_movie_list = list()
                for movie_id in ml_ratings[user_id].keys():
                    if ml_ratings[user_id][movie_id] >= min_rate:
                        user_movie_list.extend(Setup.splitted_list(color_data[movie_id], 3))

                if user_movie_list:
                    x_to_cluster = np.array(user_movie_list, dtype=float)
                    kmeans.fit_predict(x_to_cluster)
                    user_data[user_id] = [channel for center in kmeans.cluster_centers_
                                          for channel in center]
                else:
                    # Use random data when no data is available
                    user_data[user_id] = [rd.random() for i in range(3*__PALETTE_SIZE__)]

            self.user_data = user_data
            gh.save_object_to_cache(self.user_data, cache_name)

    def load_user_data(self, loading_fun='weight_average'):
        """

        Parameters
        ----------
        loading_fun: str
            default 'weight_average'
            Values: {average, weight_average, clusters}

        """
        if loading_fun is 'weight_average':
            self.load_user_data_as_color_weighted_average()
        elif loading_fun is 'average':
                self.load_user_data_as_color_average()
        elif loading_fun is 'clusters':
            self.load_user_data_as_color_clusters()
        else:
            raise ValueError("Expected one among "
                             "{average, weight_average, clusters}," +
                             " found {:s} instead.".format(loading_fun))


    def create_conversion_dict_data_keys_to_list_index(self):
        """
        Creates a dictionary from the user id and movie id to the input data index.

        Usage: convert_dict[user_id][movie_id] -> [train_index]
        """
        cache_name = self.cache_format_file_name("convert_dict")
        if self.can_use_cache(cache_name):
            self.convert_dict = gh.load_object_from_cache(cache_name)
        else:
            ml_ratings = ml_helpers.load_ml_ratings()
            train_index = 0
            convert_dict = dict()
            for user_id in ml_ratings.keys():
                convert_dict[user_id] = dict()
                for movie_id in ml_ratings[user_id].keys():
                    convert_dict[user_id][movie_id] = [train_index]
                    train_index += 1
            self.convert_dict = convert_dict
            gh.save_object_to_cache(convert_dict, cache_name)

    def create_input_data(self):
        cache_name = self.cache_format_file_name("input_data")
        if self.can_use_cache(cache_name):
            self.input_data = gh.load_object_from_cache(cache_name)
        else:
            input_data = list()
            for user_id in self.convert_dict.keys():
                for movie_id in self.convert_dict[user_id].keys():
                    input_data.append(np.array(self.user_data[user_id] + self.color_data[movie_id]))
            self.input_data = input_data
            gh.save_object_to_cache(input_data, cache_name)

    def load_labels_data(self):
        """
        Loads the labels data as a one hot ratings list or decimal values.

        E.g. rating 5 becomes [0,0,0,0,1] with one hot.
        E.g. rating 5 becomes [1] with decimal.
        """
        cache_name = self.cache_format_file_name("labels_data_", self.__LABELS_DATA_TYPE__)
        if self.can_use_cache(cache_name):
            self.labels_data = gh.load_object_from_cache(cache_name)
        else:
            labels_data = list()
            one_hot_list_default = [0] * __ratings_scale__
            ml_ratings = ml_helpers.load_ml_ratings()
            for user_id in ml_ratings.keys():
                for movie_id in ml_ratings[user_id].keys():
                    rating = ml_ratings[user_id][movie_id]
                    rating_as_one_hot = one_hot_list_default[:]
                    rating_as_one_hot[rating - 1] = 1
                    rating_as_decimal = [rating / __ratings_scale__]
                    if self.__LABELS_DATA_TYPE__ is "decimal":
                        labels_data.append(np.array(rating_as_decimal))
                    else:
                        labels_data.append(np.array(rating_as_one_hot))

            self.labels_data = labels_data
            gh.save_object_to_cache(labels_data, cache_name)

    @staticmethod
    def splitted_list(my_list, n=3) -> list:
        return [my_list[i:i + n] for i in range(0, len(my_list), n)]

    @staticmethod
    def get_only_part(n, input_labels_couples_list):
        return [entry[n] for entry in input_labels_couples_list]

    def next_batch(self, n=100, use_permutations=False, limit_permutations=50) -> tuple:
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

        index_list = np.random.choice(range(len(self.dataset["training"])), n)
        couples = self.dataset["training"][index_list].transpose()

        # # --- Debug ---
        # print(couples.shape)
        # print(couples[0][0])
        # print(couples[1][0])

        if use_permutations:
            inputs = couples[0]
            labels = couples[1]

            perm_input = list()
            perm_label = list()

            for i, item in enumerate(inputs):
                half_item_len = len(item) // 2
                perm_label.extend([labels[i]] * limit_permutations)

                user_data = item[:half_item_len]
                color_data = item[half_item_len:]

                user_data.reshape((half_item_len // 3, 3))
                color_data.reshape((half_item_len // 3, 3))

                for j in range(limit_permutations):
                    perm_user = np.random.permutation(user_data).reshape((1, half_item_len))[0]
                    perm_color = np.random.permutation(color_data).reshape((1, half_item_len))[0]
                    perm_item = np.array([perm_user, perm_color]).reshape((1, half_item_len*2))[0]
                    perm_input.append(perm_item)
            return (perm_input, perm_label)

        else:
            return (list(couples[0]), list(couples[1]))



