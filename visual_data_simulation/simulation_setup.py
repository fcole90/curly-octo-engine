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

# --- Definitions ---
__CACHE_PREFIX__ = "visual_simulation_setup"

__ratings_scale__ = 5
__PALETTE_SIZE__ = 6
__COLOR_DATA_ENTRY_SIZE = 5
__LIMIT_PERM_USER__ = 30
__LIMIT_PERM_COLOR__ = 30
# -------------------

class Setup:

    def __init__(self,
                 labels_data_type='one_hot',
                 limit_memory_usage=True,
                 movie_amount_limit=0,
                 test_mode=False,
                 use_cache=True,
                 user_data_function='weight_average',
                 allow_negative_data=True,
                 subsets_relative_sizes=None):
        """
        Parameters
        ----------
        labels_data_type: str
            default 'one_hot'
            Values: {decimal, one_hot}

        limit_memory_usage: bool
            default True
            Removes all the datasets except the final ones,
            i.e. training, test and validation sets.

        movie_amount_limit: int
            default 0
            Disabled by default, if enabled, limits the amount of movies in the
            dataset.

        test_mode: bool
            default False
            Doesn't load data automatically.

        use_cache: bool
            default True

        user_data_function: str
            default 'weight_average'
            Values: {average, weight_average, clusters}

        allow_negative_data: bool
            default True
            If enabled, the user data of poorly rated movies may become negative.

        subsets_relative_sizes: list()
            default: None
            Defines the relative sizes of the dataset subsets.
            If a list of only one element is provided, it will be used first as
            the relative size of the training to the test set, and then as
            the relative size of the test to the validation set.
            If a list of two elements is provided, the first will be used as
            the relative size of the training to the test set, and the second
            as the relative size of the test to the validation set.
            If None, the relative sizes will be set to 2 / 3.

        """

        # Check keyword arguments
        self.__LIMIT_MEMORY_USAGE__ = limit_memory_usage
        self.__TEST_MODE__ = test_mode
        self.__USE_CACHE__ = use_cache
        self.__LABELS_DATA_TYPE__ = labels_data_type
        self.__ALLOW_NEGATIVE_DATA = allow_negative_data

        if subsets_relative_sizes is None:
            self.subsets_len_ratio = [2 / 3, 2 / 3]
        elif len(subsets_relative_sizes) == 1:
            self.subsets_len_ratio = [subsets_relative_sizes[0]] * 2
        elif len(subsets_relative_sizes) > 2:
            raise ValueError("Expected list of at most 2 elements.")
        else:
            self.subsets_len_ratio = subsets_relative_sizes

        # Avoid unintended usage
        del subsets_relative_sizes

        if self.__TEST_MODE__:
            return

        self.create_conversion_data_keys_to_list_index()
        self.create_subsets_indices()
        self.load_color_data()
        self.load_user_data(user_data_function)
        self.create_input_data(user_data_function)
        self.load_labels_data()

        self.dataset_size = len(self.input_data)
        self.dataset_couples = (np.array(self.input_data), np.array(self.labels_data))

        if movie_amount_limit > 0:
            self.limit_movies(movie_amount_limit)

        self.dataset = {
            "training": (self.dataset_couples[0][self.train_indices],
                         self.dataset_couples[1][self.train_indices]),
            "test": (self.dataset_couples[0][self.test_indices],
                     self.dataset_couples[1][self.test_indices]),
            "validation": (self.dataset_couples[0][self.validation_indices],
                           self.dataset_couples[1][self.validation_indices])
        }

        if self.__LIMIT_MEMORY_USAGE__:
            del self.color_data
            del self.user_data
            del self.input_data
            del self.labels_data
            del self.convert_dict

    def cache_format_file_name(self, data_name, identifier=""):
        return "_".join([__CACHE_PREFIX__, data_name + identifier])

    def can_load_cache(self, cache_name):
        return self.__USE_CACHE__ is True and \
               not self.__TEST_MODE__ and \
               gh.cache_file_exists(cache_name)

    def can_save_cache(self):
        return self.__USE_CACHE__ is True and \
               not self.__TEST_MODE__

    def get_flag_value(self, kwargs, name, default):
        if name in kwargs:
            return kwargs[name]
        else:
            return default

    def create_conversion_data_keys_to_list_index(self):
        """
        Creates a dictionary from the user id and movie id to the convert_list index.

        Usage: convert_dict[user_id][movie_id] -> [convert_list_index]
        """
        cache_convert_dict = self.cache_format_file_name("convert_dict")
        cache_convert_list = self.cache_format_file_name("convert_list")
        cache_movie_indices = self.cache_format_file_name("movie_indices")

        if self.can_load_cache(cache_convert_dict):
            self.convert_dict = gh.load_object_from_cache(cache_convert_dict)
            self.convert_list = gh.load_object_from_cache(cache_convert_list)
            self.movie_indices = gh.load_object_from_cache(cache_movie_indices)

        else:
            ml_ratings = ml_helpers.load_ml_ratings()
            convert_list = list()
            convert_dict = dict()
            movie_indices = dict()
            convert_list_index = 0
            for user_id in ml_ratings.keys():
                for movie_id in ml_ratings[user_id].keys():
                    convert_list.append((user_id, movie_id, convert_list_index))

                    if user_id not in convert_dict.keys():
                        convert_dict[user_id] = dict()
                    convert_dict[user_id][movie_id] = convert_list_index

                    if movie_id not in movie_indices.keys():
                        movie_indices[movie_id] = list()
                    movie_indices[movie_id].append(convert_list_index)

                    convert_list_index += 1

            self.convert_list = convert_list
            self.convert_dict = convert_dict
            self.movie_indices = movie_indices

            if self.can_save_cache():
                gh.save_object_to_cache(convert_dict, cache_convert_dict)
                gh.save_object_to_cache(convert_list, cache_convert_list)
                gh.save_object_to_cache(movie_indices, cache_movie_indices)

    def create_subsets_indices(self):
        relative_len_train = self.subsets_len_ratio[0]
        relative_len_test = (1 - relative_len_train) * self.subsets_len_ratio[1]
        relative_len_validation = (1 - relative_len_test) * (1 - self.subsets_len_ratio[1])

        validation_len = lambda l: max(1, int(relative_len_validation * l))
        test_len = lambda l: max(1, int(relative_len_test * l))
        train_len = lambda l: max(1, int(relative_len_train * l))

        validation_indices = list()
        test_indices = list()
        train_indices = list()

        for user_id in self.convert_dict:
            # indices of the convert list
            movie_list_indices = list(self.convert_dict[user_id].values())
            len_movie_list_indices = len(movie_list_indices)

            vl = validation_len(len_movie_list_indices)
            ts = test_len(len_movie_list_indices)
            tr = train_len(len_movie_list_indices)

            train_indices.extend(movie_list_indices[0:tr])
            test_indices.extend(movie_list_indices[tr:(tr+ts)])
            validation_indices.extend(movie_list_indices[(tr+ts):])

        self.validation_indices = validation_indices
        self.test_indices = test_indices
        self.train_indices = train_indices

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
        if self.can_load_cache(cache_name):
            self.color_data = gh.load_object_from_cache(cache_name)
        else:
            color_dict = colors.load_as_dict_of_lists(normalize=True)
            color_dict_keys = color_dict.keys()
            self.color_data = {key: [value for color in color_dict[key]
                                     for value in color]
                               for key in color_dict_keys}

            if self.can_save_cache():
                gh.save_object_to_cache(self.color_data, cache_name)

    def create_subset_dict(self, subset_indices):
        subset_dict = dict()

        for i in subset_indices:
            item = self.convert_list[i]
            user_id = item[0]
            movie_id = item[1]

            if user_id not in subset_dict.keys():
                subset_dict[user_id] = dict()

            subset_dict[user_id][movie_id] = i

        return subset_dict

    def load_user_data_as_color_average(self, color_data=None, min_rate=3):
        """
        Loads a dict of user data as averaged colors.

        The color average is obtained averaging each column of color data.
        Each color data row, is a color data relative to a movie rated by
        the user.
        """
        if not color_data:
            color_data = self.color_data

        suffix = "_allow_negative" if self.__ALLOW_NEGATIVE_DATA else ""
        cache_name = self.cache_format_file_name("user_data_average" + suffix)
        if self.can_load_cache(cache_name):
            self.user_data = gh.load_object_from_cache(cache_name)
        else:
            ml_color_data = color_data
            ml_ratings = ml_helpers.load_ml_ratings()
            train_set_dict = self.create_subset_dict(self.train_indices)

            user_data = dict()

            for user_id in train_set_dict.keys():
                user_color_data = list()

                for movie_id in train_set_dict[user_id].keys():
                    if ml_ratings[user_id][movie_id] >= min_rate:
                        user_color_data.append(ml_color_data[movie_id])
                    elif self.__ALLOW_NEGATIVE_DATA:
                        user_color_data.append([- color for color in ml_color_data[movie_id]])

                if user_color_data:
                    user_data[user_id] = [st.mean(color_channel_list)
                     for color_channel_list in zip(*user_color_data)]
                else:
                    # Use random data when no data is available
                    user_data[user_id] = [rd.random() for i in range(3*__PALETTE_SIZE__)]

            self.user_data = user_data

            if self.can_save_cache():
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

        suffix = "_allow_negative" if self.__ALLOW_NEGATIVE_DATA else ""
        cache_name = self.cache_format_file_name("user_data_weight_avg" + suffix)
        if self.can_load_cache(cache_name):
            self.user_data = gh.load_object_from_cache(cache_name)
        else:

            if self.__ALLOW_NEGATIVE_DATA:
                weights = [-1.0, -.4142, .1892, .4142, 1.0]
            else:
                weights = [.000001, .00001, .1892, .4142, 1.0]
            ml_ratings = ml_helpers.load_ml_ratings()
            train_set_dict = self.create_subset_dict(self.train_indices)
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
                                         for movie_id in train_set_dict[user_id].keys()],
                                         [weights[ml_ratings[user_id][movie_id] - 1]
                                          for movie_id in train_set_dict[user_id].keys()])
                               for user_id in train_set_dict.keys()}


            self.user_data = {i: [sum(color_channel_list) / sum(user_color_data[i][1])
                                  for color_channel_list in zip(*user_color_data[i][0])]
                              for i in user_color_data.keys()}

            if self.can_save_cache():
                gh.save_object_to_cache(self.user_data, cache_name)

    def load_user_data_as_color_clusters(self, color_data=None, min_rate=3, clusters_number=6):
        """
        Loads a dict of user data as color cluster centroids.

        The user is a list of the centroids of the clusters of her
        favourite movies.

        The movies are considered favourite if they have been rated
        above a certain value.

        The movies are represented as colours, so the users are
        represented on the same space.

        """
        suffix = "_allow_negative" if self.__ALLOW_NEGATIVE_DATA else ""
        cache_name = self.cache_format_file_name("user_data_clusters" + suffix)
        if self.can_load_cache(cache_name):
            self.user_data = gh.load_object_from_cache(cache_name)
        else:

            if not color_data:
                color_data = self.color_data

            ml_ratings = ml_helpers.load_ml_ratings()
            train_set_dict = self.create_subset_dict(self.train_indices)

            kmeans = KMeans(n_clusters=clusters_number)

            user_data = dict()
            for user_id in train_set_dict.keys():
                user_movie_list = list()
                for movie_id in train_set_dict[user_id].keys():
                    if ml_ratings[user_id][movie_id] >= min_rate:
                        user_movie_list.extend(Setup.splitted_list(color_data[movie_id], 3))
                    elif self.__ALLOW_NEGATIVE_DATA:
                        negative_color_data = [-color for color in color_data[movie_id]]
                        user_movie_list.extend(Setup.splitted_list(negative_color_data, 3))

                if user_movie_list:
                    x_to_cluster = np.array(user_movie_list, dtype=float)
                    kmeans.fit_predict(x_to_cluster)
                    user_data[user_id] = [channel for center in kmeans.cluster_centers_
                                          for channel in center]
                else:
                    # Use random data when no data is available
                    user_data[user_id] = [rd.random() for i in range(3*__PALETTE_SIZE__)]

            self.user_data = user_data

            if self.can_save_cache():
              gh.save_object_to_cache(self.user_data, cache_name)

    def load_user_data(self, loading_fun='clusters'):
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

    def create_input_data(self, loading_fun='clusters'):
        cache_name = self.cache_format_file_name("input_data_" + loading_fun)
        if self.can_load_cache(cache_name):
            self.input_data = gh.load_object_from_cache(cache_name)
        else:
            input_data = list()
            for item in self.convert_list:
                user_id = item[0]
                movie_id = item[1]
                input_data.append(np.array(self.user_data[user_id] + self.color_data[movie_id]))
            self.input_data = input_data

            if self.can_save_cache():
                gh.save_object_to_cache(input_data, cache_name)

    def load_labels_data(self):
        """
        Loads the labels data as a one hot ratings list or decimal values.

        E.g. rating 5 becomes [0,0,0,0,1] with one hot.
        E.g. rating 5 becomes [1] with decimal.
        """
        cache_name = self.cache_format_file_name("labels_data_", self.__LABELS_DATA_TYPE__)
        if self.can_load_cache(cache_name):
            self.labels_data = gh.load_object_from_cache(cache_name)
        else:
            labels_data = list()
            one_hot_list_default = [0] * __ratings_scale__
            ml_ratings = ml_helpers.load_ml_ratings()

            for item in self.convert_list:
                user_id = item[0]
                movie_id = item[1]

                rating = ml_ratings[user_id][movie_id]
                rating_as_one_hot = one_hot_list_default[:]
                rating_as_one_hot[rating - 1] = 1
                rating_as_decimal = [rating / __ratings_scale__]
                if self.__LABELS_DATA_TYPE__ is "decimal":
                    labels_data.append(np.array(rating_as_decimal))
                else:
                    labels_data.append(np.array(rating_as_one_hot))

            self.labels_data = labels_data

            if self.can_save_cache():
                gh.save_object_to_cache(labels_data, cache_name)

    def limit_movies(self, limit):
        allowed_movies = np.random.choice(list(self.movie_indices.keys()), limit)
        allowed_movies_indices = {index for movie_id in allowed_movies
                                  for index in self.movie_indices[movie_id]}

        for i in range(len(self.train_indices) - 1, -1, -1):
            if self.train_indices[i] not in allowed_movies_indices:
                del self.train_indices[i]

        for i in range(len(self.test_indices) - 1, -1, -1):
            if self.test_indices[i] not in allowed_movies_indices:
                del self.test_indices[i]

        for i in range(len(self.validation_indices) - 1, -1, -1):
            if self.validation_indices[i] not in allowed_movies_indices:
                del self.validation_indices[i]

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

        # index_list = np.random.choice(range(len(self.dataset["training"])), n)
        index_list = np.random.choice(self.train_indices, n)
        inputs = self.dataset_couples[0][index_list]
        labels = self.dataset_couples[1][index_list]

        if use_permutations:

            half_len = len(inputs[0]) // 2

            # ----- Inputs -----
            # Each item of the original input is subdivided in the user and
            # color part.
            # Both are then reshaped to match the RGB structure (half // 3, 3)
            # and  then permutated preserving the RGB ordering.
            # The two are then restored to the previous shape and merged.
            # The merged array is reshaped as the original item and added
            # in the form of an np array.
            #
            # For each item for each permutation in the limit of permutations.
            perm_input = [np.array(
                [np.random.permutation(item[:half_len].reshape((half_len // 3, 3)))
                     .reshape((1, half_len))[0],
                 np.random.permutation(item[half_len:].reshape((half_len // 3, 3)))
                     .reshape((1, half_len))[0]])
                                  .reshape((1, half_len*2))[0]
                              for i, item in enumerate(inputs)
                              for _ in range(limit_permutations)]
            # ------------------

            # --- Labels ---
            perm_label = [labels[i]
                          for i, item in enumerate(labels)
                          for _ in range(limit_permutations)]
            # --------------

            # Previous code without list comprehension.
            #
            # for i, item in enumerate(labels):
            #     perm_label.extend([labels[i]] * limit_permutations)
            #
            # for i, item in enumerate(inputs):
            #     user_data = item[:half_len].reshape((half_len // 3, 3))
            #     color_data = item[half_len:].reshape((half_len // 3, 3))
            #
            #     # Initial preparation for list comprehension
            #     #
            #     # perm_input += [np.array([np.random.permutation(user_data).reshape((1, half_len))[0],
            #     #                         np.random.permutation(color_data).reshape((1, half_len))[0]])
            #     #                   .reshape((1, half_len*2))[0]
            #     #               for j in range(limit_permutations)]
            #
            #     for j in range(limit_permutations):
            #         perm_user = np.random.permutation(user_data).reshape((1, half_len))[0]
            #         perm_color = np.random.permutation(color_data).reshape((1, half_len))[0]
            #         perm_item = np.array([perm_user, perm_color]).reshape((1, half_len*2))[0]
            #         perm_input.append(perm_item)

            return (list(perm_input), list(perm_label))

        else:
            return (list(inputs), list(labels))



