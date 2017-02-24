import random as rd
import unittest

import image_dataset.color_dataset as cl
import tools.movielens_helpers as ml_helpers
import visual_data_simulation.simulation_setup as setup

__COLOR_DATASET_PALETTE_SIZE__ = 6
__MIN_CHANNEL_VALUE__ = 0
__MAX_CHANNEL_VALUE__ = 255
__MIN_CHANNEL_VALUE_NORMALIZED__ = 0.0
__MAX_CHANNEL_VALUE_NORMALIZED__ = 1.0

__testing_sample_size__ = 20


class MyTestCase(unittest.TestCase):

    def test_color_dataset(self):

        palette_dict = cl.load_as_dict_of_lists()

        # Check length
        for palette in palette_dict.values():
            self.assertEqual(len(palette), __COLOR_DATASET_PALETTE_SIZE__, palette)

        # Take a random list of keys
        index_list = rd.sample(list(palette_dict.keys()), __testing_sample_size__)

        # Check range
        for i in index_list:
            for color in palette_dict[i]:
                for value in color:
                    self.assertLessEqual(value, __MAX_CHANNEL_VALUE__,  msg=(value, i))
                    self.assertGreaterEqual(value, __MIN_CHANNEL_VALUE__, msg=(value, i))

        # Check range normalized
        palette_dict = cl.load_as_dict_of_lists(normalize=True)
        for i in index_list:
            for color in palette_dict[i]:
                for value in color:
                    self.assertLessEqual(value, __MAX_CHANNEL_VALUE_NORMALIZED__, msg=(value, i))
                    self.assertGreaterEqual(value, __MIN_CHANNEL_VALUE_NORMALIZED__, msg=(value, i))

    def test_load_ml_ratings(self):
        ml_ratings = ml_helpers.load_ml_ratings()
        self.assertNotEqual(ml_ratings, dict())

    def test_simulation_setup_get_color_data(self):
        original = cl.load_as_dict_of_lists(normalize=True)
        flattened = setup.__init_color_data__()

        # Take a random list of keys
        index_list = rd.sample(list(original.keys()), __testing_sample_size__)

        for i in index_list:
            flat_original = [value for color in original[i] for value in color]
            self.assertListEqual(flat_original, flattened[i])

    def test_user_data_as_color_average(self):
        user_data = setup.__user_data_as_color_average__()
        color_data = setup.__init_color_data__()

        # Take a random list of keys
        index_list = rd.sample(user_data.keys(), __testing_sample_size__)

        self.assertEqual(len(user_data[index_list[0]]),
                         len(color_data[list(color_data.keys())[0]]))

        for i in index_list:
            for value in user_data[i]:
                    self.assertLessEqual(value, __MAX_CHANNEL_VALUE_NORMALIZED__, msg=(value, i))
                    self.assertGreaterEqual(value, __MIN_CHANNEL_VALUE_NORMALIZED__, msg=(value, i))

    def test_input_data(self):
        input_data = setup.__init_input_data__()
        color_data = setup.__init_color_data__()
        user_data = setup.__init_user_data__()
        convert_dict = setup.__init_conversion_dict_data_to_train__()


        # Take a random list of keys
        index_list = rd.sample(list(user_data.keys()), __testing_sample_size__)

        for user_id in index_list:
            for movie_id in convert_dict[user_id].keys():
                input_data_entry_index = convert_dict[user_id][movie_id][0]
                input_data_entry = input_data[input_data_entry_index]
                input_data_check = user_data[user_id] + color_data[movie_id]
                self.assertListEqual(input_data_entry, input_data_check)


    def test_train_data(self):
        train_data = setup.__init_train_data__()
        convert_dict = setup.__init_conversion_dict_data_to_train__()
        ml_ratings = ml_helpers.load_ml_ratings()

        # Take a random list of keys
        index_list = rd.sample(list(ml_ratings.keys()), __testing_sample_size__)

        for user_id in index_list:
            for movie_id in ml_ratings[user_id].keys():
                rating = ml_ratings[user_id][movie_id]
                one_hot_check = [0]*5
                one_hot_check[rating - 1] = 1
                train_data_index = convert_dict[user_id][movie_id][0]
                one_hot_train_data_rating = train_data[train_data_index]
                self.assertListEqual(one_hot_check, one_hot_train_data_rating)

    def test_next_batch(self):
        setup.next_batch(100)
        setup.next_batch(100)
        setup.next_batch(100)
        setup.next_batch(100)
        setup.next_batch(100)


if __name__ == '__main__':
    unittest.main()
