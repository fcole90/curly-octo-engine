import random as rd
import unittest

import image_dataset.color_dataset as cl
import visual_data_simulation.simulation_setup as setup
import tools.movielens_helpers as ml_helpers

__COLOR_DATASET_PALETTE_SIZE__ = 6
__MIN_CHANNEL_VALUE__ = 0
__MAX_CHANNEL_VALUE__ = 255
__MIN_CHANNEL_VALUE_NORMALIZED__ = 0.0
__MAX_CHANNEL_VALUE_NORMALIZED__ = 1.0

__testing_sample_size__ = 100

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
                    self.assertLessEqual(value, __MAX_CHANNEL_VALUE__, msg=(value, i))
                    self.assertGreaterEqual(value, __MIN_CHANNEL_VALUE__, msg=(value, i))

    def test_load_ml_ratings(self):
        ml_ratings = ml_helpers.load_ml_ratings()
        self.assertNotEqual(ml_ratings, dict())

    def test_simulation_setup_get_color_data(self):
        original = cl.load_as_dict_of_lists(normalize=True)
        flattened = setup.get_color_data()

        # Take a random list of keys
        index_list = rd.sample(list(original.keys()), __testing_sample_size__)

        for i in index_list:
            flat_original = [value for color in original[i] for value in color]
            self.assertListEqual(flat_original, flattened[i])


if __name__ == '__main__':
    unittest.main()
