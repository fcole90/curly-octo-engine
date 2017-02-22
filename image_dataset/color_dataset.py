import os

"""
Tools to import the color dataset with ease.
"""

# ---------------------------------
# Definitions
# ---------------------------------
__COLOR_CHANNELS__ = 3
__HIGHEST_COLOR_VALUE__ = 255
__DATASET_FILE__ = "rgb-dataset-rebuilt.dat"


def get_dataset_file_path() -> str:
    """
    Returns the path to the color-dataset.

    Returns
    -------
    str:
        the path to the color dataset

    """
    name = __DATASET_FILE__
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def load_as_dict_of_lists(normalize=False) -> dict:
    """
    Loads the color-dataset as a dictionary of palettes.

    The dataset is a dictionary composed as:
     - key: the movie ID according to movielens
     - value: the movie palette as a list of lists

    The palette contains a list of colors, each color contains a list of RGB values.

    Parameters
    ----------
    normalize: bool opt
        if True the values are normalized in the range 0 - 1

    Returns
    -------
    dict of list of list of float

    """

    if normalize:
        factor = __HIGHEST_COLOR_VALUE__
    else:
        factor = 1

    # The color-dataset has lines in the following scheme:
    # #n::(r, g, b)::(r, g, b):: ... ::(r, g, b)\n
    with open(get_dataset_file_path(), "r") as color_dataset_file:
        movie_dict = dict()
        for line in color_dataset_file:
            line_list = line[:-1].split("::") # Also remove carriage return
            palette = []
            for color in line_list[1:]:
                color_list = color[1:-1].split(", ")
                rgb_channels_list = [float(color_list[0])/factor,
                                     float(color_list[1])/factor,
                                     float(color_list[2])/factor]
                palette.append(rgb_channels_list)
            movie_dict[int(line_list[0])] = palette

    return movie_dict


