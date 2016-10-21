import os


def get_dataset_file(name=None):
    DATASET_FILE = "rgb-dataset-rebuilt.dat"
    if not name:
        name = DATASET_FILE
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def load():
    """
    Loads the color-dataset.

    The dataset is a list of dictionaries.
    Each dictionary is composed as:
     - id: the movie ID according to movielens
     - palette: the movie palette

    Each palette is a list of tuples.

    Each tuple is a triple of RGB colors (r, g, b).

    Returns
    -------
    list of dict

    """

    rgb_file = open(get_dataset_file(), "r")

    movie_list = []
    for line in rgb_file:
        line_list = line.split("::")

        palette = []
        for color in line_list[1:]:
            rgb_list = color[1:-1].split(", ")
            rgb_tuple = (rgb_list[0], rgb_list[1], rgb_list[2])
            palette.append(rgb_tuple)

        movie_dict = {"id": line_list[0], "palette": palette}
        movie_list.append(movie_dict)

    return movie_list


def vector_load():
    """
        Loads the color-dataset as a vector.

        The dataset is a list of matrices.
        Each matrix is shaped (palette_amount x color_channels_amount)

        Returns
        -------
        list of matrices

        """
    import numpy as np
    COLOR_CHANNELS = 3
    GAMMA_VALUES = 256.0

    rgb_file = open(get_dataset_file(), "r")

    movie_list = []
    for line in rgb_file:
        # Remove the carriage return character
        line_list = line[:-1].split("::")

        palette_vector = np.zeros((COLOR_CHANNELS*len(line_list[1:])))

        # Skip the first item because it's the movie ID
        for i, color in enumerate(line_list[1:]):
            rgb_list = color[1:-1].split(", ")
            for j, channel in enumerate(rgb_list):
                palette_vector[i*j] = float(rgb_list[j])

        # Keep the values in a 0-1 range
        movie_list.append(palette_vector / GAMMA_VALUES)

    full_vector = np.zeros((len(movie_list), (COLOR_CHANNELS*len(line_list[1:]))))
    for i, movie in enumerate(movie_list):
        full_vector[i:] = movie

    return full_vector


