import os

__movielens_movies_file__ = "movies.dat"
__resources_folder__ = "resources"


def load_movielens_movies(movielens_file_path=None):
    """
    Loads the movielens movies file as a list of movies,
    each movie as a dictionary.

    Parameters
    ----------
    movielens_file_path: str, optional
        path to the movies file

    Returns
    -------
    list of dict
        A list of movies. Each movie is a dictionary
        with the following items:
            id: str - the movie ID
            name: str - the name of the movie
            genre: list of genres of the movie

    """

    if not movielens_file_path:
        os.path.join(__resources_folder__, __movielens_movies_file__)

    movielens = []

    movielens_file = open(movielens_file_path, "r", encoding="latin-1")

    for line in movielens_file:
        movie_list = line[:-1].split("::")

        movie = {
            "id": movie_list[0],
            "name": movie_list[1],
            "genre": movie_list[2].split("|")
        }

        movielens.append(movie)

    return movielens
