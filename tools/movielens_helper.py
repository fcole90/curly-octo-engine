import os

__movielens_movies_file__ = "movies.dat"
__movie_plots_file__ = "scripts_movies.dat"
__script_full_file__ = "full_scripts_texts_raw.txt"
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
        movielens_file_path = os.path.join(__resources_folder__, __movielens_movies_file__)

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

def load_movie_plots(movie_plots_file_path=None):
    """
    Loads the movie plots file as a list of movies,
    each movie as a dictionary.

    Parameters
    ----------
    movie_plots_file_path: str, optional
        path to the movie plots file

    Returns
    -------
    list of dict
        A list of movies. Each movie is a dictionary
        with the following items:
            id: str - the movie ID
            name: str - the name of the movie
            genre: list of genres of the movie

    """

    if not movie_plots_file_path:
        movie_plots_file_path = os.path.join(__resources_folder__, __movie_plots_file__)

    movie_plots = []

    movie_plots_file = open(movie_plots_file_path, "r", encoding="latin-1")

    """In this cycle the line is one step behind,
       this allows to look at the following line,
       but requires some actions do be done past the end"""

    line = None
    for following_line in movie_plots_file:
        if line == None:
            line = following_line
            continue

        if line.startswith('@newmovie'):
            movie_list = line.split("::")
            movie = {
                "id": movie_list[1],
                "is_long": movie_list[2],
                "plot": ""
            }
        else:
            movie["plot"] += line

        if following_line.startswith('@newmovie'):
            movie_plots.append(movie)

        line = following_line

    movie["plot"] += following_line # Add the latest line
    movie_plots.append(movie) #Append the latest movie

    return movie_plots


def save_plots_as_full_text(plot_list, full_text_file_path = None):
    if full_text_file_path == None:
        full_text_file_path = os.path.join(__resources_folder__, __script_full_file__)
    full_text_file = open(full_text_file_path, "w")
    for movie in plot_list:
        if len(movie["plot"]) > 500:
            full_text_file.write(movie["plot"])
    full_text_file.close()




