import os

from tools.generic_helpers import deprecated

__resources_folder__ = os.path.join('..', 'resources')

# --- Movielens resources ---
__ml_folder__ = os.path.join(__resources_folder__, "ml-1m")
__ml_ratings_file__name__ = "ratings.dat"
__ml_movies_file_name__ = "movies.dat"
# -----------------

# --- Other resources ---
__movie_plots_file_name__ = "scripts_movies.dat"
__script_full_file_name__ = "full_scripts_texts_raw.txt"
# -----------------------


# --- Movielens functions ---
def load_ml_movies(include_genres:bool=False):
    """
    Load ml_movies as a dict.

    The key is the movie id, the value the data of the movie.
    If include_genres is false, it returns the title of the movie, else it returns
    a tuple (title, list of genres).

    Parameters
    ----------
    include_genres: bool
        flag to include also the movie genres

    Returns
    -------
    Returns
    -------
    dict of (str OR tuple of (str, list of str))

    """
    # Reference structure of ml_movies.
    # MovieID::Title::Genre1|Genre2| ... |GenreN

    ml_movies_file_path = os.path.join(__ml_folder__, __ml_movies_file_name__)

    with open(ml_movies_file_path, "r", encoding="latin-1")\
        as ml_movies_file:

        ml_movies = dict()
        for line in ml_movies_file:
            line_list = line[:-1].split('::') # Also remove carriage return
            movie_id = int(line_list[0])
            title = str(line_list[1])
            genre_list = line_list[2].split('|')

            if include_genres:
                ml_movies[movie_id] = (title, genre_list)
            else:
                ml_movies[movie_id] = title
    return ml_movies


def load_ml_ratings(include_timestamp:bool=False) -> dict:
    """
    Load ml_ratings as a dict.

    UserIDs are the keys, each value is a dict of rated movies.
    MovieIDs are the keys of each contained dict.
    Each entry in the dict is composed by (rating, timestamp) if
    include_timestamps is true, else it only contains rating.

    Parameters
    ----------
    include_timestamp: bool
        flag to include also timestamps

    Returns
    -------
    dict of dict of (int OR tuple of int)

    """

    # Reference structure of ml_ratings.
    # UserID::MovieID::Rating::Timestamp

    ml_ratings_file_path = os.path.join(__ml_folder__, __ml_ratings_file__name__)
    with open(ml_ratings_file_path, "r")\
        as ml_ratings_file:

        ml_ratings = dict()
        for line in ml_ratings_file:
            line_list = line.split('::')
            user_id = int(line_list[0])
            movie_id = int(line_list[1])
            rating = int(line_list[2])
            timestamp = int(line_list[3])

            if user_id not in ml_ratings.keys():
                ml_ratings[user_id] = dict()

            if include_timestamp:
                ml_ratings[user_id][movie_id] = (rating, timestamp)
            else:
                ml_ratings[user_id][movie_id] = rating

    return ml_ratings

@deprecated
def __load_ml_movies__(ml_movies_file_path=None):
    """Deprecated. Loads the ml_movies movies file as a list of dictionaries.

    Each dictionary represents the data of a movie.

    Parameters
    ----------
    ml_movies_file_path: str, optional
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
    if not ml_movies_file_path:
        ml_movies_file_path = os.path.join(__resources_folder__, __ml_movies_file_name__)

    ml_movies = list()
    ml_movies_file = open(ml_movies_file_path, "r", encoding="latin-1")

    for line in ml_movies_file:
        movie_list = line[:-1].split("::")

        movie = {
            "id": movie_list[0],
            "name": movie_list[1],
            "genre": movie_list[2].split("|")
        }
        ml_movies.append(movie)
    ml_movies_file.close()
    return ml_movies
# ---------------------------


# --- Plots Functions ---
def load_movie_plots(movie_plots_file_path=None):
    """
    Loads the movie plots file as a list of movies, each movie as a dictionary.

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
        movie_plots_file_path = os.path.join(__resources_folder__, __movie_plots_file_name__)

    movie_plots = []

    movie_plots_file = open(movie_plots_file_path, "r", encoding="latin-1")

    """In this cycle the line is one step behind,
       this allows to look at the following line,
       but requires some actions do be done past the end"""

    line = None
    for following_line in movie_plots_file:
        if line is None:
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
    movie_plots.append(movie) # Append the latest movie

    return movie_plots


def save_plots_as_full_text(plot_list, full_text_file_path = None):
    if full_text_file_path == None:
        full_text_file_path = os.path.join(__resources_folder__, __script_full_file_name__)
    full_text_file = open(full_text_file_path, "w")
    for movie in plot_list:
        if len(movie["plot"]) > 500:
            full_text_file.write(movie["plot"])
    full_text_file.close()
# -----------------------



