import os.path
import pickle
import re
import sys

__CACHE_DIR__ = ".cache"

# --- Functions on paths ---
def get_project_root():
    """
    Get the project root dir.

    Returns
    -------
    str
        Path to the project root.
    """
    here_path = os.path.realpath(__file__)
    return os.path.abspath(os.path.join(here_path, os.pardir, os.pardir))

def make_cache_folder():
    """
    Creates a cache folder if it doesn't exist.
    """
    os.makedirs(os.path.join(get_project_root(), __CACHE_DIR__), exist_ok=True)

def get_cache_path(cache_file_name):
    return os.path.join(get_project_root(),
                 __CACHE_DIR__,
                 str(cache_file_name) + ".cache")

def cache_file_exists(cache_file_name):
    """
    Check the existence of a cache file.
    """
    return os.path.exists(get_cache_path(cache_file_name))

def save_object_to_cache(object, cache_file_name):
    make_cache_folder()
    path = get_cache_path(cache_file_name)
    with open(path, 'wb+') as cache_file:
        pickle.dump(object, cache_file)

def load_object_from_cache(cache_file_name):
    path = get_cache_path(cache_file_name)
    with open(path, "rb") as cache_file:
        return pickle.load(cache_file)





# --- Functions on strings ---
def clean_string(my_string):
    my_string = re.sub("[^A-Za-z]", " ", my_string)
    return my_string.rstrip().lstrip()

def reform_title(title):
    title_list = title.split(',')
    return title_list[0]

def clean_number(my_string):
    my_string = re.sub("[^0-9]", " ", my_string)
    return my_string.rstrip().lstrip()

# --- Deprecation ---

def deprecated(function: callable) -> callable:
    """
    Deprecation decorator.

    Parameters
    ----------
    function: callable
        function to deprecate

    Returns
    -------
        function with deprecation warning

    """
    import warnings
    def function_with_deprecations(*args, **kwargs):
        warnings.warn(function.__name__ + " is deprecated.", DeprecationWarning, stacklevel=2)
        return function(*args, **kwargs)
    return function_with_deprecations


def updating_text(*args, **kwargs):
    """
    Prints a text that can be updated multiple times.
    """
    sys.stdout.write(*args, **kwargs)
    sys.stdout.flush()