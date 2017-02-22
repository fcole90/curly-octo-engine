import re


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

