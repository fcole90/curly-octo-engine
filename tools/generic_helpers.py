import re


def clean_string(my_string):
    my_string = re.sub("[^A-Za-z]", " ", my_string)
    return my_string