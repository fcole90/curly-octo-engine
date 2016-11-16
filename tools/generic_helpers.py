import re


def clean_string(my_string):
    my_string = re.sub("[^A-Za-z]", " ", my_string)
    return my_string.rstrip()

def reform_title(title):
    title_list = title.split(',')
    return title_list[0]

def clean_number(my_string):
    my_string = re.sub("[^0-9]", " ", my_string)
    return my_string.rstrip()