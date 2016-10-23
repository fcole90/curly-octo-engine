def clean_string(my_string):
    replacements = {
        "&": " ",
        "#": " ",
        ":": " ",
        "-": " ",
        "'": " ",
        ".": " ",
        ",": " "
    }
    my_string = my_string.replace("...", " ")
    my_string = "".join([replacements.get(char, char) for char in my_string])
    return my_string