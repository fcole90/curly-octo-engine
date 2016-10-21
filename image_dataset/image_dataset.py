#!/usr/bin/env python3

from tools import downloader
from image_dataset import image_data_extractor
from image_dataset import image_merger
from image_dataset import list_url_images as li
import os
import sys


def rebuild_colors(dataset):
    prefix_dir = "images"
    movielens = open(dataset, "r", encoding="latin-1")
    rgb_file = open("rgb-dataset-rebuilt.dat", "a+")

    for line in movielens:
        movie = line.split("::")[1]
        element = {
            "name": movie,
            "url_list": [],
            "dir": line.split("::")[0]
        }
        print(movie)

        dir_path = os.path.join(prefix_dir, element["dir"])

        for file in os.listdir(dir_path):
            imglist.append(os.path.join(dir_path, file))

        im = image_merger.ImageMerger(imglist)
        im.merge_and_save(prefix_dir, element["dir"] + ".jpg")

        ide = image_data_extractor.ImageDataExtractor(os.path.join(prefix_dir, element["dir"] + ".jpg"))
        ide.print_palette("screen")
        rgb_list = ide.get_palette(6)
        rgb_file.write(element["dir"])

        for rgb in rgb_list:
            rgb_file.write("::" + str(rgb))
            
        rgb_file.write("\n")

    rgb_file.close()


def find_errors(name=None):
    if not name:
        name = "rgb-dataset-rebuilt.dat"
    rgb_file = open(name, "r")
    missing_list = []

    index = 1
    for line in rgb_file:
        # Remove the carriage return character
        id = line[:-1].split("::")[0]

        while index < int(id):
            missing_list.append(index)
            index += 1

        index += 1

    return missing_list


def retrieve_missing(miss_list):
    movielens = open("movies.dat", "r", encoding="latin-1")
    movie_list = []

    for line in movielens:
        if not line.split("::")[0] in miss_list:
            continue
        movie = line.split("::")[1]
        element = {
            "name": line.split("::")[1],
            "url_list": [],
            "dir": line.split("::")[0]
        }
        movie_list.append(element)

    if len(movie_list) == 0:
        print("No missing found!")
    else:
        print("Starting retirvial:\n" + str(movie_list))

    iterator = 0
    sublist = []

    for element in movie_list:
        images = li.search(element["name"])
        element["url_list"] = images
        sublist.append(element)
        print(str(element["name"]) + " links taken..")

        iterator += 1

        if iterator % max_threads == 0 or element == movie_list[-1]:
            d = downloader.Downloader(sublist, max_threads)
            d.get()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Exactly one argument required: movies.dat")
    else:

        prefix_dir = "images"
        max_threads = 10

        movielens = open(sys.argv[1], "r", encoding="latin-1")
        movie_list = []
        for line in movielens:
            movie = line.split("::")[1]
            element = {
                "name": line.split("::")[1],
                "url_list": [],
                "dir": line.split("::")[0]
            }
            movie_list.append(element)
            print(movie)

        iterator = 0
        sublist = []

        for element in movie_list:
            images = li.search(element["name"])
            element["url_list"] = images
            sublist.append(element)
            print(str(element["name"]) + " links taken..")

            iterator += 1

            if iterator % max_threads == 0 or element == movie_list[-1]:
                d = downloader.Downloader(sublist, max_threads)
                d.get()

                rgb_file = open("rgb-dataset.dat", "a+")

                for element in sublist:
                    imglist = []
                    dir_path = os.path.join(prefix_dir, element["dir"])
                    for file in os.listdir(dir_path):
                        imglist.append(os.path.join(dir_path, file))

                    im = image_merger.ImageMerger(imglist)
                    im.merge_and_save(prefix_dir, element["dir"] + ".jpg")
                    file_name = os.path.join(prefix_dir, element["dir"])

                    ide = image_data_extractor.ImageDataExtractor(os.path.join(prefix_dir, element["dir"] + ".jpg"))
                    ide.print_palette("screen")
                    ide.print_palette("txt", file_name + ".txt")
                    ide.print_palette("html", file_name + ".html")
                    rgb_list = ide.get_palette(6)
                    rgb_file.write(element["dir"])
                    for rgb in rgb_list:
                        rgb_file.write("::" + str(rgb))
                    rgb_file.write("\n")

                sublist = []
                rgb_file.close()
