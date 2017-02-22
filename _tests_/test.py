import soundtrack_dataset.soundtrack_retriever as sr
import  keywords.imdb_interface as imdb_r
import tools.movielens_helpers as mlens
import tools.thread_manager as tm
import time

def test_1():
    movielens = mlens.load_ml_movies("resources/movies.dat")
    soundtracks, errors = sr.get_soundtracks_from_movielens(movielens)
    return soundtracks, errors


def test_2():
    movielens = mlens.load_ml_movies("resources/movies.dat")
    for movie in movielens:
        return sr.wrapper_get_soundtrack_from_movie(movie)


def test_3():
    return sr.get_tracks_from_url("http://www.soundtrack.net/album/toy-story/")


def test_4():
    movielens = mlens.load_ml_movies("resources/movies.dat")
    # movielens = movielens[]
    soundtracks, errors = sr.get_soundtracks_from_movielens(movielens)
    sr.save_soundtrack_data(soundtracks, "resources/soundtrack.json")
    sr.save_soundtrack_data(errors, "resources/soundtrack-errors.json")
    print("Completed! " + str(len(soundtracks)) + " retrieved - " + str(len(errors)) + " failed")


def test_5():
    manager = tm.ThreadManager()

    def x_fun(x):
        for i in range(300):
            print(x + str(i))
            time.sleep(0.1)

    def y_fun(x):
        for i in range(150):
            print(x + str(i))
            time.sleep(0.1)

    op_list = []
    op_list.append(tm.Operation(manager, lambda x: y_fun('a')))
    op_list.append(tm.Operation(manager, lambda x: x_fun('b')))
    op_list.append(tm.Operation(manager, lambda x: x_fun('c')))
    op_list.append(tm.Operation(manager, lambda x: x_fun('d')))
    op_list.append(tm.Operation(manager, lambda x: x_fun('e')))
    op_list.append(tm.Operation(manager, lambda x: x_fun('f')))
    op_list.append(tm.Operation(manager, lambda x: x_fun('g')))
    op_list.append(tm.Operation(manager, lambda x: x_fun('h')))
    op_list.append(tm.Operation(manager, lambda x: x_fun('i')))
    op_list.append(tm.Operation(manager, lambda x: x_fun('j')))
    op_list.append(tm.Operation(manager, lambda x: x_fun('k')))
    op_list.append(tm.Operation(manager, lambda x: x_fun('l')))

    manager.run_all(op_list, max_threads=15)

def test_6():
    movie_data = sr.load_soundtrack_data("resources/soundtrack.json")
    movie_data_with_features, errors = sr.get_spotify_features_for_whole_dataset(movie_data)
    sr.save_soundtrack_data(movie_data_with_features, "resources/soundtrack-features.json")
    sr.save_soundtrack_data(errors, "resources/soundtrack-features-errors.json")
    print("Completed! " + str(len(movie_data_with_features)) + " retrieved - " + str(len(errors)) + " failed")

def test_7():
    manager = tm.ThreadManager()
    restore_operation = tm.Operation(manager, (lambda x: 0), op_id=9999)

    def x_fun(x):
        for i in range(300):
            print(x + str(i))
            time.sleep(0.1)

    def y_fun(x):
        raise Exception
        for i in range(150):
            print(x + str(i))
            time.sleep(0.1)

    op_list = []
    op_list.append(tm.Operation(manager, lambda x: y_fun('a')))
    op_list.append(tm.Operation(manager, lambda x: y_fun('b')))
    op_list.append(tm.Operation(manager, lambda x: x_fun('c')))
    op_list.append(tm.Operation(manager, lambda x: x_fun('d')))
    op_list.append(tm.Operation(manager, lambda x: x_fun('e')))
    op_list.append(tm.Operation(manager, lambda x: x_fun('f')))
    op_list.append(tm.Operation(manager, lambda x: x_fun('g')))
    op_list.append(tm.Operation(manager, lambda x: x_fun('h')))
    op_list.append(tm.Operation(manager, lambda x: x_fun('i')))
    op_list.append(tm.Operation(manager, lambda x: x_fun('j')))
    op_list.append(tm.Operation(manager, lambda x: x_fun('k')))
    op_list.append(tm.Operation(manager, lambda x: x_fun('l')))

    manager.run_all(op_list, restore_operation, max_threads=15)

def test_8():
    return imdb_r.get_imdb_id("Toy Story", "1995")

def test_9():
    return imdb_r.get_title_and_year("Toy Story (1995)")


def test_10():
    movie_data = mlens.load_ml_movies("resources/movies.dat")
    imdb_file = open("resources/imdb_movies.dat", 'w')
    keywords_file = open("resources/imdb_keywords.dat", 'w')
    failure_keyword_file = open("resources/keywords/failures.log", 'w')

    imdb_r.get_all_keywords_and_imdb_id_from_movielens(movie_data, imdb_file, keywords_file, failure_keyword_file)

    imdb_file.close()
    keywords_file.close()
    failure_keyword_file.close()

    print("Completed!")

def test_11():
    movie_data = mlens.load_ml_movies("resources/movies.dat")
    scripts_file = open("resources/scripts_movies.dat", 'w')
    failure_file = open("resources/scripts/failures.log", 'w')

    imdb_r.get_all_scripts_from_movielens(movie_data, scripts_file, failure_file)

    scripts_file.close()
    failure_file.close()

    print("Completed!")

if __name__ == "__main__":
    test_11()