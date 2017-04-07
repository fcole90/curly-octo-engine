"""
Edited from http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold

from tools.movielens_helpers import load_ml_movies
import visual_data_simulation.simulation_setup as setup

data = setup.Setup(user_data_function="clusters",
                   limit_memory_usage=False,
                   movie_amount_limit=20)

#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding_with_labels(title, X, labels, category=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    shown_images = np.array([[1., 1.]])  # just something big
    for i in range(len(labels)):
        dist = np.sum((X[i] - shown_images) ** 2, 1)
        if np.min(dist) < 4e-5:
            # don't show points that are too close
            continue
        shown_images = np.r_[shown_images, [X[i]]]
        if category:
            color = category[i]
        else:
            color = X[i, 0] * X[i, 1]
        plt.text(X[i, 0], X[i, 1], labels[i],
                 color=plt.cm.get_cmap(name="Paired")(color),
                 fontdict={'size': 6},
                 bbox={'facecolor': 'white', 'alpha': 0.6, 'pad': 1})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

def plot_embedding(title, X, categories, range):
    if categories:
        color = categories
    else:
        color = X[:, 0] % X[:, 1]
    plt.scatter(X[:, 0], X[:, 1], c=color, cmap=plt.cm.get_cmap("jet", max(categories)))
    plt.colorbar(ticks=range)
    plt.clim(0, max(categories))


def print_movie_colors_tsne():
    print("Computing t-SNE embedding")
    genres = {"Action": 0.,
              "Adventure": 0.1,
              "Animation": 1.,
              "Children's": 1.1,
              "Comedy": 1.2,
              "Crime": 2.,
              "Documentary": 3.,
              "Drama": 4.,
              "Fantasy": 1.3,
              "Film-Noir": 2.1,
              "Horror": 5.,
              "Musical": 6.,
              "Mystery": 2.2,
              "Romance": 4.1,
              "Sci-Fi": 2.2,
              "Thriller": 0.2,
              "War": 2.2,
              "Western": 7.}
    # genres = "Action,Adventure,Animation,Children's,Comedy,Crime,Documentary," \
    #          "Drama,Fantasy,Film-Noir,Horror,Musical,Mystery,Romance,Sci-Fi," \
    #          "Thriller,War,Western".split(",")
    movie_colors_id_list = list(data.color_data.keys())[10000]
    movie_colors_matrix = np.array([data.color_data[movie_id]
                                    for movie_id in movie_colors_id_list])
    ml_movies_data = load_ml_movies(include_genres=True)

    movie_labels = [", ".join(ml_movies_data[movie_id][1])
                    for movie_id in movie_colors_id_list]

    movie_categories = [genres[(ml_movies_data[movie_id][1][0])]
                    for movie_id in movie_colors_id_list]

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=3)
    X_tsne = tsne.fit_transform(movie_colors_matrix)

    plot_embedding_with_labels("t-SNE embedding of the movie colors", X_tsne, movie_labels, movie_categories)
    # plot_embedding("t-SNE colors", X_tsne, movie_categories, list(genres.values()))

    plt.show()

if __name__ == '__main__':
    print_movie_colors_tsne()