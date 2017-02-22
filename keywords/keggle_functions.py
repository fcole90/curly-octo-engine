from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
from sklearn.cluster import KMeans
import time

# Most of the functions in this file come or are adapted from [1]
# 1: https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-3-more-fun-with-word-vectors


def load_nltk():
    nltk.download()


def get_nltk_tokenizer():
    # Load the punkt tokenizer
    return nltk.data.load('tokenizers/punkt/english.pickle')


def review_to_wordlist(review, remove_stopwords=False):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review, "lxml").get_text()
    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]
    #
    # 5. Return a list of words
    return words


# Define a function to split a review into parsed sentences (lists of words)
def review_to_sentences(review, tokenizer, remove_stopwords=False):
    # Function to split a review into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append(review_to_wordlist(raw_sentence,
                             remove_stopwords))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences


def scan_all_sentences(list_of_sentences, update_frequency=50):
    sentences = []  # Initialize an empty list of sentences
    tokenizer = get_nltk_tokenizer()

    for i, sensence in enumerate(list_of_sentences):
        # Print some useful output
        if i % update_frequency:
            print(str(i) + " of " + str(len(list_of_sentences)))
        sentences += review_to_sentences(sensence, tokenizer)

    return sentences


def make_clusters(model):
    start = time.time()  # Start time

    # Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
    # average of 5 words per cluster
    word_vectors = model.syn0
    num_clusters = int(word_vectors.shape[0] / 5)

    # Initalize a k-means object and use it to extract centroids
    kmeans_clustering = KMeans(n_clusters=num_clusters, n_jobs=-1)
    idx = kmeans_clustering.fit_predict(word_vectors)

    # Get the end time and print how long the process took
    end = time.time()
    elapsed = end - start
    print("Time taken for K Means clustering: " + elapsed + "seconds.")

    return idx


def cluster_word_dict(idx, model):
    return dict(zip(model.index2word, idx))


def print_clusters(cluster, cluster_word_dict, amount=-1):

    if amount == -1:
        amount = len(cluster)

    for cluster in range(amount):
        #
        # Print the cluster number
        print("Cluster: " + str(cluster))
        #
        # Find all of the words for that cluster number, and print them out
        words = []
        for i, value in enumerate(cluster_word_dict.values()):
            if (value == cluster):
                c_keys = list(cluster_word_dict.keys())
                words.append(c_keys[i])
        print(str(words))
