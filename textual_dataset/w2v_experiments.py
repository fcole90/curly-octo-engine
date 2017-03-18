import logging
from gensim.models import word2vec
import gensim

# Configure logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


# Keggle default
def train_model_1(sentences):
    # Set values for various parameters
    num_features = 300  # Word vector dimensionality
    min_word_count = 40  # Minimum word count
    num_workers = 8  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words

    return word2vec.Word2Vec(sentences, workers=num_workers,
                             size=num_features, min_count=min_word_count,
                             window=context, sample=downsampling)


# Lower min word count
def train_model_2(sentences):
    # Set values for various parameters
    num_features = 300  # Word vector dimensionality
    min_word_count = 10  # Minimum word count
    num_workers = 8  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words

    return word2vec.Word2Vec(sentences, workers=num_workers,
                             size=num_features, min_count=min_word_count,
                             window=context, sample=downsampling)

# Higher context
def train_model_3(sentences):
    # Set values for various parameters
    num_features = 300  # Word vector dimensionality
    min_word_count = 40  # Minimum word count
    num_workers = 8  # Number of threads to run in parallel
    context = 15  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words

    return word2vec.Word2Vec(sentences, workers=num_workers,
                             size=num_features, min_count=min_word_count,
                             window=context, sample=downsampling)


# Get words like new_york
def get_multiwords(sentences):
    return gensim.models.Phrases(sentences)