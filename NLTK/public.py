import nltk
import random
from nltk.corpus import movie_reviews


# create a dict to indicate whether a word exits in top 3000 words list
def find_features(document, word_features):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


def return_features():
    documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]

    random.shuffle(documents)

    all_words = []

    for w in movie_reviews.words():
        all_words.append(w.lower())

    all_words = nltk.FreqDist(all_words)

    # most frequently used top 3000 words
    word_features = list(all_words.keys())[:3000]
    featuresets = [(find_features(rev, word_features), category) for (rev, category) in documents]
    return featuresets

fea = return_features()
print(fea[0][0])

