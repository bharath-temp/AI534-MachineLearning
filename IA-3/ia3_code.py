import numpy as np
import pandas as pd
from heapq import nlargest
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from enum import Enum

new_line = '\n'

class Sentiment(Enum):
    POS = 1
    NEG = 0

class Processing:

    loaded_data = None
    text_list = None
    bag_of_words = None
    freq_words = None
    count_vectorizer = None
    tfidf_vectorizer = None

    def __init__(self, path, sent):
        self.path = path
        self.sentiment = Sentiment[sent].value

    def load_data(self):
        df = pd.read_csv(self.path)
        self.loaded_data = df[df["sentiment"] == self.sentiment]

    def column_to_list(self, column):
        self.text_list = self.loaded_data[column].tolist()

    def feature_extraction_count(self):
        self.count_vectorizer = CountVectorizer()
        self.count_vectorizer.fit(self.text_list)
        self.bag_of_words = self.count_vectorizer.transform(self.text_list)

    def top_frequent_words_count(self, n):
        sum_words = self.bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in self.count_vectorizer.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        self.freq_words = words_freq[:n]

        return self.freq_words

    def feature_extraction_tfidf(self):
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_vectorizer.fit(self.text_list)
        self.bag_of_words = self.tfidf_vectorizer.transform(self.text_list)

    def top_frequent_words_tfidf(self, n):
        sum_words = self.bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in self.tfidf_vectorizer.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        self.freq_words = words_freq[:n]

        return self.freq_words



def main():

    pos_processor = Processing("IA3-train.csv", "POS")
    pos_processor.load_data()
    pos_processor.column_to_list("text")
    pos_processor.feature_extraction_count()


    neg_processor = Processing("IA3-train.csv", "NEG")
    neg_processor.load_data()
    neg_processor.column_to_list("text")
    neg_processor.feature_extraction_count()

    top_pos_words_count = pos_processor.top_frequent_words_count(10)
    top_neg_words_count = neg_processor.top_frequent_words_count(10)

    print(f"The Top Positive words(count): {top_pos_words_count}")
    print(f"{new_line}")
    print(f"The Top Negative words(count): {top_neg_words_count}")
    print(f"{new_line}")

    pos_processor.feature_extraction_tfidf()
    neg_processor.feature_extraction_tfidf()

    top_pos_words_tfidf = pos_processor.top_frequent_words_tfidf(10)
    top_neg_words_tfidf = neg_processor.top_frequent_words_tfidf(10)

    print(f"{new_line}")
    print(f"The Top Positive words(tfidf): {top_pos_words_tfidf}")
    print(f"{new_line}")
    print(f"The Top Negative words(tfidf): {top_neg_words_tfidf}")


if __name__ == '__main__':
    main()