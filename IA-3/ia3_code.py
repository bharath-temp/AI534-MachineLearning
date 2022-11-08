import numpy as np
import pandas as pd
from heapq import nlargest
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer

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
    vectorizer = None

    def __init__(self, path, sent):
        self.path = path
        self.sentiment = Sentiment[sent].value

    def load_data(self):
        df = pd.read_csv(self.path)
        self.loaded_data = df[df["sentiment"] == self.sentiment]

    def column_to_list(self, column):
        self.text_list = self.loaded_data[column].tolist()

    def feature_extraction(self):
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(self.text_list)
        self.bag_of_words = self.vectorizer.transform(self.text_list)

    def top_frequent_words(self, n):
        sum_words = self.bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in self.vectorizer.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        self.freq_words = words_freq[:n]

        return self.freq_words



def main():

    pos_processor = Processing("IA3-train.csv", "POS")
    pos_processor.load_data()
    pos_processor.column_to_list("text")
    pos_processor.feature_extraction()


    neg_processor = Processing("IA3-train.csv", "NEG")
    neg_processor.load_data()
    neg_processor.column_to_list("text")
    neg_processor.feature_extraction()

    top_pos_words = pos_processor.top_frequent_words(10)
    top_neg_words = neg_processor.top_frequent_words(10)

    print(f"The Top Positive words are: {top_pos_words}")
    print(f"{new_line}")
    print(f"The Top Negative words are: {top_neg_words}")


if __name__ == '__main__':
    main()