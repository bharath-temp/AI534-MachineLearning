import numpy as np
import pandas as pd
from heapq import nlargest
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from enum import Enum

new_line = '\n'

categories = ["positive", "negative"]

class Sentiment(Enum):
    POS = 1
    NEG = 0

class Processing:

    label_data = None
    loaded_data = None
    text_list = None
    sentiment_list = None
    bag_of_words = None
    encoded_bag_of_words = None
    freq_words = None
    count_vectorizer = None
    tfidf_vectorizer = None
    sentiment = None

    def __init__(self, path, sent=None):
        self.path = path
        if sent != None:
            self.sentiment = Sentiment[sent].value

    def load_data(self):
        df = pd.read_csv(self.path)
        self.label_data = df["sentiment"]
        if self.sentiment != None:
            self.loaded_data = df[df["sentiment"] == self.sentiment]
        else:
            self.loaded_data = df

    def column_to_list(self, column):
        if column is "text":
            self.text_list = self.loaded_data[column].tolist()
        elif column is "sentiment":
            self.sentiment_list = self.loaded_data[column].tolist()

    def feature_extraction_count(self):
        self.count_vectorizer = CountVectorizer(lowercase=True)
        self.count_vectorizer.fit(self.text_list)
        self.bag_of_words = self.count_vectorizer.fit_transform(self.text_list)
        self.encoded_bag_of_words = self.bag_of_words.toarray()

    def top_frequent_words_count(self, n):
        sum_words = self.bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in self.count_vectorizer.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        self.freq_words = words_freq[:n]

        return self.freq_words

    def feature_extraction_tfidf(self):
        self.tfidf_vectorizer = TfidfVectorizer(use_idf=True, lowercase=True)
        self.tfidf_vectorizer.fit(self.text_list)
        self.bag_of_words = self.tfidf_vectorizer.fit_transform(self.text_list)
        self.encoded_bag_of_words = self.bag_of_words.toarray()

    def top_frequent_words_tfidf(self, n):
        sum_words = self.bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in self.tfidf_vectorizer.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        self.freq_words = words_freq[:n]

        return self.freq_words

    def linear_svm(self):
        clf = svm.SVC(kernel='linear', C = 1.0)
        clf.fit(self.bag_of_words, self.label_data)
        # Obtain accuracy on the train set
        y_hat = clf.predict(self.encoded_bag_of_words)
        acc = accuracy_score(y_hat, self.label_data)
        print(f"Accuracy on the train set: {acc:.4f}")
        return clf



def main():

    """
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
    """

    train_processor = Processing("IA3-train.csv")
    train_processor.load_data()
    train_processor.column_to_list("text")
    train_processor.column_to_list("sentiment")
    train_processor.feature_extraction_tfidf()
    clf = train_processor.linear_svm()

    test_processor = Processing("IA3-dev.csv")
    test_processor.load_data()
    test_processor.column_to_list("text")
    test_processor.column_to_list("sentiment")

    trained_tfidf_vectorizer = train_processor.tfidf_vectorizer
    x_test = trained_tfidf_vectorizer.transform(test_processor.text_list)
    
    y_predict = clf.predict(x_test)

    acc = accuracy_score(y_predict, test_processor.label_data)
    print(f"Accuracy on the test set: {acc:.4f}")
    print(classification_report(test_processor.label_data, y_predict))






if __name__ == '__main__':
    main()

