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

c_values = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
gamma_values = [-5, -4, -3, -2, -1, 0, 1]

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

    def linear_svm(self, c_val=1.0):
        clf = svm.SVC(kernel='poly', C = c_val)
        clf.fit(self.bag_of_words, self.label_data)
        # Obtain accuracy on the train set
        y_hat = clf.predict(self.encoded_bag_of_words)
        acc = accuracy_score(y_hat, self.label_data)
        #print(f"Accuracy on the train set: {acc:.4f}")
        return clf


class Classification:

    bag_of_words = None
    encoded_bag_of_words = None
    label_data = None

    def __init__(self, bag_of_words, encoded_bag_of_words, label_data):
        self.bag_of_words = bag_of_words
        self.label_data = label_data
        self.encoded_bag_of_words = encoded_bag_of_words 

    def linear_svm(self, c_val=1.0):
        clf = svm.SVC(kernel='linear', C = c_val)
        clf.fit(self.bag_of_words, self.label_data)
        # Obtain accuracy on the train set
        y_hat = clf.predict(self.encoded_bag_of_words)
        acc = accuracy_score(y_hat, self.label_data)
        print(f"Accuracy on the train set (linear_svm): {acc:.4f}")
        return clf

    def quadratic_svm(self, c_val=1.0):
        clf = svm.SVC(kernel='poly', C = c_val)
        clf.fit(self.bag_of_words, self.label_data)
        # Obtain accuracy on the train set
        y_hat = clf.predict(self.encoded_bag_of_words)
        acc = accuracy_score(y_hat, self.label_data)
        print(f"Accuracy on the train set (quadratic_svm): {acc:.4f}")
        return clf

    def rbf_svm(self, c_val=1.0, gamma_val='scale'):
        clf = svm.SVC(kernel='rbf', C = c_val, gamma = gamma_val)
        clf.fit(self.bag_of_words, self.label_data)
        # Obtain accuracy on the train set
        y_hat = clf.predict(self.encoded_bag_of_words)
        acc = accuracy_score(y_hat, self.label_data)
        print(f"Accuracy on the train set (rbf_svm): {acc:.4f}")
        return clf



def main():

    """
    # This section of code pre-processes our data and completes
    # section 0.a of the assignment which is to check the top
    # ten most frequent words with the respectice vectorizer.
    #
    #
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

    test_processor = Processing("IA3-dev.csv")
    test_processor.load_data()
    test_processor.column_to_list("text")
    test_processor.column_to_list("sentiment")

    trained_tfidf_vectorizer = train_processor.tfidf_vectorizer
    x_test = trained_tfidf_vectorizer.transform(test_processor.text_list)

    trained_classifier = Classification(train_processor.bag_of_words, train_processor.encoded_bag_of_words, train_processor.label_data)

    """
    for val in c_values:
        linear_clf = trained_classifier.linear_svm(pow(10, val))
        quadratic_clf = trained_classifier.quadratic_svm(pow(10, val))
        ylinear_predict = linear_clf.predict(x_test)
        yquadratic_predict = quadratic_clf.predict(x_test)
        acc_linear = accuracy_score(ylinear_predict, test_processor.label_data)
        acc_quadratic = accuracy_score(yquadratic_predict, test_processor.label_data)
        print(f"Accuracy on the linear-svm test set with C value 10^{val}: {acc_linear:.4f}")
        print(f"Accuracy on the quadratic-svm test set with C value 10^{val}: {acc_quadratic:.4f}")
        print(f"{new_line}")
        #print(classification_report(test_processor.label_data, y_predict))
    """

    for val in c_values:
        for gamma in gamma_values:
            rbf_clf = trained_classifier.rbf_svm(pow(10, val), pow(10, gamma))
            yrbf_predict = rbf_clf.predict(x_test)
            acc_rbf = accuracy_score(yrbf_predict, test_processor.label_data)
            print(f"Accuracy on the rbf-svm test set with C value 10^{val} and gamma value {gamma}: {acc_rbf:.4f}")
            print(f"{new_line}")




if __name__ == '__main__':
    main()

