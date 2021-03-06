# 朴素贝叶斯分类器
import sys
import jieba
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB


def input_data(train_file, test_file):
    train_words = []
    train_tags = []
    test_words = []
    test_tags = []
    with open(train_file, 'r', encoding='utf-8') as f1:
        for line in f1:
            tks = line.split('\t', 1)
            train_words.append(tks[1])
            train_tags.append(tks[0])
        print(train_words)
        print(train_tags)
        # train_words = np.r_[train_words, test_tags]
    with open(test_file, 'r', encoding='utf-8') as f1:
        for line in f1:
            tks = line.split('\t', 1)
            test_words.append(tks[1])
            test_tags.append(tks[0])
        print(test_words)
        print(test_tags)
        # test_words = np.r_[test_words, test_tags]
    return train_words, train_tags, test_words, test_tags

#
# with open('stopwords.txt', 'r') as f:
#     stopwords = set([w.strip() for w in f])
comma_tokenizer = lambda x: jieba.cut(x, cut_all=True)


def vectorize(train_words, test_words):
    v = HashingVectorizer(tokenizer=comma_tokenizer, n_features=30000)
    train_data = v.fit_transform(train_words)
    test_data = v.fit(test_words)
    return train_data, test_data


def evaluate(actual, pred):
    m_precision = metrics.precision_score(actual, pred)
    m_recall = metrics.recall_score(actual, pred)
    print('precision:{0:.3f}'.format(m_precision))
    print('recall:{0:.3f}'.format(m_recall))


def train_clf(train_data, train_tags):
    clf = MultinomialNB(alpha=0.01)
    clf.fit(train_data, np.asarray(train_tags))
    return clf


def main(train_file_path, test_file_path):
    train_file = train_file_path
    test_file = test_file_path
    train_words, train_tags, test_words, test_tags = input_data(train_file, test_file)
    train_data, test_data = vectorize(train_words, test_words)
    clf = train_clf(train_data, train_tags)
    pred = clf.predict(test_data)
    evaluate(np.asarray(test_tags), pred)


if __name__ == '__main__':
    main('sample_data/train_file.txt', 'sample_data/test_file.txt')
