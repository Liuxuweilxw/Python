from numba import cuda
import os
from sklearn import svm
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np


precision = []
recall = []

# 从txt文件获取训练集和测试集
def input_data(train_file, test_file):
    train_words = []
    train_tags = []
    test_words = []
    test_tags = []
    with open(train_file, 'r', encoding='utf-8') as f1:
        for line in f1:
            #每一行根据空格被分为tags 和 句子
            tks = line.split('\t', 1)
            #句子
            train_words.append(tks[1])
            #tags
            train_tags.append(tks[0])
        # print(train_words)
        # print(train_tags)
        # train_words = np.r_[train_words, test_tags]
    with open(test_file, 'r', encoding='utf-8') as f1:
        for line in f1:
            tks = line.split('\t', 1)
            test_words.append(tks[1])
            test_tags.append(tks[0])
        # print(test_words)
        # print(test_tags)
        # test_words = np.r_[test_words, test_tags]
    return train_words, train_tags, test_words, test_tags

comma_tokenizer = lambda x: jieba.cut(x, cut_all=True)

def vectorize(train_words, test_words):
    v = HashingVectorizer(tokenizer=comma_tokenizer, n_features=30000)
    train_data = v.fit_transform(train_words)
    test_data = v.fit_transform(test_words)
    return train_data, test_data

def tfvectorize(train_words,test_words):
    v = TfidfVectorizer(tokenizer=comma_tokenizer,binary = False, decode_error = 'ignore')
    train_data = v.fit_transform(train_words)
    test_data = v.transform(test_words)
    return train_data,test_data

def covectorize(train_words,test_words):
    v = CountVectorizer(tokenizer=comma_tokenizer,binary = False, decode_error = 'ignore')
    train_data = v.fit_transform(train_words)
    test_data = v.transform(test_words)
    return train_data,test_data

# 朴素贝叶斯分类器
def train_clf_bayes(train_data, train_tags, alpha):
    clf = MultinomialNB(alpha=alpha)
    clf.fit(train_data, np.asarray(train_tags))
    return clf

# svm分类器
def train_clf_svc(train_data, train_tags):
    # 惩罚因子为20时 准确率为0.3左右，40时为0.5左右，100时为0.6左右
    clf = svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
                  gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True,
                  tol=0.001, verbose=False)
    clf.fit(train_data, np.asarray(train_tags))
    return clf

#计算准确率和召回率
def evaluate(actual, pred):
    m_precision = metrics.precision_score(actual, pred, average = 'weighted')
    m_recall = metrics.recall_score(actual, pred, average= 'weighted')
    print('precision:{0:.4f}'.format(m_precision))
    print('recall:{0:0.4f}'.format(m_recall))

def evaluate_bayes_test_alpha(actual, pred):
    m_precision = metrics.precision_score(actual, pred, average = 'weighted')
    m_recall = metrics.recall_score(actual, pred, average= 'weighted')
    # print('precision:{0:.4f}'.format(m_precision))
    # print('recall:{0:0.4f}'.format(m_recall))
    precision.append(m_precision)
    recall.append(m_recall)

# 对不同alpha值进行测试 找出准确率达到最大时的给定范围内的alpha值
def bayes_test_alpha(train_data,train_tags,test_tags):
    alphas = []
    i = 0
    for alpha in np.arange(0, 2, 0.01):
        alphas.append([i, alpha])
        i += 1
        # 朴素贝叶斯分类器
        clf = train_clf_bayes(train_data, train_tags, alpha)
        # 由测试数据预测标签结果
        re = clf.predict(test_data)
        # print("目标结果：", test_tags)
        # print("预测结果：", re)
        # 比较目标标签与预测标签 计算准确率和召回率
        evaluate_bayes_test_alpha(np.asarray(test_tags), re)
    print(alphas)
    print(precision)
    print(recall)
    print(np.amax(precision))
    print(np.argmax(precision))

def bayes_test(train_data,train_tags,alpha):
    # 朴素贝叶斯分类器
    clf = train_clf_bayes(train_data, train_tags, alpha)
    # 由测试数据预测标签结果
    re = clf.predict(test_data)
    # print("目标结果：", test_tags)
    # print("预测结果：", re)
    # 比较目标标签与预测标签 计算准确率和召回率
    evaluate(np.asarray(test_tags), re)

def svc_test(train_data,train_tags):
    # svm分类器
    clf = train_clf_svc(train_data, train_tags)
    # 由测试数据预测标签结果
    re = clf.predict(test_data)
    # print("目标结果：", test_tags)
    # print("预测结果：", re)
    # 比较目标标签与预测标签 计算准确率和召回率
    evaluate(np.asarray(test_tags), re)

if __name__ == '__main__':

    train_words, train_tags, test_words, test_tags = input_data('sample_data/train_file.txt', 'sample_data/test_file.txt')

    # bayes + this 0.7874
    # svc + this 0.601
    # train_data, test_data = covectorize(train_words, test_words)

    # bayes + this 0.7892
    # svc + this 0.0010
    train_data, test_data = tfvectorize(train_words, test_words)

    # bayes + this error : ValueError: Negative values in data passed to MultinomialNB (input X)
    # svc +this error: float() argument must be a string or a number, not 'HashingVectorizer'
    # train_data, test_data = vectorize(train_words, test_words)

    # svc_test(train_data,train_tags)
    # svc_test(train_data,train_tags)

    # bayes_test(train_data, train_tags, alpha=0.2)
    # bayes_test(train_data,train_tags, alpha=0.2)

    # 对不同alpha值进行测试 选取准确率达到最大时的alpha值
    bayes_test_alpha(train_data,train_tags,test_tags)

