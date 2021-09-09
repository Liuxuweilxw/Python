# -*- coding: utf-8 -*-
from sklearn import svm
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
import os
import re

precision = []
recall = []
train_words=[]
train_tags=[]
test_words = []
test_tags = []

comma_tokenizer = lambda x: jieba.cut(x)

# 从txt文件获取训练集和测试集
def input_data_train(train_file, train_tag):
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in range(2):
            tks = f.readline().strip()
        tks = re.sub(r'[^\u4e00-\u9fa5]', '', tks)
        train_words.append(tks)
        train_tags.append(str(train_tag))

def input_data_test(test_file, test_tag):
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in range(2):
            tks = f.readline().strip()
        tks = re.sub(r'[^\u4e00-\u9fa5]', '', tks)
        test_words.append(tks)
        test_tags.append(str(test_tag))
        

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
    print('precision:{0:.4f}'.format(m_precision))
    print('recall:{0:0.4f}'.format(m_recall))
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
        pre = clf.predict(test_data)
        # print("目标结果：", test_tags)
        # print("预测结果：", re)
        # 比较目标标签与预测标签 计算准确率和召回率
        evaluate_bayes_test_alpha(np.asarray(test_tags), pre)
    # print(alphas)
    # print(precision)
    # print(recall)
    print(np.amax(precision))
    # print(np.argmax(precision))
    print('alpha:'+ str(alphas[int(np.argmax(precision))]))

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


def test_dataset(test_root_path, test_tag):
    test_paths = os.listdir(test_root_path)
    for r_path in test_paths:
        a_path = test_root_path + '/' + r_path
        input_data_test(a_path, test_tag)

def train_dataset(train_root_path, train_tag):
    train_paths = os.listdir(train_root_path)
    for r_path in train_paths:
        a_path = train_root_path +'/'+r_path
        input_data_train(a_path, train_tag)

def train_paths(train_path):
    paths_next = os.listdir(train_path)
    for path_next in paths_next:
        train_tag = path_next
        train_root_path = train_path+'/' + train_tag
        train_dataset(train_root_path,train_tag)

def test_paths(test_path):
    paths_next = os.listdir(test_path)
    for path_next in paths_next:
        test_tag = path_next
        test_root_path = test_path+'/' + test_tag
        test_dataset(test_root_path,test_tag)

if __name__ == '__main__':
    # train_path = 'D:/dataset_train'
    # test_path = 'D:/dataset_test'

    train_path = 'D:/1'
    test_path = 'D:/2'

    # train_path = '/Volumes/Data/dataset_train'
    # test_path = '//Volumes/Data/dataset_test'

    train_paths(train_path)
    test_paths(test_path)
    # print(train_words)
    # print(test_words)

    #
    # # bayes + this 0.7874
    # # svc + this 0.601
    # train_data, test_data = covectorize(train_words, test_words)

    # bayes + this 0.7892
    # svc + this 0.0010
    train_data, test_data = tfvectorize(train_words, test_words)

    # # bayes + this error : ValueError: Negative values in data passed to MultinomialNB (input X)
    # # svc +this error: float() argument must be a string or a number, not 'HashingVectorizer'
    # # train_data, test_data = vectorize(train_words, test_words)

    #print(train_data)
    #print(test_data)
    #
    # svc_test(train_data,train_tags)
    #
    bayes_test(train_data,train_tags, alpha=0.19)
    #
    # 对不同alpha值进行测试 选取准确率达到最大时的alpha值 0.19 0.7531
    # bayes_test_alpha(train_data,train_tags,test_tags)

