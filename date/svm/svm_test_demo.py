from sklearn import svm
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=196)

x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 2, 3]
clf = svm.SVC()
clf.fit(x, y)
print(clf.score([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 3]))

result = clf.predict([[0, 1]])
print(result)