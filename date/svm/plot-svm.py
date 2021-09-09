from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
x = np.r_[np.random.randn(20,2)-[2,2], np.random.randn(20,2)+[2,2]]
# x是40行2列的数组 表示40个点
# print(x)
# np.c_ 可以用于拼接数组，要求行数相同 拼接后的数组水平方向排列
# np.r_ 可以用于拼接数组，要求列数相同 拼接后的数组纵向排列
y = [0]*20 + [1]*20
# print(y)
# 生成一个一行四十列的数组 前二十个元素为0 后二十个元素为1
# x 中的每个点都对应于y 中的一个值表示第几类
clf = svm.SVC(kernel='linear')
clf.fit(x, y)
# 可以找出不同类之间的边界点 并用 support_vectors_ 得到 与边界点相切的直线截距

w = clf.coef_[0]
# w[0]y+w[1]x+b1=0
# 斜率a = -w[0]/w[1]
# 截距b = -b1/w[1]
# b1 = clf.intercept_[0]/w[1]
# print(w)
a = -w[0]/w[1]
# print(a)

xx = np.linspace(-5, 5)
# print(xx)
# 水平方向未知数范围

yy = a*xx-clf.intercept_[0]/w[1]
# 得到不同的xx点对应的yy值

# 画出与点相切的线
b = clf.support_vectors_[0]
yy_down = a*xx+(b[1]-a*b[0])
# yy_down 为 以与方程同斜率不同截距 xx 对应的值 表示下面的超平面边界
b = clf.support_vectors_[1]
yy_up = a*xx+(b[1]-a*b[0])
# yy_up 为 以与方程同斜率不同截距 xx 对应的值 表示上面的超平面边界

print("W:", w)
# w 为系数矩阵
print("a:", a)
# a 为直线方程的系数

print("support_vectors_:", clf.support_vectors_)
print("clf.coef_:", clf.coef_)
# clf.coef_ 即为系数矩阵
# support_vectors_

plt.figure(figsize=(8,4))
plt.plot(xx,yy)
plt.plot(xx,yy_down)
plt.plot(xx,yy_up)
plt.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],s=80)
plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.Paired)

plt.axis('tight')
plt.show()



