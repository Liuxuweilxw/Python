import mnist_loader
import network
import network2

# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# # 100个隐藏神经元 输入层有784个神经元 输出层有10个神经元
# net = network.Network([784, 100, 10])
# # 迭代期为10 小批量数据为10 学习速率为3.0
# net.SGD(training_data, 10, 10, 3.0, test_data=test_data)
#
# #计算最后得到的 偏置值和权重作用下的神经网络识别准确率 还可以输出最后一次的权重和偏置值 作为最终训练的结果
# a = net.evaluate(test_data)
# print(a/len(test_data))


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network2.Network([784, 100, 10],cost=network2.CrossEntropyCost)
# net.large_weight_initializer()
# net.SGD(training_data,400,10,0.5,evaluation_data=test_data,lmbda=5.0,
#         monitor_evaluation_cost=True, monitor_evaluation_accuracy=True,
#         monitor_training_cost=True, monitor_training_accuracy=True)

# net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, lmbda = 5.0, monitor_evaluation_accuracy=True, monitor_training_accuracy=True)
# net.SGD(training_data, 30, 10, 0.5,lmbda=5.0, evaluation_data=validation_data,monitor_evaluation_accuracy=True)
net.SGD(training_data, 10, 10, 0.1,lmbda=5.0, evaluation_data=validation_data,monitor_evaluation_accuracy=True)
net.save("C:/Users/liuxuwei/Desktop/out/a.txt")