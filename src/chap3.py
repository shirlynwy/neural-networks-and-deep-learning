import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# print('size of test_data');
# print(len(test_data))
# # print(test_data[1])
# # print('length of 1 test data')
# print(test_data[1][1])
# print(type(test_data[1]))
# below 3 lines changed from py2 to py3
# test_data = list(test_data)
# training_data = list(training_data)
# test_data = list(test_data)
import network2
net = network2.Network([784, 100, 10],cost=network2.CrossEntropyCost)
net.SGD(training_data, 60, 10, 0.1,
         lmbda = 5,
         evaluation_data=test_data,
         monitor_evaluation_accuracy=True,
         monitor_evaluation_cost=True,
         monitor_training_accuracy=True,
         monitor_training_cost=True)
