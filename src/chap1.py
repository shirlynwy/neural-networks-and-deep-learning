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
import network
net = network.Network([784, 30, 10])
net.SGD(training_data, 60, 10, 3.0, test_data=test_data)

# net = network.Network([784, 100, 10])
# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)