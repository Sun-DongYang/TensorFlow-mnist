from tensorflow.examples.tutorials.mnist import input_data

# 载入mnist数据集
mnist = input_data.read_data_sets(r"E:\PythonSpace\mnist_data\\")

# 输出训练集的大小
print("Training data size: ", mnist.train.num_examples)
# 输出验证集的大小
print("Val data size: ", mnist.validation.num_examples)
# 输出测试集的大小
print("Test data size: ", mnist.test.num_examples)

# 输出训练集图片的信息
print("Example training data: ", mnist.train.images[0])

# 输出训练集图片的标签,这里显示的是单标签类型但是书上显示的是二进制的向量
print("Example training data label: ", mnist.train.labels[0])

batch_size = 100
#从训练集选取batch_size个训练数据
xs,ys = mnist.train.next_batch(batch_size)
# 输出图片信息shape
print("X shape: ", xs.shape)
# 输出图片的标签shape
print("Y shape: ", ys.shape)
