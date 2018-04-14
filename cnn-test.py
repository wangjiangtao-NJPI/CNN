# @Time    : 2018/4/11 10:49
# @Author  : 王江涛
# @Git     : wangjiangtao-NJPI
# @File    : cnn-test.py
# @Software: PyCharm

import numpy as np

np.random.seed(1337)  # for reproducibility，生成同一个随机数
from keras.datasets import mnist  # MNIST 数据集来自美国国家标准与技术研究所, 自 250 个不同人手写的数字构成
from keras.models import Sequential  # 序贯模型是多个网络层的线性堆叠，也就是“一条路走到黑”。
from keras.layers import Dense, Dropout, Activation, Flatten  # dense(就是隐藏层)
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils  # 格式转换函数，将标签转换为binary class matrices二进制矩阵形式
from keras import backend as K  # backend框架

# 全局变量
batch_size = 128  # 每处理128个样本进行一次梯度更新
nb_classes = 10
epochs = 12
# input image dimensions图片维数
img_rows, img_cols = 28, 28
# number of convolutional filters to use卷积核数量
nb_filters = 32
# size of pooling area for max pooling最大池化的池化区域尺寸
pool_size = (2, 2)
# convolution kernel size卷积核尺寸
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets装载数据
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 根据不同的backend定下不同的格式
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# 将输入数据的数据类型转换为float32（32位浮点数）
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# 归一化
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# 转换为one_hot类型（一个长度为n的数组，只有一个元素是1.0，其他元素是0.0）
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# 构建模型
model = Sequential()  # Keras中的模型可以有两种——序贯和通过API函数，序贯最常用
""" 
添加一个2D卷积层来处理2D MNIST输入的图像；
传递给Conv2D（）函数的第一个参数是输出通道的数量，设置为32个输出通道；
下一个输入是kernel_size，我们选择了一个1×1移动窗口；
填充(padding)=same,是指如果采样到了边界，继续采样，后面补0。
keras中数据是以张量的形式表示的，张量的形状称之为shape，input_shape就是指输入张量的shape。
"""
model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]),
                        padding='same',
                        input_shape=input_shape))  # 卷积层1
model.add(Activation('relu'))  # 激活层，激活函数是整流线性单元；
model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1])))  # 卷积层2
model.add(Activation('relu'))  # 激活层
model.add(MaxPooling2D(pool_size=pool_size))  # 池化层
model.add(Dropout(0.25))  # 神经元随机失活
model.add(Flatten())  # 拉成一维数据
model.add(Dense(128))  # 全连接层1
model.add(Activation('relu'))  # 激活层
model.add(Dropout(0.5))  # 随机失活
model.add(Dense(nb_classes))  # 全连接层2
model.add(Activation('softmax'))  # Softmax评分

# 编译模型
# 使用softmax作为输出层的激活函数时，损失函数采用categorical_crossentropy绝对交叉熵；
# optimizer优化器采用adadelta自适应学习率；
# metrics 性能评估指标列表，包含评估模型在训练和测试时的网络性能的指标，对分类问题，一般设置metrics=['accuracy']；
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
# 训练模型
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(X_test, Y_test))
# 评估模型
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
