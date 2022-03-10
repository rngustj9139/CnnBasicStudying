import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from termcolor import colored

def p_color(input_string, input_object): # shape를 표시해주는데 다른 색으로 표시해줌
    print(colored(input_string + ':', 'magenta'))
    print(colored('shape: ', 'cyan'), input_object.shape, '\n')

tf.random.set_seed(0)
test_image = tf.random.normal(mean=0, stddev=1, shape=(1, 50, 50, 1)) # 그레이 스케일이라서 채널이 1개
conv = Conv2D(filters=8, kernel_size=3, padding='valid', strides=1)
conved = conv(test_image)

p_color('input', test_image)
p_color('after convolution calculation', conved)
p_color('conv weight', conv.get_weights()[0])
p_color('conv bias', conv.get_weights()[1])

print("######################################################")

test_image = tf.random.normal(mean=0, stddev=1, shape=(1, 4, 4, 1))
maxpool = MaxPooling2D(pool_size=2, strides=2)
maxpooled = maxpool(test_image)

p_color('input', test_image)
p_color('after maxpool', maxpooled)

print("######################################################")
test_image = tf.random.normal(mean=0, stddev=1, shape=(1, 4, 4, 1))
averagepool = AveragePooling2D(pool_size=2, strides=2)
averagepooled = averagepool(test_image)

print("######################################################")

test_image = tf.random.uniform(minval=0, maxval=10, shape=(1, 4, 4, 1), dtype=tf.int32)
maxpool = MaxPooling2D(pool_size=2, strides=1) # overlap 된다.
maxpooled = maxpool(test_image)

p_color('input', test_image)
p_color('after maxpool', maxpooled)


p_color('input', test_image)
p_color('after averagepolling', averagepooled)

