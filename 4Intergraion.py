import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense
from termcolor import colored

def p_color(input_string, input_object): # shape를 표시해주는데 다른 색으로 표시해줌
    print(colored(input_string + ':', 'magenta'))
    print(colored('shape: ', 'cyan'), input_object.shape, '\n')

test_image = tf.random.normal(mean=0, stddev=1, shape=(32, 50, 50, 3))

conv = Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu')
conv_pool = MaxPooling2D(pool_size=2, strides=2)
flatten = Flatten()
dense = Dense(units=10, activation='softmax') # classification으로 softmax를 활성함수로 사용

p_color('inputs', test_image)
x = conv(test_image)
p_color('after conv', x)
x = conv_pool(x)
p_color('after pool', x)
x = flatten(x)
p_color('after flatten', x)
x = dense(x)
p_color('after dense', x)