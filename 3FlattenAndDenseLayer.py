import numpy as np
import tensorflow as tf
from keras.layers import Flatten, Dense
from termcolor import colored

def p_color(input_string, input_object): # shape를 표시해주는데 다른 색으로 표시해줌
    print(colored(input_string + ':', 'magenta'))
    print(colored('shape: ', 'cyan'), input_object.shape, '\n')

feature_map = tf.random.normal(mean=0, stddev=1, shape=(32, 11, 11, 128))

flatten = Flatten() # 단순히 펴주기만 한다
flattened = flatten(feature_map)

p_color('feature map', feature_map)
p_color('after flatten', flattened)

print(11*11*128) # 15488 나옴 (flatten layer의 원소 개수)

test_feature = tf.random.normal(mean=0, stddev=1, shape=(32, 15488))
dense = Dense(units=64, activation='relu')
densed = dense(test_feature)

p_color('test_feature map', test_feature)
p_color('after dense', densed)

print(dense.get_weights()[0].shape)