import numpy as np
import tensorflow as tf
from keras.layers import Conv2D

tf.random.set_seed(0)

test_image = tf.random.normal(mean = 0, stddev = 1, shape = (1, 7, 7, 1)) # shape = (배치사이즈, 행, 열, 차원)

conv = Conv2D(filters=1, kernel_size=3, strides=1, padding='valid') # 필터(커널)
conved = conv(test_image)

print(conved.numpy().shape)
print(conved.numpy().squeeze().shape) # 쓸모없는 정보를 날려줌
print('conved tf:\n', conved.numpy().squeeze(), '\n')

test_image = test_image.numpy().squeeze()
w, b = conv.get_weights() # 필터의 가중치와 bias
print(w.shape) # (3, 3, 1, 1) 나옴 (행, 열, 인풋채널갯수, 아웃풋채널갯수)
w = w.squeeze()
print(w.shape)

valid_idx = int((w.shape[0] - 1) / 2) # valid padding일 때 슬라이딩의 스타트 포인트((커널사이즈 - 1) / 2)
print(valid_idx)
H, W = test_image.shape # input 이미지의 shape
print(H, W)
H_conved, W_conved = 5, 5 # 컨볼루션 연산 후 아웃풋의 shape, [(H_in + 2p - k)/s] + 1 == [(7 + 2 * 0 - 3)/1] + 1 == 5
conved_manual = np.zeros(shape=(H_conved, W_conved))
print(conved_manual)
for r_idx in range(valid_idx, H - valid_idx - 1): # 각각 슬라이딩의 스타팅 포인트
    for c_idx in range(valid_idx, W - valid_idx - 1):
        receptive_field = test_image[r_idx - valid_idx:r_idx + valid_idx + 1, c_idx - valid_idx:c_idx + valid_idx + 1]
        conved_tmp = receptive_field * w
        conved_tmp = np.sum(conved_tmp) + b
        conved_manual[r_idx - valid_idx:r_idx + valid_idx + 1, c_idx - valid_idx:c_idx + valid_idx + 1] = conved_tmp
print('conved manual:\n', conved_manual, '\n')
print('conved tf:\n', conved.numpy().squeeze(), '\n') # conved manual과 conved by tf의 비교 (동일한 것을 확인할 수 있다.)








