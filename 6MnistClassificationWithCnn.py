import os

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense

from keras.losses import SparseCategoricalCrossentropy
from keras import optimizers # optimizers.SGD
from keras.metrics import Mean, SparseCategoricalAccuracy

