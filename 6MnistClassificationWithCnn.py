import os

import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Activation

from keras.losses import SparseCategoricalCrossentropy # 사용할 Loss function
from keras import optimizers # optimizers.SGD
from keras.metrics import Mean, SparseCategoricalAccuracy

from utils.learning_env_setting1 import dir_setting
from utils.dataset_utils import load_processing_mnist
from utils.cp_utils import save_metrics_model, metric_visualizer
from utils.basic_utils import resetter, training_reporter

# === Hyperparameter Setting start ===
CONTINUE_FLAG = False
dir_name = 'exp1'
start_epoch = 0

train_ratio = 0.8
train_batch_size, test_batch_size = 32, 128

epochs = 30
learning_rate = 0.01
# === Hyperparameter Setting end ===

path_dict = dir_setting(dir_name, CONTINUE_FLAG)
train_ds, validation_ds, test_ds = load_processing_mnist(train_ratio, train_batch_size, test_batch_size)
losses_accs = {
    'train_losses':[],
    'train_accs':[],
    'validation_losses':[],
    'validation_accs':[]
}

# === Model Implementation start ===
# model = Sequential()
# model.add(Conv2D(filters=8, kernel_size=(5, 5), padding='same', activation='relu'))
# model.add(MaxPooling2D())
class MnistClassifier(Model):
    def __init__(self):
        super(MnistClassifier, self).__init__()

        self.conv1 = Conv2D(filters=8, kernel_size=(5, 5), padding='same', activation='relu')
        # input 이미지는 28 x 28이다. 따라서 ((28 - 5)/1) + 1 => 24가 output의 크기이다. but same padding(stride없음)이므로 28이 output의 크기이다. [ ((n - k)/s) + 1 ]
        self.conv1_pool = MaxPooling2D(pool_size=2, strides=2)
        self.conv2 = Conv2D(filters=32, kernel_size=5, padding='same', activation='relu')
        self.conv2_pool = MaxPooling2D(pool_size=2, strides=2)

        self.flatten = Flatten()
        self.dense1 = Dense(units=64, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv1_pool(x)
        x = self.conv2(x)
        x = self.conv2_pool(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x
# === Model Implementation end ===

loss_objects = SparseCategoricalCrossentropy # 사용할 Loss function
optimizer = optimizers.SGD(learning_rate=learning_rate)

train_loss = Mean()
train_acc = SparseCategoricalAccuracy()

validation_loss = Mean()
validation_acc = SparseCategoricalAccuracy()

test_loss = Mean()
test_acc = SparseCategoricalAccuracy()

model = MnistClassifier()

@tf.function
def trainer():
    global train_ds, loss_objects, optimizer
    global train_loss, train_acc

    for images, labels in train_ds:
        with tf.GradientTape() as tape: # forward propagation 진행, (GradientTape =>자동 미분을 통해 gradient를 계산해낸다. 모든 연산을 테이프에 저장함)
            predictions = model(images)
            loss = loss_objects(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_acc(labels, predictions)

def validation():
    global validation_ds, loss_objects
    global validation_loss, validation_acc

    for images, labels in validation_ds:
        predictions = model(images)
        loss = loss_objects(labels, predictions)

        validation_loss(loss)
        validation_acc(labels, predictions)

def test():
    global test_ds, loss_objects
    global test_loss, test_acc

    for images, labels in test_ds:
        predictions = model(images)
        loss = loss_objects(labels, predictions)

        test_loss(loss)
        test_acc(labels, predictions)

for epoch in range(start_epoch, epochs):
    trainer()
    validation()

    training_reporter(epoch, losses_accs,
                      train_loss=train_loss, train_acc=train_acc,
                      validation_loss=validation_loss, validation_acc=validation_acc)
    save_metrics_model(epoch, model, losses_accs, path_dict)
    metric_visualizer(losses_accs, path_dict['cp_path'])
    resetter(train_loss, train_acc, validation_loss, validation_acc) # 초기화


