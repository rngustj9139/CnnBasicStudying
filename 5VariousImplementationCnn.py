import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Layer
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.layers import Flatten, Dense

######################################### Sequential

model = Sequential()

# feature extractor
model.add(Conv2D(filters=8, kernel_size=5, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Conv2D(filters=32, kernel_size=5, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=2, strides=2))

# classifier
model.add(Flatten())
model.add(Dense(units=64, activation="relu"))
model.add(Dense(units=10, activation="softmax"))

model.build(input_shape=(None, 28, 28, 1))
model.summary()

#########################################

######################################### Model Sub-classing (커넥션이 조금 더 복잡할 경우 사용한다.)

class CNN_Model(Model):
    def __init__(self):
        super(CNN_Model, self).__init__()

        # feature extractor
        self.conv1 = Conv2D(filters=8, kernel_size=5, padding="same", activation="relu")
        self.conv1_pool = MaxPooling2D(pool_size=2, strides=2)
        self.conv2 = Conv2D(filters=32, kernel_size=5, padding="same", activation="relu")
        self.conv2_pool = MaxPooling2D(pool_size=2, strides=2)

        # classifier
        self.flatten = Flatten()
        self.dense1 = Dense(units=64, activation="relu")
        self.dense2 = Dense(units=10, activation="softmax")

    def call(self, x):
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        x = self.conv1_pool(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        x = self.conv2_pool(x)
        print(x.shape)

        x = self.flatten(x)
        print(x.shape)
        x = self.dense1(x)
        print(x.shape)
        x = self.dense2(x)
        print(x.shape)

        return x

model = CNN_Model()
model.build(input_shape=(None, 28, 28, 1))
model.summary()

#########################################

######################################### trainable parameters

print(model.layers)
print(len(model.layers))

for layer in model.layers:
    if len(layer.get_weights()) != 0: # pooling과 flatten의 경우에만 trainable parameter가 0개이다.
        w, b = layer.get_weights()
        print(w.shape, b.shape, '\n')


#########################################

######################################### sequential + model sub-classing (하이브리드)
class CNN_Model(Model):
    def __init__(self):
        super(CNN_Model, self).__init__()

        # feature extractor
        self.fe = Sequential(name="feature extractor")
        self.fe.add(Conv2D(filters=8, kernel_size=5, padding="same", activation="relu"))
        self.fe.add(MaxPooling2D(pool_size=2, strides=2))
        self.fe.add(Conv2D(filters=32, kernel_size=5, padding="same", activation="relu"))
        self.fe.add(MaxPooling2D(pool_size=2, strides=2))

        # classifier
        self.classifier = Sequential(name="classifier")
        self.classifier.add(Flatten())
        self.classifier.add(Dense(units=64, activation="relu"))
        self.classifier.add(Dense(units=10, activation="softmax"))

    def call(self, x):
        x = self.fe(x)
        x = self.classifier(x)

        return x

model = CNN_Model()
model.build(input_shape=(None, 28, 28, 1))
model.summary()

#########################################

######################################### layer sub-classing

class ConvLayer(Layer):
    def __init__(self, filters):
        super(ConvLayer, self).__init__()

        self.conv = Conv2D(filters=filters, kernel_size=5, padding="same")
        self.conv_act = Activation("relu")
        self.conv_pool = MaxPooling2D(pool_size=2, strides=2)

    def call(self, x):
        x = self.conv(x)
        x = self.conv_act(x)
        x = self.conv_pool(x)

        return x

class CNN_Model(Model):
    def __init__(self):
        super(CNN_Model, self).__init__()

        self.conv1 = ConvLayer(8)
        self.conv2 = ConvLayer(16)
        self.conv3 = ConvLayer(32)

        # classifier
        self.flatten = Flatten()
        self.dense1 = Dense(units=64, activation="relu")
        self.dense2 = Dense(units=10, activation="softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x

model = CNN_Model()
model.build(input_shape=(None, 28, 28, 1))
model.summary()

#########################################