import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model


mnist = tf.kears.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train /= 255
x_test /= 255

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Datset.from_tensor_slices(
	(x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Datset.from_tensor_slices((x_test, y_test)).batch(32)


class MyModel(Model):
	def __init__(self):
		super().__init__()
		self.conv1 = Conv2D(32, 3, activation='relu')
		self.flatten = Flatten()
		self.d1 = Dense(128, activation='relu')
		self.d2 = Dense(10, activation='softmax')