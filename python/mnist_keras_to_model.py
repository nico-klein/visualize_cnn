#  3 epochs: test loss:  0.09  test acc:  0.973
#  5 epochs: test loss:  0.078  test acc:  0.977
# 10 epochs: test loss:  0.071  test acc:  0.981

import tensorflow as tf
import tensorflowjs as tfjs
import os


def learn_and_save_model(model, x_train, y_train, x_test, y_test, epochs, pathName):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # train
    model.fit(x=x_train, y=y_train, epochs=epochs)

    # store model for tensorflow js
    tfjs.converters.save_keras_model(model, os.path.abspath(os.path.dirname(__file__) + pathName))

    model.summary()

    # evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("test loss: ", round(test_loss, 3), " test acc: ", round(test_acc, 3))

######
# main
######


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255

# input image dimensions
img_rows, img_cols = 28, 28
hidden_nodes = 256
output_nodes = 10


# without conv layer
model_simple = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(units=hidden_nodes, activation=tf.nn.relu),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(units=output_nodes, activation=tf.nn.softmax)

])

learn_and_save_model(model=model_simple, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                     epochs=10, pathName="/../model_simple")

# with conv
# add depth for Conv2D
print("tf.keras.backend.image_data_format():" + tf.keras.backend.image_data_format())
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)  # channes_last

model_cnn = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', activation=tf.nn.relu, input_shape=input_shape),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(filters=64, kernel_size=5, padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=hidden_nodes, activation=tf.nn.relu),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(units=output_nodes, activation=tf.nn.softmax)
])

learn_and_save_model(model=model_cnn, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                     epochs=10, pathName="/../model_cnn")


# small_cnn
model_small_cnn = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=10, kernel_size=5, padding='same', activation=tf.nn.relu, input_shape=input_shape),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=hidden_nodes, activation=tf.nn.relu),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(units=output_nodes, activation=tf.nn.softmax)
])

learn_and_save_model(model=model_small_cnn, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                     epochs=10, pathName="/../model_small_cnn")
