import tensorflow as tf
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense


def kminst_model_conv(convolution_deep):
    model = tf.keras.Sequential()
    model.add(
        Conv2D(
            filters=32,
            kernel_size=2,
            padding="same",
            activation="relu",
            input_shape=(28, 28, 1),
        )
    )
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))
    for i in range(convolution_deep):
        model.add(
            Conv2D(
                filters=64 * (2**i), 
                kernel_size=2,
                padding="same",
                activation="relu"
            )
        )
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))
    return model


def kminst_model_no_drop():
    model = tf.keras.Sequential()
    model.add(
        Conv2D(
            filters=64,
            kernel_size=2,
            padding="same",
            activation="relu",
            input_shape=(28, 28, 1),
        )
    )
    model.add(MaxPooling2D(pool_size=2))
    model.add(
        Conv2D(
            filters=128, kernel_size=2, padding="same", activation="relu"
        )
    )
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    return model
