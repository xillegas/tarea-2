import tensorflow as tf
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense

def kminst_model(train_images, train_labels, batch_size, deep):
    model = tf.keras.Sequential()
    model.add(Conv2D(filters=64, kernel_size= 2, padding='same', activation='relu', input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))
    for _ in range(deep):
        model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )
    eval_model = model.evaluate(test_images, test_labels, verbose=0)
    return (eval_model, model.fit(train_images, train_labels, batch_size=batch_size, epochs=10))
