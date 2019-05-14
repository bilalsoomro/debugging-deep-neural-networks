import keras
import numpy as np
import os
from keras.utils import to_categorical
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

DATA_PATH = 'dataset/training/'

def get_x_y(FEATURES_PATH):
    labels = os.listdir(DATA_PATH)

    # Getting first arrays
    X = np.load(FEATURES_PATH + labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in tqdm(enumerate(labels[1:])):
        x = np.load(FEATURES_PATH + label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value=(i + 1)))

    return X, y

def create_model():
    input_shape = Input(shape=(90, 80, 1))

    #Layers
    c_conv_1 = Conv2D(8, kernel_size=(2, 2), activation='relu')
    c_max_pool_1 = MaxPooling2D(pool_size=(2, 2))
    c_drop_1 = Dropout(0.2)
    c_conv_2 = Conv2D(16, kernel_size=(2, 2), activation='relu')
    c_max_pool_2 = MaxPooling2D(pool_size=(2, 2))
    c_drop_2 = Dropout(0.2)
    c_conv_3 = Conv2D(32, kernel_size=(2, 2), activation='relu')
    c_max_pool_3 = MaxPooling2D(pool_size=(2, 2))
    c_drop_3 = Dropout(0.2)
    c_flatten = Flatten()
    c_dense_1 = Dense(512, activation='relu')
    c_drop_4 = Dropout(0.2)
    c_dense_2 = Dense(256, activation='relu')
    c_drop_5 = Dropout(0.2)
    c_dense_3 = Dense(128, activation='relu')
    c_drop_6 = Dropout(0.2)
    c_dense_output = Dense(35, activation='softmax')

    classifier = c_conv_1(input_shape)
    classifier = c_max_pool_1(classifier)
    classifier = c_drop_1(classifier)
    classifier = c_conv_2(classifier)
    classifier = c_max_pool_2(classifier)
    classifier = c_drop_2(classifier)
    classifier = c_conv_3(classifier)
    classifier = c_max_pool_3(classifier)
    classifier = c_drop_3(classifier)
    classifier = c_flatten(classifier)

    classifier = c_dense_1(classifier)
    classifier = c_drop_4(classifier)
    classifier = c_dense_2(classifier)
    classifier = c_drop_5(classifier)
    classifier = c_dense_3(classifier)
    classifier = c_drop_6(classifier)
    classifier = c_dense_output(classifier)

    model = Model(inputs=input_shape, outputs=classifier)
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(),
                metrics=['accuracy'])

    return model

def main():
    # load up training, validation and testing data
    # X_train, y_train = get_x_y('features/training/')
    # X_valid, y_valid = get_x_y('features/validation/')
    # X_test, y_test = get_x_y('features/testing/')

    # Save the features after combining as that steps takes time
    # np.save('features/full_train_x.npy', X_train)
    # np.save('features/full_train_y.npy', y_train)
    # np.save('features/full_valid_x.npy', X_valid)
    # np.save('features/full_valid_y.npy', y_valid)
    # np.save('features/full_test_x.npy', X_test)
    # np.save('features/full_test_y.npy', y_test)

    # Loading the features
    X_train = np.load('features/full_train_x.npy')
    y_train = np.load('features/full_train_y.npy')
    X_valid = np.load('features/full_valid_x.npy')
    y_valid = np.load('features/full_valid_y.npy')
    X_test = np.load('features/full_test_x.npy')
    y_test = np.load('features/full_test_y.npy')


    # shuffle the data
    X_train, y_train = shuffle(X_train, y_train, random_state=0)
    X_valid, y_valid = shuffle(X_train, y_train, random_state=0)
    X_test, y_test = shuffle(X_train, y_train, random_state=0)

    X_train = X_train.reshape(X_train.shape[0], 90, 80, 1)
    X_valid = X_valid.reshape(X_valid.shape[0], 90, 80, 1)
    X_test = X_test.reshape(X_test.shape[0], 90, 80, 1)

    # Change labels to one-hot encoding
    y_train_hot = to_categorical(y_train)
    y_valid_hot = to_categorical(y_valid)
    y_test_hot = to_categorical(y_test)

    model = create_model()
    model.summary()

    if not os.path.exists('models'):
        os.makedirs('models')

    history1 = model.fit(X_train, y_train_hot, epochs=50, verbose=1, validation_data=(X_valid, y_valid_hot))
    model.save('models/speech_classifier_model.h5')
    model.save_weights('models/speech_classifier_weights.h5')

    if not os.path.exists('figures'):
        os.makedirs('figures')

    plt.plot(history1.history['acc'])
    plt.plot(history1.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig('figures/model_training_curve.png', bbox_inches='tight')
    plt.clf()

    plt.plot(history1.history['loss'])
    plt.plot(history1.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig('figures/model_loss_curve.png', bbox_inches='tight')


    acc = model.evaluate(X_test, y_test_hot)
    print('Model acc: ', acc)

if __name__ == '__main__':
    main()