from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Dropout
from keras.models import Model, load_model
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
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

def create_models():
    input_shape = Input(shape=(90, 80, 1))

    # Encoder Layers
    conv_1 = Conv2D(16, (2, 2), activation='relu', padding='same')
    max_pool_1 = MaxPooling2D((3, 2), padding='same')
    conv_2 = Conv2D(32, (2, 2), activation='relu', padding='same')
    max_pool_2 = MaxPooling2D((3, 2), padding='same')
    conv_3 = Conv2D(64, (2, 2), activation='relu', padding='same')
    max_pool_3 = MaxPooling2D((2, 2), padding='same')
    conv_4 = Conv2D(128, (2, 2), activation='relu', padding='same')

    encoded = Flatten(name='encoder')

    # Bottleneck
    dense_1 = Dense(256, name='bottleneck')
    dense_2 = Dense(6400)
    reshape = Reshape((5, 10, 128))

    # Decoder Layers
    conv_5 = Conv2D(128, (2, 2), activation='relu', padding='same')
    up_samp_1 = UpSampling2D((2, 2))
    conv_6 = Conv2D(64, (2, 2), activation='relu', padding='same')
    up_samp_2 = UpSampling2D((3, 2))
    conv_7 = Conv2D(32, (2, 2), activation='relu', padding='same')
    up_samp_3 = UpSampling2D((3, 2))

    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='decoder')

    ####################################################################################################
    #-----------------------------------------Full Autoencoder-----------------------------------------#
    ####################################################################################################

    autoencoder = conv_1(input_shape)
    autoencoder = max_pool_1(autoencoder)
    autoencoder = conv_2(autoencoder)
    autoencoder = max_pool_2(autoencoder)
    autoencoder = conv_3(autoencoder)
    autoencoder = max_pool_3(autoencoder)
    autoencoder = conv_4(autoencoder)
    autoencoder = encoded(autoencoder)
    autoencoder = dense_1(autoencoder)
    autoencoder = dense_2(autoencoder)
    autoencoder = reshape(autoencoder)
    autoencoder = conv_5(autoencoder)
    autoencoder = up_samp_1(autoencoder)
    autoencoder = conv_6(autoencoder)
    autoencoder = up_samp_2(autoencoder)
    autoencoder = conv_7(autoencoder)
    autoencoder = up_samp_3(autoencoder)
    autoencoder = decoded(autoencoder)

    autoencoder = Model(inputs=input_shape, outputs=autoencoder)

    ####################################################################################################
    #------------------------------------------Encoder-------------------------------------------------#
    ####################################################################################################

    encoder = conv_1(input_shape)
    encoder = max_pool_1(encoder)
    encoder = conv_2(encoder)
    encoder = max_pool_2(encoder)
    encoder = conv_3(encoder)
    encoder = max_pool_3(encoder)
    encoder = conv_4(encoder)
    encoder = encoded(encoder)
    encoder = dense_1(encoder)

    encoder_model = Model(inputs=input_shape, outputs=encoder)

    ####################################################################################################
    #------------------------------------------Decoder------------------------------------------------#
    ####################################################################################################

    bottleneck_input_shape = Input(shape=(256,))
    decoder_model = dense_2(bottleneck_input_shape)
    decoder_model = reshape(decoder_model)
    decoder_model = conv_5(decoder_model)
    decoder_model = up_samp_1(decoder_model)
    decoder_model = conv_6(decoder_model)
    decoder_model = up_samp_2(decoder_model)
    decoder_model = conv_7(decoder_model)
    decoder_model = up_samp_3(decoder_model)
    decoder_model = decoded(decoder_model)

    decoder_model = Model(inputs=bottleneck_input_shape, outputs=decoder_model)

    return encoder_model, decoder_model, autoencoder

def plotDecodedSamples(decoder_model):
    for i in range(3):
        # plot bottleneck code
        
        mu, sigma = 0, 0.1 # mean and standard deviation
        s = np.random.normal(mu, sigma, 256)

        np.save('figures/autoencoder_' + str(i) + '_before.npy', s)
        s = np.expand_dims(s, 0)
        decoded_noise = decoder_model.predict(s)

        # plot decoded utterance

        ax2 = plt.subplot(1, 3, i + 1)
        ax2.set_title("Sample " + str(i+1))
        ax2.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        right=False,
        left=False,
        labelbottom=False,
        labelright=False,
        labelleft=False,
        labeltop=False) # labels along the bottom edge are off
        
        visual = np.flipud(decoded_noise.squeeze().T)
        plt.imshow(visual, origin='lower')
        np.save('figures/autoencoder_' + str(i) + '_after.npy', decoded_noise.squeeze())

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig('figures/test_decoded_samples.png', bbox_inches='tight', dpi=1000)

def plotSingleDecodedNoiseSample(decoder_model):
    mu, sigma = 0, 0.1 # mean and standard deviation
    s = np.random.normal(mu, sigma, 256)
    s = np.expand_dims(s, 0)

    decoded_noise = decoder_model.predict(s)

    plt.figure(figsize=(18, 4))
    ax = plt.subplot(2, 2, 1)
    visual = np.flipud(decoded_noise.reshape(90, 80).T)
    plt.imshow(visual, origin="lower")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    plt.savefig('figures/decoded_gaussian_noise.png', bbox_inches='tight')

def plot_encoded_decoded_samples(encoder, decoder, x_test):
    num_speeches = 10
    np.random.seed(0)
    random_test_utterances = np.random.randint(x_test.shape[0], size=num_speeches)

    encoded_utterances = encoder.predict(x_test)
    decoded_utterances = decoder.predict(encoded_utterances)

    for i, utterance_idx in enumerate(random_test_utterances):
        # plot original image
        ax = plt.subplot(3, num_speeches, i + 1)
        visual = np.flipud(x_test[utterance_idx].reshape(90, 80).T)
        plt.imshow(visual)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # plot encoded image
        ax = plt.subplot(3, num_speeches, num_speeches + i + 1)
        plt.imshow(encoded_utterances[utterance_idx].reshape(16, 16))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # plot reconstructed image
        ax = plt.subplot(3, num_speeches, 2*num_speeches + i + 1)
        visual = np.flipud(decoded_utterances[utterance_idx].reshape(90, 80).T)
        plt.imshow(visual)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    plt.savefig('figures/autoencoder_results.png', bbox_inches='tight')

def main():
    # load up training, validation and testing data
    # X_train, _ = get_x_y('features/training/')
    # X_valid, _ = get_x_y('features/validation/')
    # X_test, _ = get_x_y('features/testing/')

    # Save the features after combining as that steps takes time
    # np.save('features/full_train_x.npy', X_train)
    # np.save('features/full_valid_x.npy', X_valid)
    # np.save('features/full_test_x.npy', X_test)

    # Loading the features
    X_train = np.load('features/full_train_x.npy')
    X_valid = np.load('features/full_valid_x.npy')
    X_test = np.load('features/full_test_x.npy')

    X_train = X_train.reshape(X_train.shape[0], 90, 80, 1)
    X_valid = X_valid.reshape(X_valid.shape[0], 90, 80, 1)
    X_test = X_test.reshape(X_test.shape[0], 90, 80, 1)

    encoder, decoder, autoencoder = create_models()

    if not os.path.exists('models'):
        os.makedirs('models')

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    history1 = autoencoder.fit(X_train, X_train, epochs=50, validation_data=(X_valid, X_valid))
    autoencoder.save('models/autoencoder.h5')
    autoencoder.save_weights('models/autoencoder_weights.h5')

    encoded_features = []
    for i in range(X_test.shape[0]):
        encoded_utterance = encoder.predict(np.expand_dims(X_test[i], 0))
        encoded_features.append(encoded_utterance)

    np.save('features/full_test_x_encoded.npy', encoded_features)

    if not os.path.exists('figures'):
        os.makedirs('figures')
    
    plt.plot(history1.history['loss'])
    plt.plot(history1.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper right')
    plt.savefig('figures/autoencoder_loss_curve.png', bbox_inches='tight')

    # To visualize encoded and decoded samples after training
    # autoencoder.load_weights('models/autoencoder_weights.h5')
    # plotSingleDecodedNoiseSample(decoder)
    # plotDecodedSamples(decoder)
    # plot_encoded_decoded_samples(encoder, decoder, X_valid)

if __name__ == '__main__':
    main()