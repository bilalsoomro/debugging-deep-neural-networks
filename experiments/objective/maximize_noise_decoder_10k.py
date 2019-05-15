import numpy as np
from keras.models import load_model, Model
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape, UpSampling2D, Dropout, Input
import sys
import os
from tqdm import tqdm
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

DATA_PATH = '../dataset/training/'

# Maximizing input based on https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
def maximizeInput(iterate, selected_input):
    rate = 0.1
    gradients = []
    val = np.copy(selected_input)

    for _ in range(20):
        _, grads_value = iterate([val])
        val += grads_value * rate
        gradients.append(np.linalg.norm(grads_value))

    return val, gradients

# This function creates the combined model of the classifier and decoder(prior)
def init_models(autoencoder_weights, classifier_weights):

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

    # Initializes the layers with weights
    autoencoder.load_weights(autoencoder_weights)

    ####################################################################################################
    #------------------------------------------Classifier----------------------------------------------#
    ####################################################################################################

    # Layers
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

    # Model
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

    classifier = Model(inputs=input_shape, outputs=classifier)

    # Initializes the classifier's layers with weights
    classifier.load_weights(classifier_weights)

    combined_model = dense_2(bottleneck_input_shape)
    combined_model = reshape(combined_model)
    combined_model = conv_5(combined_model)
    combined_model = up_samp_1(combined_model)
    combined_model = conv_6(combined_model)
    combined_model = up_samp_2(combined_model)
    combined_model = conv_7(combined_model)
    combined_model = up_samp_3(combined_model)
    combined_model = decoded(combined_model)

    combined_model = c_conv_1(combined_model)
    combined_model = c_max_pool_1(combined_model)
    combined_model = c_drop_1(combined_model)
    combined_model = c_conv_2(combined_model)
    combined_model = c_max_pool_2(combined_model)
    combined_model = c_drop_2(combined_model)
    combined_model = c_conv_3(combined_model)
    combined_model = c_max_pool_3(combined_model)
    combined_model = c_drop_3(combined_model)
    combined_model = c_flatten(combined_model)

    combined_model = c_dense_1(combined_model)
    combined_model = c_drop_4(combined_model)
    combined_model = c_dense_2(combined_model)
    combined_model = c_drop_5(combined_model)
    combined_model = c_dense_3(combined_model)
    combined_model = c_drop_6(combined_model)
    combined_model = c_dense_output(combined_model)

    full_model = Model(inputs=bottleneck_input_shape, outputs=combined_model)
    
    return full_model, decoder_model

def main():
    # Load model
    full_model, decoder_model = init_models('../models/autoencoder_weights.h5', '../models/speech_classifier_weights.h5')
    second_model = load_model('../models/speech_classifier_model_2.h5')

    classes = os.listdir(DATA_PATH)

    class_1 = int(sys.argv[1])
    print('class_chosen', class_1, classes[class_1])

    # build a loss function that maximizes the activation of a specific class
    loss = K.mean(full_model.output[:, class_1])
    # compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, full_model.input)[0]
    # normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    # this function returns the loss and grads given the input speech
    iterate = K.function([full_model.input], [loss, grads])

    # Stores the prediction score of noise sample according to original combined decoder + speech classifier before maximization
    original_scores = []
    # Stores the prediction score of noise sample according to separate speech classifier before maximization
    original_scores_2 = []
    # Stores the prediction score of noise sample according to original combined decoder + speech classifier after maximization
    new_scores = []
    # Stores the prediction score of noise sample according to separate speech classifier after maximization
    new_scores_2 = []

    # Number of samples
    samples = 10000

    for _ in tqdm(range(samples)):
        # Generate random gaussian noise
        mu, sigma = 0, 0.1 # mean and standard deviation
        original_input = np.random.normal(mu, sigma, 256)
        original_input = np.expand_dims(original_input, 0)

        # Calculate accuracy of noise input with both classifiers
        o_score_1 = full_model.predict(original_input)
        original_scores.append(o_score_1)

        # Decode original bottleneck code
        original_input_decoded = decoder_model.predict(original_input)

        o_score_2 = second_model.predict(original_input_decoded)
        original_scores_2.append(o_score_2)

        maximized_noise, _ = maximizeInput(iterate, original_input)

        # Decode maximized bottleneck code
        maximized_input_decoded = decoder_model.predict(maximized_noise)

        # Calculate accuracy of maximized input with both classifiers
        n_score_1 = full_model.predict(maximized_noise)
        new_scores.append(n_score_1)
        n_score_2 = second_model.predict(maximized_input_decoded)
        new_scores_2.append(n_score_2)
    
    directories = ['results', 'results/decoder']
    for d in directories:
        if not os.path.exists(d):
            os.makedirs(d)

    # Save all prediction scores for analysis
    np.savez('results/decoder/10k_class_' + classes[class_1], class_index=class_1, orig_score_1=original_scores, max_score_1=new_scores, orig_score_2=original_scores_2, max_score_2=new_scores_2)

if __name__ == '__main__':
    main()