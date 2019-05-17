import numpy as np
from keras.models import load_model, Model
from keras import backend as K
from keras.utils import to_categorical
from keras.layers import (Conv2D, MaxPooling2D, Reshape, Flatten, Input, Dense, UpSampling2D, Dropout)
import sys
import time
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
import csv
import pickle

DATA_PATH = '../../dataset/training/'

# Maximizing input based on https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
def get_loss_and_gradients(model, class_chosen):
    # build a loss function that maximizes the activation of a specific class
    loss = K.mean(model.output[:, class_chosen])
    # compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, model.input)[0]
    # normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    # this function returns the loss and grads given the input speech
    iterate = K.function([model.input], [loss, grads])
    return iterate

def maximizeInput(iterate, selected_input, lr):
    gradients = []
    val = np.copy(selected_input)

    for _ in range(20):
        _, grads_value = iterate([val])
        val += grads_value * lr
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

# This function prints the input features
def plotFigure(selected_input, filename):
    plt.imshow(selected_input, origin='lower')
    plt.colorbar()
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()

def max_noise_samples_classifier(original_model, classes):
    for class_idx in tqdm(range(len(classes))):

        # Get function which returns loss and gradients of specific neuron in output layer
        loss_grads = get_loss_and_gradients(original_model, class_idx)
        samples_maximized = 0

        # Stop when we maximize 3 samples successfully
        while samples_maximized < 3:
            print('Trying to maximize noise to class:', classes[class_idx])
            timestamp = str(int(time.time()))
            # Generate random gaussian noise
            mu, sigma = 0, 0.1 # mean and standard deviation
            original_input = np.random.normal(mu, sigma, ((1, 90, 80, 1)))

            # Calculate accuracy of noise input
            orig_scores = original_model.predict(original_input)

            # Perform activation maximization
            maximized_noise, gradients = maximizeInput(loss_grads, original_input, 0.01)

            # Calculate accuracy of maximized input
            new_scores = original_model.predict(maximized_noise)
            max_pred_class = np.argmax(new_scores[0])

            # if sample is successfully maximized to specificed class
            if max_pred_class == class_idx:
                print('Successfully maximized noise sample to class', classes[class_idx], 'using classifier')
                samples_maximized += 1
                # Plot figures of original input
                filename =  'results/classifier_noise_max/sample_' + timestamp + '_before_class_' + classes[class_idx] + '_maximized.png'
                plotFigure(original_input.squeeze().T, filename)
                # Plot figures of maximized input
                filename =  'results/classifier_noise_max/sample_' + timestamp + '_after_class_' + classes[class_idx] + '_maximized.png'
                plotFigure(maximized_noise.squeeze().T, filename)
                # Get the difference between the original and maximized input
                difference = original_input.squeeze() - maximized_noise.squeeze()
                filename =  'results/classifier_noise_max/sample_' + timestamp + '_diff_class_' + classes[class_idx] + '_maximized.png'
                plotFigure(difference.T, filename)

                # Save the maximized input for synthesis
                np.save('results/synthesize/sample_' + timestamp + '_classifier_maximized_noise_to_class_' + classes[class_idx] + '.npy', maximized_noise.squeeze())
        
                data = {
                    0: orig_scores,
                    1: new_scores,
                    3: gradients
                }

                with open('results/classifier_noise_max/sample_' + timestamp + '_scores_' + classes[class_idx] + '.pickle', 'wb') as f:
                    # Pickle the 'data' dictionary using the highest protocol available.
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def max_test_samples_classifier(original_model, classes):
    # Get test features
    X_test = np.load('../../features/full_test_x.npy')
    y_test = np.load('../../features/full_test_y.npy')
    y_test_hot = to_categorical(y_test)

    for class_idx in tqdm(range(len(classes))):

        test_samples_indices = []
        for j in range(np.shape(y_test_hot)[0]):
            if(y_test_hot[j][class_idx] == 1):
                test_samples_indices.append(j)

        np.random.shuffle(test_samples_indices)

        # Get function which returns loss and gradients of specific neuron in output layer
        loss_grads = get_loss_and_gradients(original_model, class_idx)
        samples_maximized = 0

        # Stop when we maximize 3 samples successfully
        while samples_maximized < 3:
            timestamp = str(int(time.time()))
            
            # Pick sample from test set
            idx = np.random.choice(test_samples_indices)
            print('Trying to max sample idx:', idx)
            # Pick sample from test set
            original_input = X_test[idx]
            original_input = original_input.reshape(1, 90, 80, 1)

            # Calculate accuracy of original input
            orig_scores = original_model.predict(original_input)

            # Perform activation maximization
            maximized_input, gradients = maximizeInput(loss_grads, original_input, 0.01)

            # Calculate accuracy of maximized input
            new_scores = original_model.predict(maximized_input)
            max_pred_class = np.argmax(new_scores[0])

            # if sample is successfully maximized to specificed class
            if max_pred_class == class_idx:
                print('Successfully maximized test sample to class', classes[class_idx], 'using classifier')
                samples_maximized += 1
                # Plot figures of original input
                filename =  'results/classifier_test_max/sample_' + timestamp + '_before_class_' + classes[class_idx] + '_maximized.png'
                plotFigure(original_input.squeeze().T, filename)
                # Plot figures of maximized input
                filename =  'results/classifier_test_max/sample_' + timestamp + '_after_class_' + classes[class_idx] + '_maximized.png'
                plotFigure(maximized_input.squeeze().T, filename)
                # Get the difference between the original and maximized input
                difference = original_input.squeeze() - maximized_input.squeeze()
                filename =  'results/classifier_test_max/sample_' + timestamp + '_diff_class_' + classes[class_idx] + '_maximized.png'
                plotFigure(difference.T, filename)

                # Save the original input for synthesis
                np.save('results/synthesize/sample_' + timestamp + '_classifier_before_maximized_class_to_class_' + classes[class_idx] + '.npy', original_input.squeeze())
                # Save the maximized input for synthesis
                np.save('results/synthesize/sample_' + timestamp + '_classifier_after_maximized_class_to_class_' + classes[class_idx] + '.npy', maximized_input.squeeze())

        
                data = {
                    0: orig_scores,
                    1: new_scores,
                    3: gradients
                }

                with open('results/classifier_test_max/sample_' + timestamp + '_scores_' + classes[class_idx] + '.pickle', 'wb') as f:
                    # Pickle the 'data' dictionary using the highest protocol available.
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def max_noise_samples_classifier_decoder(combined_model, decoder_model, classes):
    for class_idx in tqdm(range(len(classes))):

        # Get function which returns loss and gradients of specific neuron in output layer
        loss_grads = get_loss_and_gradients(combined_model, class_idx)
        samples_maximized = 0

        # Stop when we maximize 3 samples successfully
        while samples_maximized < 3:
            print('Trying to maximize noise to class:', classes[class_idx])
            timestamp = str(int(time.time()))
            # Generate random gaussian noise
            mu, sigma = 0, 0.1 # mean and standard deviation
            original_input = np.random.normal(mu, sigma, 256)
            original_input = np.expand_dims(original_input, 0)

            # Decode original bottleneck code
            original_input_decoded = decoder_model.predict(original_input)

            # Calculate accuracy of noise input
            orig_scores = combined_model.predict(original_input)

            # Perform activation maximization
            maximized_noise, gradients = maximizeInput(loss_grads, original_input, 0.1)

            # Decode maximized bottleneck code
            maximized_input_decoded = decoder_model.predict(maximized_noise)

            # Calculate accuracy of maximized input
            new_scores = combined_model.predict(maximized_noise)
            max_pred_class = np.argmax(new_scores[0])

            # if sample is successfully maximized to specificed class
            if max_pred_class == class_idx:
                print('Successfully maximized noise sample to class', classes[class_idx], 'using classifier and decoder')
                samples_maximized += 1
                # Plot figures of original input
                filename =  'results/combined_noise_max/sample_' + timestamp + '_before_class_' + classes[class_idx] + '_maximized.png'
                plotFigure(original_input_decoded.squeeze().T, filename)
                # Plot figures of maximized input
                filename =  'results/combined_noise_max/sample_' + timestamp + '_after_class_' + classes[class_idx] + '_maximized.png'
                plotFigure(maximized_input_decoded.squeeze().T, filename)
                # Get the difference between the original and maximized input
                difference = original_input_decoded.squeeze() - maximized_input_decoded.squeeze()
                filename =  'results/combined_noise_max/sample_' + timestamp + '_diff_class_' + classes[class_idx] + '_maximized.png'
                plotFigure(difference.T, filename)

                # Save the maximized input for synthesis
                np.save('results/synthesize/sample_' + timestamp + '_classifier_decoder_maximized_noise_to_class_' + classes[class_idx] + '.npy', maximized_input_decoded.squeeze())
        
                data = {
                    0: orig_scores,
                    1: new_scores,
                    3: gradients
                }

                with open('results/combined_noise_max/sample_' + timestamp + '_scores_' + classes[class_idx] + '.pickle', 'wb') as f:
                    # Pickle the 'data' dictionary using the highest protocol available.
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def max_test_samples_classifier_decoder(combined_model, decoder_model, classes):
    # Get test features
    X_test = np.load('../../features/full_test_x_encoded.npy')
    y_test = np.load('../../features/full_test_y.npy')
    y_test_hot = to_categorical(y_test)

    for class_idx in tqdm(range(len(classes))):

        test_samples_indices = []
        for j in range(np.shape(y_test_hot)[0]):
            if(y_test_hot[j][class_idx] == 1):
                test_samples_indices.append(j)

        np.random.shuffle(test_samples_indices)

        # Get function which returns loss and gradients of specific neuron in output layer
        loss_grads = get_loss_and_gradients(combined_model, class_idx)
        samples_maximized = 0

        # Stop when we maximize 3 samples successfully
        while samples_maximized < 3:
            timestamp = str(int(time.time()))
            
            idx = np.random.choice(test_samples_indices)
            print('y label sample idx:', str(idx), 'belongs to class', classes[int(y_test[idx])])
            print('Maximizing to class', classes[class_idx])
            # Pick sample from test set
            original_input = X_test[idx]

            # Decode original bottleneck code
            original_input_decoded = decoder_model.predict(original_input)

            # Calculate accuracy of noise input
            orig_scores = combined_model.predict(original_input)
            max_idx_orig = np.argmax(orig_scores[0])
            print('Predicted originally as:', max_idx_orig)
            
            # Perform activation maximization
            maximized_input, gradients = maximizeInput(loss_grads, original_input, 0.1)
            
            # Decode maximized bottleneck code
            maximized_input_decoded = decoder_model.predict(maximized_input)

            # Calculate accuracy of maximized input
            new_scores = combined_model.predict(maximized_input)
            max_pred_class = np.argmax(new_scores[0])

            # if sample is successfully maximized to specificed class
            if max_pred_class == class_idx:
                print('Successfully maximized test sample to class', classes[class_idx], 'using classifier and decoder')
                samples_maximized += 1
                # Plot figures of original input
                filename =  'results/combined_text_max/sample_' + timestamp + '_before_class_' + classes[class_idx] + '_maximized.png'
                plotFigure(original_input_decoded.squeeze().T, filename)
                # Plot figures of maximized input
                filename =  'results/combined_text_max/sample_' + timestamp + '_after_class_' + classes[class_idx] + '_maximized.png'
                plotFigure(maximized_input_decoded.squeeze().T, filename)
                # Get the difference between the original and maximized input
                difference = original_input_decoded.squeeze() - maximized_input_decoded.squeeze()
                filename =  'results/combined_text_max/sample_' + timestamp + '_diff_class_' + classes[class_idx] + '_maximized.png'
                plotFigure(difference.T, filename)

                # Save the original input for synthesis
                np.save('results/synthesize/sample_' + timestamp + '_classifier_decoder_before_maximized_class_to_class_' + classes[class_idx] + '.npy', original_input_decoded.squeeze())
                # Save the maximized input for synthesis
                np.save('results/synthesize/sample_' + timestamp + '_classifier_decoder_after_maximized_class_to_class_' + classes[class_idx] + '.npy', maximized_input_decoded.squeeze())

        
                data = {
                    0: orig_scores,
                    1: new_scores,
                    3: gradients
                }

                with open('results/combined_text_max/sample_' + timestamp + '_scores_' + classes[class_idx] + '.pickle', 'wb') as f:
                    # Pickle the 'data' dictionary using the highest protocol available.
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def main():
    # "combined_models" is the speech classifier and decoder (prior) combined.
    combined_model, decoder_model = init_models('../../models/autoencoder_weights.h5', '../../models/speech_classifier_weights.h5')
    # "original_model" is the original speech classifier model
    original_model = load_model('../../models/speech_classifier_model.h5')

    # Gets labels
    classes = os.listdir(DATA_PATH)

    directories = ['results', 'results/classifier_noise_max', 'results/classifier_test_max', 
    'results/combined_noise_max', 'results/combined_text_max', 'results/synthesize']
    for d in directories:
        if not os.path.exists(d):
            os.makedirs(d)

    # Generate maximized features from noise using only classifier
    max_noise_samples_classifier(original_model, classes)

    # Generate maximized features from test samples using only classifier
    max_test_samples_classifier(original_model, classes)

    # Generate maximized features from noise samples using combined classifier and decoder model
    max_noise_samples_classifier_decoder(combined_model, decoder_model, classes)

    # Generate maximized features from test samples using combined classifier and decoder model
    max_test_samples_classifier_decoder(combined_model, decoder_model, classes)

if __name__ == '__main__':
    main()