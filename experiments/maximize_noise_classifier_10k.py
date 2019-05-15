import numpy as np
from keras.models import load_model
from keras import backend as K
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
    rate = 0.01
    gradients = []
    val = np.copy(selected_input)

    for _ in range(20):
        _, grads_value = iterate([val])
        val += grads_value * rate
        gradients.append(np.linalg.norm(grads_value))

    return val, gradients

def main():
    # Load model
    model = load_model('../models/speech_classifier_model.h5')
    second_model = load_model('../models/speech_classifier_model_2.h5')
    classes = os.listdir(DATA_PATH)

    class_1 = int(sys.argv[1])
    print('class_chosen', class_1, classes[class_1])

    # build a loss function that maximizes the activation of a specific class
    loss = K.mean(model.output[:, class_1])
    # compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, model.input)[0]
    # normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    # this function returns the loss and grads given the input speech
    iterate = K.function([model.input], [loss, grads])

    # Stores the prediction score of noise sample according to original speech classifier before maximization
    original_scores = []
    # Stores the prediction score of noise sample according to separate speech classifier before maximization
    original_scores_2 = []
    # Stores the prediction score of noise sample according to original speech classifier after maximization
    new_scores = []
    # Stores the prediction score of noise sample according to separate speech classifier after maximization
    new_scores_2 = []

    # Number of samples
    samples = 10000

    for _ in tqdm(range(samples)):
        # Generate random gaussian noise
        mu, sigma = 0, 0.1 # mean and standard deviation
        original_input = np.random.normal(mu, sigma, ((1, 90, 80, 1)))

        # Calculate accuracy of noise input with both classifiers
        o_score_1 = model.predict(original_input)
        original_scores.append(o_score_1)

        o_score_2 = second_model.predict(original_input)
        original_scores_2.append(o_score_2)

        maximized_noise, _ = maximizeInput(iterate, original_input)

        # Calculate accuracy of maximized input with both classifiers
        n_score_1 = model.predict(maximized_noise)
        new_scores.append(n_score_1)
        n_score_2 = second_model.predict(maximized_noise)
        new_scores_2.append(n_score_2)

    directories = ['results', 'results/classifier']
    for d in directories:
        if not os.path.exists(d):
            os.makedirs(d)

    # Save all prediction scores for analysis
    np.savez('results/classifier/10k_class_' + classes[class_1], class_index=class_1, orig_score_1=original_scores, max_score_1=new_scores, orig_score_2=original_scores_2, max_score_2=new_scores_2)

if __name__ == '__main__':
    main()