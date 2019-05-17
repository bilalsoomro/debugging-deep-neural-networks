from pathlib import Path
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import simplejson
from keras.models import load_model

DATA_PATH = '../../dataset/training/'

def getExperiment(filename):
    if '_classifier_maximized_noise_to_class_' in filename:
        return 1
    elif '_classifier_before_maximized_class_to_class_' in filename or '_classifier_after_maximized_class_to_class_' in filename:
        return 2
    elif '_classifier_decoder_maximized_noise_to_class_' in filename:
        return 3
    elif '_classifier_decoder_before_maximized_class_to_class_' in filename or '_classifier_decoder_after_maximized_class_to_class_' in filename:
        return 4

def main():
    first_model = load_model('../../models/speech_classifier_model.h5')
    second_model = load_model('../../models/speech_classifier_model_2.h5')

    classes = os.listdir(DATA_PATH)

    wav_paths = []
    for pth in Path('results/synthesize').iterdir():
        if pth.suffix == '.npy':
            wav_paths.append(str(pth))

    np.random.shuffle(wav_paths)
    print('total files', len(wav_paths))

    with open('input.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['experiment', 'timestamp', 'class_label', 'audio_url', 'pred_1_class', 'pred_2_class'])

        for path in tqdm(wav_paths):

            original_feature = np.load(path)
            feature = original_feature.reshape(1, 90, 80, 1)

            orig_scores = first_model.predict(feature)
            class_idx_1 = np.argmax(orig_scores[0])

            new_scores = second_model.predict(feature)
            class_idx_2 = np.argmax(new_scores[0])

            filename = os.path.basename(path)
            wav_filename = '20180510_mixture_lj_checkpoint_step000320000_ema_' + filename

            filename_split = wav_filename.split('_')
            class_label = filename_split[-1].split('.')[0]
            timestamp = filename_split[7]

            spamwriter.writerow([getExperiment(filename), timestamp, class_label, 'audio_directory/' + wav_filename + '.wav', classes[class_idx_1], classes[class_idx_2]])

if __name__ == '__main__':
    main()