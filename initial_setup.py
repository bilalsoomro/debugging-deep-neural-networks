import subprocess
import shutil
import os

wavenet_dir = 'wavenet_vocoder'

# Reads line in input file and moves it to output dir
def extract_dataset(input_file, output_dir):
    with open(input_file) as in_file:
        for line in in_file:
            line = line.rstrip()
            directory = line.split('/')
            if not os.path.exists(output_dir + directory[0]):
                os.makedirs(output_dir + directory[0])

            newpath = shutil.copy('dataset/training/' + line, output_dir + line)
            if not os.path.isfile(newpath):
                print('file not copied: ' + newpath)
            else:
                os.remove('dataset/training/' + line)

def main():
    # Create directories for dataset
    directories = ['dataset', 'dataset/training', 'dataset/validation', 'dataset/testing']
    for d in directories:
        if not os.path.exists(d):
            os.makedirs(d)

    # Download Speech Commands V2 dataset and split data to training, validation & testing
    if not os.path.isfile('speech_commands_v0.02.tar.gz'):
        subprocess.call(['wget', 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'])
        subprocess.call(['tar', '-xzfv', 'speech_commands_v0.02.tar.gz', '-C', 'dataset/training'])
        extract_dataset('dataset/training/validation_list.txt', 'dataset/validation/')
        extract_dataset('dataset/training/testing_list.txt', 'dataset/testing/')

        # These files are removed because the feature extraction code will not require these
        subprocess.call(['rm', '-rf', 'dataset/training/validation_list.txt'])
        subprocess.call(['rm', '-rf', 'dataset/training/testing_list.txt'])
        subprocess.call(['rm', '-rf', 'dataset/training/README.md'])
        subprocess.call(['rm', '-rf', 'dataset/training/LICENSE'])
        subprocess.call(['rm', '-rf', 'dataset/training/_background_noise_'])
        subprocess.call(['rm', '-rf', 'dataset/training/.DS_Store'])

if __name__ == '__main__':
    main()