import subprocess
import os

wavenet_dir = 'wavenet_vocoder/'
train_data_dir = 'dataset/training/'
valid_data_dir = 'dataset/validation/'
test_data_dir = 'dataset/testing/'

train_features_dir = 'features/training/'
valid_features_dir = 'features/validation/'
test_features_dir = 'features/testing/'

params = wavenet_dir + '20180510_mixture_lj_checkpoint_step000320000_ema.json'

def main():
    # Create directories to extract features to
    directories = ['features', train_features_dir, valid_features_dir, test_features_dir]
    for d in directories:
        if not os.path.exists(d):
            os.makedirs(d)

    # Download pre-trained LJSpeech model hyperparameters
    if not os.path.isfile(wavenet_dir + '/' + '20180510_mixture_lj_checkpoint_step000320000_ema.json'):
        subprocess.call(['wget', '-P', wavenet_dir, 'https://www.dropbox.com/s/0vsd7973w20eskz/20180510_mixture_lj_checkpoint_step000320000_ema.json'])

    # Download pre-trained LJSpeech model
    if not os.path.isfile(wavenet_dir + '/' + '20180510_mixture_lj_checkpoint_step000320000_ema.pth'):
        subprocess.call(['wget', '-P', wavenet_dir, 'https://www.dropbox.com/s/zdbfprugbagfp2w/20180510_mixture_lj_checkpoint_step000320000_ema.pth'])

    # Install WaveNet
    owd = os.getcwd()
    os.chdir(wavenet_dir)
    subprocess.call(['pip3', 'install', '-U', '-e', '.[train]'])

    os.chdir(owd)
    
    # Extract features from training set
    subprocess.call(['python3', wavenet_dir + 'preprocess.py', 'speechcommands', 
        train_data_dir, train_features_dir, '--preset=' + params])

    # Extract features from validation set
    subprocess.call(['python3', wavenet_dir + 'preprocess.py', 'speechcommands', 
        valid_data_dir, valid_features_dir, '--preset=' + params])

    # Extract features from testing set
    subprocess.call(['python3', wavenet_dir + 'preprocess.py', 'speechcommands', 
        test_data_dir, test_features_dir, '--preset=' + params])

if __name__ == '__main__':
    main()