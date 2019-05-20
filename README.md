# Towards Debugging Deep Neural Networks by Generating Speech Utterances
This repo contains the code for the paper "Towards Debugging Deep Neural Networks by Generating Speech Utterances"


## Main files:
- initial_setup.py - Downloads and prepares Speech Commands V2 dataset ([link](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz)).
- extract_features.py - Uses [WaveNet](https://github.com/r9y9/wavenet_vocoder) feature extraction code to extract features from speech dataset.
- train_classifier.py - Contains code to train main speech classifier for maximization and same code was used to train a separate classifier for objective evaluations.
- train_autoencoder.py - Trains an autoencoder with speech dataset. Decoder part is then taken as learned prior.

## Experiments
Experiments folder contains the code used to perform objective evaluations, perceptual evaluations and the speaker & maximization effects.

### Objective evaluations
- maximize_noise_classifier_10k.py - python script to maximize 10k noise samples to a class using the speech classifier
- maximize_noise_decoder_10k.py - python script to maximize 10k noise samples to a class using the combined model of speech classifier and decoder(prior)
- 10k_maximization_experiment.sh - bash script to run the objective experiments
- noise_to_class_heatmaps.py - python script to visualize the results of the 10k noise samples maximization experiments
    - usage: python3 noise_to_class_heatmaps.py results/decoder classifier_heatmap.pdf

### Perceptual evaluations
- max_samples_perceptual - python script to maximize samples required for perceptual experiments. It generates a total of 630 samples which are evaluated by human listeners using Amazon's Mechanical Turks crowd-sourcing service.
  - Classifier noise max (3 samples per class) = 3 x 35 = 105
  - Classifier test max (3 samples per class) = 3 x 35 = 105 x 2 (before & after) = 210
  - Decoder noise max (3 samples per class) = 3 x 35 = 105
  - Decoder test max (3 samples per class) = 3 x 35 = 105 x 2 (before & after) = 210

- synthesize_audio.sh bash script to synthesize audio files using WaveNet vocoder
- produce_csv.py - python script used to generate csv input file for MTurk
- turk_analysis.py - python script used to visualize MTurk results and visualize results of the evaluations
    - Usage: python3 turk_analysis.py Batch_3584220_batch_results_updated.csv classes output.pdf

### t-SNE analysis
- prepare_data.py - python script to maximize encoded test samples to their respective class. It also gets which test samples are misclassified and extracts speaker_ids from the speech dataset.
- latent_tsne.py - python script to generate visualizations on speaker and maximization effect

## License
Code original to this project is under MIT license. Code in the directory '/wavenet_vocoder' is included under MIT license by Ryuichi Yamamoto ([link](https://github.com/r9y9/wavenet_vocoder/blob/master/LICENSE.md)).

