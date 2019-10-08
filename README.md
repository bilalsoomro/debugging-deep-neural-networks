# Towards Debugging Deep Neural Networks by Generating Speech Utterances
This repo contains the code for the paper "Towards Debugging Deep Neural Networks by Generating Speech Utterances".

(https://arxiv.org/abs/1907.03164)


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

### Generated Audio Samples
- Maximizing using only trained Speech classifier
    - Random noise to class "Yes" - [Maximized](http://cs.uef.fi/~anssk/audio_maximization/20180510_mixture_lj_checkpoint_step000320000_emasample_1553098532_maximized_feature_to_class_yes.npy.wav)
    - Random noise to class "No" - [Maximized](http://cs.uef.fi/~anssk/audio_maximization/20180510_mixture_lj_checkpoint_step000320000_emasample_1553098197_maximized_feature_to_class_no.npy.wav)
    - Random noise to class "Up" - [Maximized](http://cs.uef.fi/~anssk/audio_maximization/20180510_mixture_lj_checkpoint_step000320000_emasample_1553099828_maximized_feature_to_class_up.npy.wav)
    - Random noise to class "Down" - [Maximized](http://cs.uef.fi/~anssk/audio_maximization/20180510_mixture_lj_checkpoint_step000320000_emasample_1553098022_maximized_feature_to_class_down.npy.wav)
    - Random noise to class "Go" - [Maximized](http://cs.uef.fi/~anssk/audio_maximization/20180510_mixture_lj_checkpoint_step000320000_emasample_1553098926_maximized_feature_to_class_go.npy.wav)

    - Class "On" - [WaveNet](http://cs.uef.fi/~anssk/audio_maximization/20180510_mixture_lj_checkpoint_step000320000_emasample_1553113920_original_feature_of_class_on_to_class_on.npy.wav) -> [Maximized](http://cs.uef.fi/~anssk/audio_maximization/20180510_mixture_lj_checkpoint_step000320000_emasample_1553113920_maximized_feature_of_class_on_to_class_on.npy.wav)
    - Class "Yes" - [WaveNet](http://cs.uef.fi/~anssk/audio_maximization/20180510_mixture_lj_checkpoint_step000320000_emasample_1553115970_original_feature_of_class_yes_to_class_yes.npy.wav) -> [Maximized](http://cs.uef.fi/~anssk/audio_maximization/20180510_mixture_lj_checkpoint_step000320000_emasample_1553115970_maximized_feature_of_class_yes_to_class_yes.npy.wav)
    - Class "Up" - [WaveNet](http://cs.uef.fi/~anssk/audio_maximization/20180510_mixture_lj_checkpoint_step000320000_emasample_1553152951_original_feature_of_class_up_to_class_up.npy.wav) -> [Maximized](http://cs.uef.fi/~anssk/audio_maximization/20180510_mixture_lj_checkpoint_step000320000_emasample_1553152951_maximized_feature_of_class_up_to_class_up.npy.wav)
    - Class "Stop" - [WaveNet](http://cs.uef.fi/~anssk/audio_maximization/20180510_mixture_lj_checkpoint_step000320000_emasample_1553165449_original_feature_of_class_stop_to_class_stop.npy.wav) -> [Maximized](http://cs.uef.fi/~anssk/audio_maximization/20180510_mixture_lj_checkpoint_step000320000_emasample_1553165449_maximized_feature_of_class_stop_to_class_stop.npy.wav)
    - Class "Tree" - [WaveNet](http://cs.uef.fi/~anssk/audio_maximization/20180510_mixture_lj_checkpoint_step000320000_emasample_1553113407_original_feature_of_class_tree_to_class_tree.npy.wav) -> [Maximized](http://cs.uef.fi/~anssk/audio_maximization/20180510_mixture_lj_checkpoint_step000320000_emasample_1553113407_maximized_feature_of_class_tree_to_class_tree.npy.wav)

- Maximizing using combined Speech classifier and decoder
    - Random noise to class "Yes" - [link](http://cs.uef.fi/~anssk/audio_maximization/20180510_mixture_lj_checkpoint_step000320000_emasample_1553351404_maximized_feature_to_class_yes.npy.wav)
    - Random noise to class "No" - [link](http://cs.uef.fi/~anssk/audio_maximization/20180510_mixture_lj_checkpoint_step000320000_emasample_1553351019_maximized_feature_to_class_no.npy.wav)
    - Random noise to class "Up" - [link](http://cs.uef.fi/~anssk/audio_maximization/20180510_mixture_lj_checkpoint_step000320000_emasample_1553352876_maximized_feature_to_class_up.npy.wav)
    - Random noise to class "Down" - [link](http://cs.uef.fi/~anssk/audio_maximization/20180510_mixture_lj_checkpoint_step000320000_emasample_1553350828_maximized_feature_to_class_down.npy.wav)
    - Random noise to class "Go" - [link](http://cs.uef.fi/~anssk/audio_maximization/20180510_mixture_lj_checkpoint_step000320000_emasample_1553351932_maximized_feature_to_class_go.npy.wav)

    - Class "On" - [WaveNet](http://cs.uef.fi/~anssk/audio_maximization/20180510_mixture_lj_checkpoint_step000320000_emasample_1553626531_original_feature_of_class_on_to_class_on.npy.wav) -> [Maximized](http://cs.uef.fi/~anssk/audio_maximization/20180510_mixture_lj_checkpoint_step000320000_emasample_1553626531_maximized_feature_of_class_on_to_class_on.npy.wav)
    - Class "Yes" - [Wavenet](http://cs.uef.fi/~anssk/audio_maximization/20180510_mixture_lj_checkpoint_step000320000_emasample_1553626541_original_feature_of_class_yes_to_class_yes.npy.wav) -> [Maximized](http://cs.uef.fi/~anssk/audio_maximization/20180510_mixture_lj_checkpoint_step000320000_emasample_1553626541_maximized_feature_of_class_yes_to_class_yes.npy.wav)
    - Class "Up" - [WaveNet](http://cs.uef.fi/~anssk/audio_maximization/20180510_mixture_lj_checkpoint_step000320000_emasample_1553626773_original_feature_of_class_up_to_class_up.npy.wav) -> [Maximized](http://cs.uef.fi/~anssk/audio_maximization/20180510_mixture_lj_checkpoint_step000320000_emasample_1553626773_maximized_feature_of_class_up_to_class_up.npy.wav)
    - Class "Stop" - [WaveNet](http://cs.uef.fi/~anssk/audio_maximization/20180510_mixture_lj_checkpoint_step000320000_emasample_1553626829_original_feature_of_class_stop_to_class_stop.npy.wav) -> [Maximized](http://cs.uef.fi/~anssk/audio_maximization/20180510_mixture_lj_checkpoint_step000320000_emasample_1553626829_maximized_feature_of_class_stop_to_class_stop.npy.wav)
    - Class "Tree" - [WaveNet](http://cs.uef.fi/~anssk/audio_maximization/20180510_mixture_lj_checkpoint_step000320000_emasample_1553626526_original_feature_of_class_tree_to_class_tree.npy.wav) -> [Maximized](http://cs.uef.fi/~anssk/audio_maximization/20180510_mixture_lj_checkpoint_step000320000_emasample_1553626526_maximized_feature_of_class_tree_to_class_tree.npy.wav)

    

## License
Code original to this project is under MIT license. Code in the directory '/wavenet_vocoder' is included under MIT license by Ryuichi Yamamoto ([link](https://github.com/r9y9/wavenet_vocoder/blob/master/LICENSE.md)).
