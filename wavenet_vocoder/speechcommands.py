from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import audio

from nnmnkwii import preprocessing as P
from hparams import hparams
from os.path import exists
import librosa

from wavenet_vocoder.util import is_mulaw_quantize, is_mulaw, is_raw

from hparams import hparams


def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []

    labels = os.listdir(in_dir)
    for label in tqdm(labels):
        mel_vectors = []

        wavfiles = [in_dir + label + '/' + wavfile for wavfile in os.listdir(in_dir + label)]
        for wavfile in wavfiles:
            futures.append(executor.submit(
                partial(_process_utterance, out_dir, wavfile)))
            
        for future in tqdm(futures):
            mel = future.result()
            # To make fixed size inputs for models
            while mel.shape[0] > 90:
                mel = np.delete(mel, 1, 0)
            while mel.shape[0] < 90:
                padding = np.zeros(80)
                mel = np.vstack([mel, padding])

            # print('mel shape', np.shape(mel))
            mel_vectors.append(mel)

        print('mel shape: ', str(np.shape(mel_vectors)))
        np.save(out_dir + label + '.npy', mel_vectors)

def _process_utterance(out_dir, wav_path):
    # Load the audio to a numpy array:
    wav = audio.load_wav(wav_path)

    if hparams.rescaling:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max

    # Mu-law quantize
    if is_mulaw_quantize(hparams.input_type):
        # [0, quantize_channels)
        out = P.mulaw_quantize(wav, hparams.quantize_channels)

        # Trim silences
        start, end = audio.start_and_end_indices(out, hparams.silence_threshold)
        wav = wav[start:end]
        out = out[start:end]
        constant_values = P.mulaw_quantize(0, hparams.quantize_channels)
        out_dtype = np.int16
    elif is_mulaw(hparams.input_type):
        # [-1, 1]
        out = P.mulaw(wav, hparams.quantize_channels)
        constant_values = P.mulaw(0.0, hparams.quantize_channels)
        out_dtype = np.float32
    else:
        # [-1, 1]
        out = wav
        constant_values = 0.0
        out_dtype = np.float32

    # Compute a mel-scale spectrogram from the trimmed wav:
    # (N, D)
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32).T
    
    return mel_spectrogram.astype(np.float32)