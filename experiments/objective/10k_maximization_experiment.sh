#!/bin/bash
# These scripts below maximize the input for their respective class 
# First arg = class index

for i in {0..34}
   do
      python3 maximize_noise_classifier_10k.py $i
   done

for i in {0..34}
   do
      python3 maximize_noise_decoder_10k.py $i
   done