#!/bin/bash
wavenet="../../../../wavenet_vocoder/"

for f in *.npy
    do python3 $wavenet"synthesis.py" --preset=$wavenet"20180510_mixture_lj_checkpoint_step000320000_ema.json" --conditional="$f" $wavenet"20180510_mixture_lj_checkpoint_step000320000_ema.pth" generated --file-name-suffix="_$f"    
done