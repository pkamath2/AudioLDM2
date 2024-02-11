import gradio as gr
import numpy as np
import IPython
from IPython.display import Audio, display
import torch
import librosa, librosa.display

from audioldm2 import text_to_audio, build_model

import matplotlib.pyplot as plt
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


audioldm = None
current_model_name = None

model_name = 'audioldm_16k_crossattn_t5' # doesnt work very well?
# model_name = 'audioldm2-full' # Needs more memory than what I have
# model_name = 'audioldm_48k'

audioldm=build_model(model_name=model_name)


text = "A drumstick is continuously hitting a hard metal surface in a large room."
latent_t_per_second=25.6
sample_rate=16000
duration = 10.0 #Duration is minimum 10 secs. The generated sounds are weird for <10secs
guidance_scale = 3
random_seed = 45
n_candidates = 1


#wav = text2audio(text, duration, guidance_scale, random_seed, n_candidates, model_name)
wav, intermediates, predicted_noise, predicted_uncond_noise = text_to_audio(
    latent_diffusion=audioldm,
    text=text,
    x_T=None,
    seed=random_seed,
    duration=duration,
    guidance_scale=guidance_scale,
    n_candidate_gen_per_text=int(n_candidates),
    latent_t_per_second=latent_t_per_second,
)  # [bs, 1, samples]
print(len(wav))