import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import os, sys, io
import librosa, librosa.display
import soundfile as sf
import torch
import pickle
import urllib.request

from tqdm import tqdm
import json

import sys
sys.path.insert(0, '../')
from audioldm2 import text_to_audio, build_model, seed_everything, make_batch_for_text_to_audio
from audioldm2.latent_diffusion.modules.diffusionmodules.util import (
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
    noise_like,
    extract_into_tensor,
)
from audioldm2.latent_diffusion.models.ddim import DDIMSampler
from audioldm2.utilities import *
from audioldm2.utilities.audio import *
from audioldm2.utilities.data import *
from audioldm2.utils import default_audioldm_config

from audioldm2.gaverutils import gaver_sounds

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

model_name = 'audioldm_16k_crossattn_t5'
latent_t_per_second=25.6
sample_rate=16000
duration = 10.0 #Duration is minimum 10 secs. The generated sounds are weird for <10secs
guidance_scale = 3
random_seed = 42
n_candidates = 1
batch_size = 1
ddim_steps = 20

latent_diffusion = None


def get_config(filepath='config/config.json'):
    config = {}
    with open(filepath, 'r') as f:
        config = json.load(f)
    return config


def populate_prompts():
    prompts_map = {}
    prompts = get_config()
    for prompt in prompts:
        prompts_map[prompt['text']] = {'id':prompt['id'], 'slider_words': prompt['slider_words']}        
    return prompts_map

@st.cache_data
def get_model():
    print('Loading model')
    
    latent_diffusion = build_model(model_name=model_name)
    latent_diffusion.latent_t_size = int(duration * latent_t_per_second)

    print('Model loaded')
    return latent_diffusion


def sample_diffusion(latent_diffusion):
    #batch, x_init=None, random_seed=45, attention_weights=None
    batch = {
        "text": [st.session_state['prompt_selected']],  # list
        "fname": ['some-name']  # list
    }
    x_init=None
    attention_weights = get_attention_weights(latent_diffusion)

    with torch.no_grad():
        seed_everything(int(random_seed))
        if x_init is None:
            x_init = torch.randn((1, 8, 256, 16), device="cuda")

        text_conditioning = batch["text"]

        # The logic behind creating this "conditioning dict" is a little convoluted. 
        # We are able to use only text for conditioning, but for some reason the system expects other MAE related values to be
        # set on the dict. Anyways... junk code - but works. 
        cond_batch = make_batch_for_text_to_audio(text_conditioning[0], transcription="", waveform=None, batchsize=batch_size)
        _, c = latent_diffusion.get_input(cond_batch, latent_diffusion.first_stage_key,unconditional_prob_cfg=0.0)  # Do not output unconditional information in the c
        cond_dict = latent_diffusion.filter_useful_cond_dict(c)

        uncond_dict = {}
        for key in latent_diffusion.cond_stage_model_metadata:
            model_idx = latent_diffusion.cond_stage_model_metadata[key]["model_idx"]
            uncond_dict[key] = latent_diffusion.cond_stage_models[
                model_idx
            ].get_unconditional_condition(batch_size)
        

        shape = (latent_diffusion.channels, latent_diffusion.latent_t_size, latent_diffusion.latent_f_size)
        device=latent_diffusion.device
        eta=1.0
        temperature = 1.0
        noise = noise_like(x_init.shape, device, repeat=False) * temperature

        ddim_sampler = DDIMSampler(latent_diffusion, device=device)
        ddim_sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=eta, verbose=False)
        
        timesteps = ddim_sampler.ddim_timesteps

        intermediates = {"x_prev": [x_init], "predicted_noise_orig": [x_init], "predicted_noise_sega": [x_init]}
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")
        iterator = tqdm(time_range, desc="DDIM Sampler", total=total_steps)
    
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            t_in = torch.full((batch_size,), step, device=device, dtype=torch.long)
    
            #from p_sample_ddim
            model_uncond = ddim_sampler.model.apply_model(x_init, t_in, uncond_dict) #Unconditioned epsilon estimate
            model_cond = ddim_sampler.model.apply_model(x_init, t_in, cond_dict, attention_weights=attention_weights) #Conditioned epsilon estimate

            
            # CFG; model_output is the estimated error after CFG
            e_t = model_uncond + guidance_scale * (model_cond - model_uncond)

        
            alphas = ddim_sampler.ddim_alphas
            alphas_prev = ddim_sampler.ddim_alphas_prev
    
            sqrt_one_minus_alphas = ddim_sampler.ddim_sqrt_one_minus_alphas
            sigmas = ddim_sampler.ddim_sigmas
            
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((batch_size, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((batch_size, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((batch_size, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((batch_size, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)
    
            noise = sigma_t * noise_like(x_init.shape, device, repeat=False) * temperature
            
            pred_x0 = (x_init - sqrt_one_minus_at * e_t) / a_t.sqrt()
            dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            
    
            x_init = x_prev
            #return x_prev, pred_x0, e_t, model_uncond => img, pred_x0, predicted_noise, predicted_uncond_noise
    
            log_every_t = 1
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates["x_prev"].append(x_prev)
                intermediates["predicted_noise_orig"].append(e_t)
    
        mel = latent_diffusion.decode_first_stage(x_prev)
        waveform = latent_diffusion.mel_spectrogram_to_waveform(
            mel, savepath="", bs=None, name="", save=False
        )

        fig =plt.figure(figsize=(10, 5))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(waveform[0][0], hop_length=512)),ref=np.max)
        librosa.display.specshow(D, y_axis='linear', sr=16000, hop_length=512, x_axis='time')
        io_buf = io.BytesIO()
        fig.savefig(io_buf, format='raw')
        io_buf.seek(0)
        img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                            newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        io_buf.close()

        return waveform[0][0], img_arr
        #return mel, waveform, model_cond, model_uncond, e_t, intermediates 
        #=> mel of estimated z; wav of estimated z; predicted noise (epsilon); predicted noise unconditional; predicted noise with CFG

def get_attention_weights(latent_diffusion):
    prompt = st.session_state['prompt_selected']

    tokens = latent_diffusion.cond_stage_models[latent_diffusion.cond_stage_model_metadata["crossattn_flan_t5"]["model_idx"]].get_words_token_mapping(prompt) #print the mapping
    context, attn_mask = latent_diffusion.cond_stage_models[0].encode_text(prompt)

    attention_weights = torch.from_numpy(np.array([1.0 for i in range(context.shape[1])])).float().cuda()
    word_token_map = {}
    for ind, word in enumerate([i for i in prompt.split(' ')]):
        print(word, tokens[ind])
        word_token_map[word] = tokens[ind]

        if 'slider_'+word in st.session_state:
            print(st.session_state['slider_'+word], tokens[ind][0], tokens[ind][-1]+1)
            attention_weights[tokens[ind][0]:tokens[ind][-1]+1] = st.session_state['slider_'+word]
    
    print(attention_weights)

    return attention_weights

def main():

    latent_diffusion = get_model()

    prompts_map = populate_prompts()


    st.markdown("<h2 style='text-align: center;'>'Text-to-Continuous' Semantic Control For Audio</h2>", unsafe_allow_html=True)

    prompt_selected =  st.selectbox('Select a prompt', sorted(prompts_map.keys()), key='prompt_selected')
    slider_words = prompts_map[prompt_selected]['slider_words']
    slider_id = str(prompts_map[prompt_selected]['id'])

    
    

    s_wav, s_spec = sample_diffusion(latent_diffusion)
    # s_wav = st.session_state['wav']
    # s_spec = st.session_state['spectrogram']
    print(s_wav)

    col1, col2, col3, col4 = st.columns((0.3,0.1,0.4,0.2))

    with col1:
        st.markdown("<br/>", unsafe_allow_html=True)
        display_text = prompt_selected
        for word in slider_words:
            display_text = display_text.replace(word, "<span style='background-color: yellow; color:black;'>"+word+"</span>")
        st.markdown("<div style='text-align: left;'>"+display_text+"</div>", unsafe_allow_html=True)
        st.markdown("<br/>", unsafe_allow_html=True)
        for word in slider_words:
            slider_position=st.slider(word, min_value=-5.0, max_value=5.0, value=1.0, step=0.1,  format=None, key='slider_'+word, help=None, args=None, kwargs=None, disabled=False)
    with col2:
        vert_space = '<div style="padding: 25%;">&nbsp;</div>'
        st.markdown(vert_space, unsafe_allow_html=True)
        # st.button("**Generate** =>", on_click=sample_diffusion(latent_diffusion), type='primary')
    with col3:
        st.image(s_spec)
        st.audio(s_wav, format="audio/wav", start_time=0, sample_rate=16000)

    st.markdown('<div style="text-align:center;color:white"><i>All audio samples on this page are generated with a sampling rate of 16kHz.</i></div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()