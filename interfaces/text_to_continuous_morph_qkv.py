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
from stqdm import stqdm
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

from diffusion_helper_qkv import *

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
random_seed = 1947
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
        prompts_map[prompt['source_prompt']] = prompt        
    return prompts_map

@st.cache_resource
def get_model():
    print('Loading model')
    
    latent_diffusion = build_model(model_name=model_name)
    latent_diffusion.latent_t_size = int(duration * latent_t_per_second)

    print('Model loaded')
    return latent_diffusion


def generate_left_target(latent_diffusion):
    generate_left_button_state = None
    if 'generate_left_button' in st.session_state:
        generate_left_button_state = st.session_state['generate_left_button']
    
    if generate_left_button_state:
        source_text_str = st.session_state['source_text']

        with st.spinner('Running...'):
            wav, img = sample_diffusion_attention_edit(latent_diffusion, source_text=None, target_text=source_text_str, random_seed=random_seed,\
                                                       ddim_steps=ddim_steps)
            st.session_state['source_image_placeholder'].image(img)
            st.session_state['source_audio_placeholder'].audio(wav, format="audio/wav", start_time=0, sample_rate=16000)

            st.session_state['source_image_placeholder_img'] = img
            st.session_state['source_audio_placeholder_wav'] = wav
   


def generate_right_target(latent_diffusion):
    generate_right_button_state = None
    if 'generate_right_button' in st.session_state:
        generate_right_button_state = st.session_state['generate_right_button']
    
    if generate_right_button_state:
        target_text_str = st.session_state['target_text']

        with st.spinner('Running...'):
            wav, img = sample_diffusion_attention_edit(latent_diffusion, source_text=None, target_text=target_text_str, random_seed=random_seed,\
                                                       ddim_steps=ddim_steps)
            st.session_state['target_image_placeholder'].image(img)
            st.session_state['target_audio_placeholder'].audio(wav, format="audio/wav", start_time=0, sample_rate=16000)

            st.session_state['target_image_placeholder_img'] = img
            st.session_state['target_audio_placeholder_wav'] = wav


def generate_morph(latent_diffusion):
    generate_morph_button_state = None
    if 'generate_morph_button' in st.session_state:
        generate_morph_button_state = st.session_state['generate_morph_button']

    if generate_morph_button_state:
        print('Inside sample diffusion morph -', generate_morph_button_state,'-', st.session_state['interpolation_level'])

        source_text_str = st.session_state['source_text']
        target_text_str = st.session_state['target_text']

        with st.spinner('Running...'):
            wav, img = sample_diffusion_attention_edit(latent_diffusion, source_text=source_text_str, target_text=target_text_str, \
                                                    random_seed=random_seed, interpolation_level=st.session_state['interpolation_level'],\
                                                        interpolate_terms=['q','k','v'], ddim_steps=ddim_steps)
            st.session_state['morph_image_placeholder'].image(img)
            st.session_state['morph_audio_placeholder'].audio(wav, format="audio/wav", start_time=0, sample_rate=16000)

            st.session_state['morph_image_placeholder_img'] = img
            st.session_state['morph_audio_placeholder_wav'] = wav


def main():
    
    css="""
    <style>
        [data-testid="stAppViewBlockContainer"] {
            padding: 2rem;
        }
    </style>
    """
    st.write(css, unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>'Text-to-Continuous' <br/> Sound Space Exploration w/ Text</h3>", unsafe_allow_html=True)

    print('before get model')
    latent_diffusion = get_model()
    print('after get model')

    prompts_map = populate_prompts()

    col1, col2, col3, col4, col5 = st.columns((0.3,0.05,0.3,0.05,0.3))
    
    with col1:
        st.markdown("<h3 style='text-align: center;'>Left</h3>", unsafe_allow_html=True)
        st.text_input(label="Source Text", key="source_text", value="", label_visibility='hidden')

        source_image_placeholder = st.empty()
        source_audio_placeholder = st.empty()
        st.session_state['source_image_placeholder'] = source_image_placeholder
        st.session_state['source_audio_placeholder'] = source_audio_placeholder
        if 'source_image_placeholder_img' in st.session_state:
            source_image_placeholder.image(st.session_state['source_image_placeholder_img'])
        if 'source_audio_placeholder_wav' in st.session_state:
            source_audio_placeholder.audio(st.session_state['source_audio_placeholder_wav'], format="audio/wav", start_time=0, sample_rate=16000)
        
        st.button("Generate Left Target", on_click=generate_left_target(latent_diffusion), type='secondary', key="generate_left_button", use_container_width=True)
        
    with col2:
        st.markdown("<br/>", unsafe_allow_html=True)

    with col3:
        st.markdown("<h3 style='text-align: center;'></h3>", unsafe_allow_html=True)
        slider_position=st.slider('Interpolation Level', min_value=0.0, max_value=1.0, value=0.0, step=0.01, label_visibility="hidden",  \
                                  format=None, key='interpolation_level', disabled=False)
        st.markdown("<br/>", unsafe_allow_html=True)
        
        morph_image_placeholder = st.empty()
        morph_audio_placeholder = st.empty()
        st.session_state['morph_image_placeholder'] = morph_image_placeholder
        st.session_state['morph_audio_placeholder'] = morph_audio_placeholder
        if 'morph_image_placeholder_img' in st.session_state:
            morph_image_placeholder.image(st.session_state['morph_image_placeholder_img'])
        if 'morph_audio_placeholder_wav' in st.session_state:
            morph_audio_placeholder.audio(st.session_state['morph_audio_placeholder_wav'], format="audio/wav", start_time=0, sample_rate=16000)


        st.button("Generate Morph", on_click=generate_morph(latent_diffusion), type='primary', key="generate_morph_button", use_container_width=True)

    with col4:
        st.markdown("<br/>", unsafe_allow_html=True)

    with col5:
        st.markdown("<h3 style='text-align: center;'>Right</h3>", unsafe_allow_html=True)
        st.text_input(label="Target Text", key="target_text", value="", label_visibility="hidden")

        target_image_placeholder = st.empty()
        target_audio_placeholder = st.empty()
        st.session_state['target_image_placeholder'] = target_image_placeholder
        st.session_state['target_audio_placeholder'] = target_audio_placeholder
        if 'target_image_placeholder_img' in st.session_state:
            target_image_placeholder.image(st.session_state['target_image_placeholder_img'])
        if 'target_audio_placeholder_wav' in st.session_state:
            target_audio_placeholder.audio(st.session_state['target_audio_placeholder_wav'], format="audio/wav", start_time=0, sample_rate=16000)

        st.button("Generate Right Target", on_click=generate_right_target(latent_diffusion), type='secondary', key="generate_right_button", use_container_width=True)


    st.markdown('<div style="text-align:center;color:white"><i>All audio samples on this page are generated with a sampling rate of 16kHz.</i></div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()