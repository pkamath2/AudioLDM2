import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import os, sys, io
import librosa, librosa.display
import soundfile as sf
import torch
import pickle
import urllib.request

import sys
sys.path.insert(0, '../')
from utils import util, google_analytics
from networks import stylegan_encoder
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import uuid
from argparse import Namespace

config = util.get_config('../interface/frontend/src/config.json')
config = Namespace(**dict(**config))


def get_vector(x):
    return torch.from_numpy(x).float().cuda()

def reconstruct(encoded):
    model = st.session_state['model_picked']
    G = st.session_state['G']
    reconstructed_audio = G.synthesis(encoded)
    filler = torch.full((1, 1, 1, reconstructed_audio[0].shape[1]), torch.min(reconstructed_audio)).cuda()
    reconstructed_audio = torch.cat([reconstructed_audio, filler], dim=2)
    reconstructed_audio = util.renormalize(reconstructed_audio, (torch.min(reconstructed_audio), torch.max(reconstructed_audio)), (-50, 0))
    reconstructed_audio = reconstructed_audio.detach().cpu().numpy()[0]
    reconstructed_audio_wav = util.pghi_istft(reconstructed_audio, hop_size=config.model_list[model]['hop_size'], stft_channels=config.stft_channels)
    return reconstructed_audio_wav, reconstructed_audio

@st.cache_data
def get_dimcontrol_model(model):
    print('getting model', model)
    try:
        stylegan_pkl = config.model_list[model]['ckpt_stylegan2_path']
        encoder_pkl = config.model_list[model]['ckpt_encoder_path']
        stylegan_pkl_url = config.model_list[model]['stylegan_pkl_url']
        encoder_pkl_url = config.model_list[model]['encoder_pkl_url']
    except:
        print("Unknown Model!")
        return None, None

    if not os.path.isfile(stylegan_pkl):
        os.makedirs(config.model_list[model]['ckpt_download_stylegan2_path'], exist_ok=True)
        urllib.request.urlretrieve(stylegan_pkl_url, stylegan_pkl)

    if not os.path.isfile(encoder_pkl):
        os.makedirs(config.model_list[model]['ckpt_download_encoder_path'], exist_ok=True)
        urllib.request.urlretrieve(encoder_pkl_url, encoder_pkl)

    with open(stylegan_pkl, 'rb') as pklfile:
        network = pickle.load(pklfile)
        G = network['G'].eval().cuda()

    netE = stylegan_encoder.load_stylegan_encoder(domain=None, nz=G.z_dim,
                                                outdim=G.z_dim,
                                                use_RGBM=True,
                                                use_VAE=False,
                                                resnet_depth=34,
                                                ckpt_path=encoder_pkl).eval().cuda()
    return G, netE

# for hits & scratches right now
@st.cache_data
def get_concept_directions():
    brightness_vector = np.load('direction_vectors/brightness.npy')
    rate_vector = np.load('direction_vectors/rate.npy')
    impacttype_vector = np.load('direction_vectors/impacttype.npy')
    
    print(np.linalg.norm(brightness_vector), np.linalg.norm(rate_vector), np.linalg.norm(impacttype_vector))
    brightness_vector = get_vector(brightness_vector/np.linalg.norm(brightness_vector)) #Can we do something with the magnitude?
    rate_vector = get_vector(rate_vector/np.linalg.norm(rate_vector)) #Can we do something with the magnitude?
    impacttype_vector = get_vector(impacttype_vector/np.linalg.norm(impacttype_vector)) #Can we do something with the magnitude?

    return brightness_vector, rate_vector, impacttype_vector

def sample(pos, session_uuid=''):
    G = st.session_state['G']
    netE = st.session_state['netE']

    brightness_vector, rate_vector, impacttype_vector = get_concept_directions()

    if 'initial_sample' not in st.session_state:
        z = torch.from_numpy(np.random.rand(1, G.z_dim)).cuda()
        w = G.mapping(z, None)
        st.session_state['initial_sample'] = w
        print(st.session_state['initial_sample'].shape)

    start_time = time.time()
    w = st.session_state['initial_sample']

    w_ = w.clone()
    w_ += brightness_vector * pos[0]
    w_ += rate_vector * pos[1]
    w_ += impacttype_vector * pos[2]

    img = G.synthesis(w_)    
    audio, img_1 = reconstruct(w_)
    print("--- Time taken to synthesize from G and invert using PGHI = %s seconds ---" % (time.time() - start_time))
    
    fig =plt.figure(figsize=(7, 5))
    a=librosa.display.specshow(img_1[0],x_axis='time', y_axis='linear',sr=config.sample_rate)
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw')
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()

    os.makedirs(config.pgc_tmp_audio_loc_path, exist_ok=True)
    sf.write(f'{config.pgc_tmp_audio_loc_path}{session_uuid}_temp_audio_loc.wav', audio.astype(float), 16000)
    print('--------------------------------------------------')


    audio_file = open(f'{config.pgc_tmp_audio_loc_path}{session_uuid}_temp_audio_loc.wav', 'rb')
    audio_bytes = audio_file.read()

    return img_arr, audio_bytes


def change_z():
    selected_option = st.session_state['selected_preset_option']
    print(selected_option)
    if selected_option == 'Random':
        if 'initial_sample' in st.session_state:
            del st.session_state['initial_sample'] 
    else:
        st.session_state['initial_sample'] = st.session_state['real_data_dict'][selected_option]
    st.session_state['brightness_slider_position'] = 0
    st.session_state['rate_slider_position'] = 0
    st.session_state['impacttype_slider_position'] = 0

def map_dropdown_name(input):
    return config.model_list[input]['name']

def main():
    if config.allow_analytics:
        google_analytics.set_google_analytics()
    st.markdown("<h2 style='text-align: center;'>Audio Texture Generation <br/>Guided by Semantic Prototypes</h2>", unsafe_allow_html=True)

    if 'session_uuid' not in st.session_state:
        st.session_state['session_uuid'] = str(uuid.uuid4())
    session_uuid = st.session_state['session_uuid']

    st.sidebar.title('Model Options')

    model_names = []
    for key in config.model_list:
        model_names.append(key)
    model_names = tuple(model_names)
    model_picked =  st.sidebar.selectbox('Select a model', model_names, format_func=map_dropdown_name, key='model_picked')
    
    G, netE = get_dimcontrol_model(model_picked)
    st.session_state['G'] = G
    st.session_state['netE'] = netE

    st.sidebar.markdown("<h1 style='text-align: center;'>Semantic Guidance</h1>", unsafe_allow_html=True)

    initialOptionsList = ['Random']

    option = st.sidebar.selectbox(
    'Choose sample',
    initialOptionsList,key='selected_preset_option', on_change=change_z)


    birghtness_position=st.sidebar.slider('Brightness', min_value=-5.0, max_value=5.0, value=0.0, step=0.1,  format=None, key='brightness_slider_position', help=None, args=None, kwargs=None, disabled=False)
    rate_position=st.sidebar.slider('Rate', min_value=-5.0, max_value=5.0, value=0.0, step=0.1,  format=None, key='rate_slider_position', help=None, args=None, kwargs=None, disabled=False)
    impacttype_position=st.sidebar.slider('Impact Type', min_value=-5.0, max_value=5.0, value=0.0, step=0.1,  format=None, key='impacttype_slider_position', help=None, args=None, kwargs=None, disabled=False)
    
    
    dc_col1, dc_col2, dc_col3 = st.columns((1,2,1))
    spectrogram_placeholder = dc_col2.empty()
    audio_placeholder = dc_col2.empty()

    

    s = sample([birghtness_position, rate_position, impacttype_position], session_uuid=session_uuid)
    spectrogram_placeholder.image(s[0],width=700)
    audio_placeholder.audio(s[1], format="audio/wav", start_time=0)

if __name__ == '__main__':
    main()