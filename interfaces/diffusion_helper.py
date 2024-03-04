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

from audioldm2.latent_diffusion.modules.attention import SpatialTransformer, CrossAttention

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


class SaveAttentionMatrices:
    def __init__(self):
        self.attention = []
    def __call__(self, module, module_in, module_out):
        ret = module_out[1]
        if ret is not None:
            ret = ret.detach().cpu()    
        self.attention.append(ret)
            
    def clear(self):
        self.attention = []

def clear_attention_matrices(save_output):
    save_output.clear()
    
def register_save_attention(latent_diffusion, save_output):
    save_attention_hook_handles = []
    save_attention_hook_layer_names = []
    for n, m in latent_diffusion.named_modules():
        if(isinstance(m, CrossAttention)):
            if 'attn2' in n:
                handle = m.register_forward_hook(save_output)
                save_attention_hook_handles.append(handle)
                save_attention_hook_layer_names.append(n)
    return save_attention_hook_handles, save_attention_hook_layer_names

def unregister_save_attention(save_attention_hook_handles):
    for handle in save_attention_hook_handles:
        handle.remove()


# Clone and return only the conditional attention matrices
def clone_attention_matrices(save_output): 
    cloned_matrices = []
    for attn in save_output.attention:
        if attn is not None:
            cloned_matrices.append(attn.clone().detach())
        else:
            cloned_matrices.append(None)
    # print('11111', len(cloned_matrices))
    return cloned_matrices


#simple one to one implementation
def get_tokens(latent_diffusion, source_text, dest_text, source_word_index=None):
    source_tokens = latent_diffusion.cond_stage_models\
    [latent_diffusion.cond_stage_model_metadata["crossattn_flan_t5"]["model_idx"]].get_words_token_mapping(source_text)

    dest_tokens = latent_diffusion.cond_stage_models\
    [latent_diffusion.cond_stage_model_metadata["crossattn_flan_t5"]["model_idx"]].get_words_token_mapping(dest_text)
    print(source_tokens, dest_tokens)

    if source_word_index is None:
        return source_tokens, dest_tokens
    else:
        return [source_tokens[source_word_index]], [dest_tokens[source_word_index]]
    

class EditAttentionMatrices:
    def __init__(self, layer_name, save_attention_hook_layer_names):
        self.layer_name = layer_name
        self.save_attention_hook_layer_names = save_attention_hook_layer_names

    def __call__(self, module, module_in, kwargs):
        attention_weights = None
        
        if 'attention_weights' in kwargs and kwargs['attention_weights'] is not None:
            attention_weights = kwargs['attention_weights']

            if 'reweights' in attention_weights:
                attention_weights['reweights'].cuda()
                # print('In edit', attention_weights['reweights'])

            
            # if attention_weights['type'] == 'reweights':
            #     attention_weights['reweights'] = attention_reweights
    
            # if attention_weights['type'] == 'interpolates':
                # print(self.layer_name, self.save_attention_hook_layer_names)
            timestep = attention_weights['timestep'] #set in the sample_diffusion func
            layer_id = self.save_attention_hook_layer_names.index(self.layer_name)
            # print('timestep = ', timestep, ' | layer_name = ', self.layer_name)

            attention_weights_interpolates = []
            attention_weights_interpolates_idx = []

            source_idxs = attention_weights['source_tokens']
            dest_idxs = attention_weights['target_tokens']

            edit_level = attention_weights['interpolates_mult']

            # cond_timestep_attention_matrices = attention_weights['attention_matrix']
            source_attention_matrices = attention_weights['source_attention_matrices']
            target_attention_matrices = attention_weights['target_attention_matrices']
            for source_idx, dest_idx in zip(source_idxs, dest_idxs):
                final_matrices = None
                # 1. if source_idx is list and dest_idx is not list
                if type(source_idx) == list and type(dest_idx) != list:
                    # print('here1')
                    source_matrices_mean = torch.mean(source_attention_matrices[layer_id][:,:,source_idx[0]:source_idx[-1]+1], dim=-1).cuda()
                    target_matrices = target_attention_matrices[layer_id][:,:,dest_idx].cuda()
                    final_matrices = (1-edit_level) * target_matrices + edit_level * source_matrices_mean

                # 2. if source_idx is not list and dest_idx is list
                if type(source_idx) != list and type(dest_idx) == list:
                    # print('here2')
                    source_matrices = source_attention_matrices[layer_id][:,:,source_idx].cuda()
                    target_matrices = torch.mean(target_attention_matrices[layer_id][:,:,dest_idx[0]:dest_idx[-1]+1], dim=-1).cuda()
                    final_matrices = (1-edit_level) * target_matrices + edit_level * source_matrices_mean
                    
                # 3. if both source_idx and dest_idx are not lists
                if type(source_idx) != list and type(dest_idx) != list:
                    # print('here3')
                    source_matrices = source_attention_matrices[layer_id][:,:,source_idx].cuda()
                    target_matrices = torch.mean(target_attention_matrices[layer_id][:,:,dest_idx], dim=-1).cuda()
                    final_matrices = (1-edit_level) * target_matrices + edit_level * source_matrices_mean

                # 4. if both source_idx and dest_idx are lists
                if type(source_idx) == list and type(dest_idx) == list:
                    # print('here34', source_idx, dest_idx)
                    source_matrices_mean = torch.mean(source_attention_matrices[layer_id][:,:,source_idx[0]:source_idx[-1]+1], dim=-1).cuda()
                    target_matrices = torch.mean(target_attention_matrices[layer_id][:,:,dest_idx[0]:dest_idx[-1]+1], dim=-1).cuda()
                    final_matrices = (1-edit_level) * target_matrices + edit_level * source_matrices_mean
            
                
                attention_weights_interpolates.append(final_matrices)
                attention_weights_interpolates_idx.append(dest_idx)

                
                attention_weights['interpolates'] = attention_weights_interpolates
                attention_weights['interpolates_idx'] = attention_weights_interpolates_idx
                # attention_weights['interpolates_mult'] = 0.9

        kwargs['attention_weights'] = attention_weights
        # print('kwargs::::', kwargs)
        return module_in, kwargs

#kwargs =>{'context': None, 'mask': None, 'attention_weights': None}
    


def register_edit_attention(latent_diffusion, save_attention_hook_layer_names):
    edithook_handles = []
    for n, m in latent_diffusion.named_modules():
        if(isinstance(m, CrossAttention)):
            if 'attn2' in n:
                edit_attention = EditAttentionMatrices(layer_name=n, save_attention_hook_layer_names=save_attention_hook_layer_names)
                handle = m.register_forward_pre_hook(edit_attention, with_kwargs=True) 
                edithook_handles.append(handle)
    return edithook_handles

def unregister_edit_attention(edithook_handles):
    for handle in edithook_handles:
        handle.remove()


def set_attention_weight_for_word(prompt, selected_word_list, value_list, latent_diffusion):
    tokens = latent_diffusion.cond_stage_models[latent_diffusion.cond_stage_model_metadata["crossattn_flan_t5"]["model_idx"]].get_words_token_mapping(prompt) #print the mapping
    context, attn_mask = latent_diffusion.cond_stage_models[0].encode_text(prompt)

    attention_reweights = torch.from_numpy(np.array([1.0 for i in range(context.shape[1])])).float().cuda()

    print(prompt, selected_word_list)
    for ind, word in enumerate(selected_word_list):
        ind_in_prompt = prompt.split(' ').index(word)
        attention_reweights[tokens[ind_in_prompt][0]:tokens[ind_in_prompt][-1]+1] = value_list[ind]
    print(attention_reweights)
    return attention_reweights




def sample_diffusion_attention_edit(latent_diffusion, source_text, target_text, batch_size=1, ddim_steps=20, guidance_scale=3.0, random_seed=42, \
                                    selected_word_list=None, value_list=None, attention_edit_level=0.9,
                                   replace_source_word_id=None):
    # selected_word_list and value_list is reweighting for source_text. Not target_text

    edit_mode = False
    if source_text is not None:
        edit_mode = True
    
    with torch.no_grad():
        seed_everything(int(random_seed))
        x_init = torch.randn((1, 8, 256, 16), device="cuda")

        save_output = SaveAttentionMatrices()

        uncond_dict = {}
        for key in latent_diffusion.cond_stage_model_metadata:
            model_idx = latent_diffusion.cond_stage_model_metadata[key]["model_idx"]
            uncond_dict[key] = latent_diffusion.cond_stage_models[
                model_idx
            ].get_unconditional_condition(batch_size)

        if edit_mode:
            source_cond_batch = make_batch_for_text_to_audio(source_text, transcription="", waveform=None, batchsize=batch_size)
            _, c = latent_diffusion.get_input(source_cond_batch, latent_diffusion.first_stage_key,unconditional_prob_cfg=0.0)  # Do not output unconditional information in the c
            source_cond_dict = latent_diffusion.filter_useful_cond_dict(c)

        target_cond_batch = make_batch_for_text_to_audio(target_text, transcription="", waveform=None, batchsize=batch_size)
        _, c = latent_diffusion.get_input(target_cond_batch, latent_diffusion.first_stage_key,unconditional_prob_cfg=0.0)  # Do not output unconditional information in the c
        target_cond_dict = latent_diffusion.filter_useful_cond_dict(c)

        shape = (latent_diffusion.channels, latent_diffusion.latent_t_size, latent_diffusion.latent_f_size)
        device=latent_diffusion.device
        eta=1.0
        temperature = 1.0
        noise = noise_like(x_init.shape, device, repeat=False) * temperature

        ddim_sampler = DDIMSampler(latent_diffusion, device=device)
        ddim_sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=eta, verbose=False)
        
        timesteps = ddim_sampler.ddim_timesteps

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")
        iterator = stqdm(time_range, desc="DDIM Sampler", total=total_steps)

        attention_weights = {}
        
        if selected_word_list is not None:
            attention_reweights = set_attention_weight_for_word(prompt=target_text, selected_word_list=selected_word_list,\
                                                                value_list=value_list, latent_diffusion=latent_diffusion)
            attention_weights['reweights'] = attention_reweights #reweights array

        if edit_mode:
            source_tokens, target_tokens = get_tokens(latent_diffusion, source_text, target_text, source_word_index=replace_source_word_id)
            print(source_tokens, target_tokens)
            attention_weights['source_tokens'] = source_tokens
            attention_weights['target_tokens'] = target_tokens

        for i, step in enumerate(iterator):
            attention_weights['timestep'] = i

            index = total_steps - i - 1
            t_in = torch.full((batch_size,), step, device=device, dtype=torch.long)

            model_uncond = ddim_sampler.model.apply_model(x_init, t_in, uncond_dict) 
            # clear_attention_matrices(save_output) # we dont need the uncond matrices

            # edithook_handles = None
            if edit_mode:

                save_attention_hook_handles, save_attention_hook_layer_names = register_save_attention(latent_diffusion, save_output)

                model_source_cond = ddim_sampler.model.apply_model(x_init, t_in, source_cond_dict) 
                source_attention_matrices = clone_attention_matrices(save_output) #returns only crossattn layers. Not selfattn.
                clear_attention_matrices(save_output)
    
    
                #First run. Only get attention matrices
                model_target_cond = ddim_sampler.model.apply_model(x_init, t_in, target_cond_dict) 
                target_attention_matrices = clone_attention_matrices(save_output) #returns only crossattn layers. Not selfattn.
                clear_attention_matrices(save_output)
    
                unregister_save_attention(save_attention_hook_handles)
    
    
                # Edit attention
                edithook_handles = register_edit_attention(latent_diffusion, save_attention_hook_layer_names)
    
                attention_weights['source_attention_matrices'] = source_attention_matrices
                attention_weights['target_attention_matrices'] = target_attention_matrices
                
                attention_weights['interpolates_mult'] = attention_edit_level
                model_target_cond = ddim_sampler.model.apply_model(x_init, t_in, target_cond_dict, attention_weights=attention_weights) 
                clear_attention_matrices(save_output)
                unregister_edit_attention(edithook_handles)
    
    
                # CFG; model_output is the estimated error after CFG
                e_t = model_uncond + guidance_scale * (model_target_cond - model_uncond)

            else:
                model_target_cond = ddim_sampler.model.apply_model(x_init, t_in, target_cond_dict, attention_weights=attention_weights) 
                
                # CFG; model_output is the estimated error after CFG
                e_t = model_uncond + guidance_scale * (model_target_cond - model_uncond)

            
        
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
    
        # return mel, waveform, save_attention_hook_handles, edithook_handles
            
        
        
    
