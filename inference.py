#!/usr/bin/env python
# coding: utf-8

# ## Tacotron 2 inference code 
# Edit the variables **checkpoint_path** and **text** to match yours and run the entire code to generate plots of mel outputs, alignments and audio synthesis from the generated mel-spectrogram using Griffin-Lim.

# #### Import libraries and setup matplotlib

# In[1]:
    
    
import matplotlib
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pylab as plt

# import IPython.display as ipd

import sys
sys.path.append('waveglow/')
import numpy as np
import torch
import scipy.io.wavfile

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser


# In[2]:


def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom', 
                       interpolation='none')


# #### Setup hparams

# In[3]:


hparams = create_hparams()
hparams.sampling_rate = 22050


# #### Load model from checkpoint

# In[4]:


checkpoint_path = "tacotron2_statedict.pt"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()


# #### Load WaveGlow for mel2audio synthesis and denoiser

# In[5]:


waveglow_path = 'waveglow_256channels.pt'
waveglow = torch.load(waveglow_path)['model']
waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)


# #### Prepare text input

# In[6]:


text = "Hello! I'm here to kill all humans."
sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
sequence = torch.autograd.Variable(
    torch.from_numpy(sequence)).cuda().long()


# #### Decode text input and plot results

# In[7]:


mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
plot_data((mel_outputs.float().data.cpu().numpy()[0],
           mel_outputs_postnet.float().data.cpu().numpy()[0],
           alignments.float().data.cpu().numpy()[0].T))


# #### Synthesize audio from spectrogram using WaveGlow

# In[8]:


with torch.no_grad():
    audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
# noisy = audio.cpu().numpy()
# noisy = noisy.astype('float32')
# scipy.io.wavfile.write("audio_file.wav", hparams.sampling_rate, noisy.T)

# #### (Optional) Remove WaveGlow bias

# In[9]:


audio_denoised = denoiser(audio, strength=0.01)[:, 0]
arr = audio_denoised.cpu().numpy()
scipy.io.wavfile.write("audio_file_denoised.wav", hparams.sampling_rate, arr.T)
 
