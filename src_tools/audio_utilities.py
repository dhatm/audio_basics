############################################################
#
# Methods useful for analyzing audio data.
#
#   play_audio: plays sound file to listen to
#   plot_waveform: plots amplitude vs time of audio waveform
#   plot_specgram: plots frequency vs time spectrogram of audio waveform (i.e. a short-time fourier transform with sliding window)
#   plot_mel_specgram: plots mel-scale frequency vs time mel-spectrogram of audio waveform (i.e. a short-time fourier transform where frequencies are converted to the mel scale.
#
############################################################

#----------------------------------------------------------
# Imports

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchaudio.transforms as torchTransforms

from IPython.display import Audio, display

#----------------------------------------------------------
# play_audio
#
# Description: Wrapper that handles single- and duo-channel audio. Displays Audio controls to play the audio waveform over speakers. 
#              Only works in a notebook. 
#              
#
# Inputs [I] and outputs [O]:
#   [I] waveform: waveform to play, e.g. from waveform, sample_rate = torchaudio.load(filepath)
#   [I] sample_rate: sample_rate of waveform, e.g. from waveform, sample_rate = torchaudio.load(filepath)
#


def play_audio(waveform, sample_rate):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  if num_channels == 1:
    display(Audio(waveform[0], rate=sample_rate))
  elif num_channels == 2:
    display(Audio((waveform[0], waveform[1]), rate=sample_rate))
  else:
    raise ValueError("Waveform with more than 2 channels are not supported.")


#----------------------------------------------------------
# plot_waveform
#
# Description: displays input waveform time-series data as time vs amplitude
#
# Inputs [I] and outputs [O]:
#   [I] waveform: waveform to play, e.g. from waveform, sample_rate = torchaudio.load(filepath)
#   [I] sample_rate: sample_rate of waveform, e.g. from waveform, sample_rate = torchaudio.load(filepath)
#


def plot_waveform(waveform, sample_rate, title='waveform',xlabel='',ylabel=''):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if ylabel:
            axes[c].set_ylabel(ylabel)
        #elif num_channels > 1:
        #    axes[c].set_ylabel(f"{ylabel}, Channel {c+1}")
        if xlabel:
            axes[c].set_xlabel(xlabel)

    figure.suptitle(title)
    plt.show(block=False)


#----------------------------------------------------------
# plot_specgram
#
# Description: displays spectrogram time-frequency plots of single or duo channel input waveform
# Note: this uses the default psd on a dB scale to diplay the spectrogram
#
# Inputs [I] and outputs [O]:
#   [I] waveform: waveform to play, e.g. from waveform, sample_rate = torchaudio.load(filepath)
#   [I] sample_rate: sample_rate of waveform, e.g. from waveform, sample_rate = torchaudio.load(filepath)
#   [I] n_fft (int, optional): Size of FFT, creates ``NFFT // 2 + 1`` bins. (Default: ``1024``)
#   [I] hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``512 (win_length // 2)``)
#
#   Example
#       >>> waveform, sample_rate = torchaudio.load("test.wav")
#       >>> plot_specgram(waveform, sample_rate=sample_rate)  

def plot_specgram(waveform, sample_rate, title="Spectrogram",xlabel='frequency (Hz)',ylabel='time (sec)',
                  n_fft=1024, hop_length=512):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate, NFFT=n_fft, noverlap=hop_length)
        if ylabel:
            axes[c].set_ylabel(ylabel)
        #if num_channels > 1:
        #    axes[c].set_ylabel(f"Channel {c+1}")
        if xlabel:
            axes[c].set_xlabel(xlabel)
    figure.suptitle(title)
    plt.show(block=False)

#----------------------------------------------------------
# plot_specgram_experimental
#
# Description: displays spectrogram time-frequency plots of audio waveform. This is experimental, to compare with plot_mel_specgram and also plot_specgram (mainly for debugging or future change testing)
#
# Inputs [I] and outputs [O]:
#   [I] waveform: waveform to play, e.g. from waveform, sample_rate = torchaudio.load(filepath)
#   [I] sample_rate (int, optional): sample_rate of waveform, e.g. from waveform, sample_rate = torchaudio.load(filepath) (Default: ``16000``)
#   [I] title (str, optional): title of plot. (Default: ``"Mel Spectrogram"``)
#   [I] xlabel (str, optional): x-axis label. (Default: ``""``)
#   [I] ylabel (str, optional): y-axis label. (Default: ``""``)
#   [I] n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``1024``)
#   [I] win_length (int or None, optional): Window size. (Default: ``None``)
#   [I] hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``512 (win_length // 2)``)
#   [I] power (float, optional): Exponent for the magnitude spectrogram,
#           (must be > 0) e.g., 1 for energy, 2 for power, etc. (Default: ``2``)
#   [I] center (bool, optional): whether to pad :attr:`waveform` on both sides so
#           that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
#           (Default: ``True``)
#   [I] pad_mode (string, optional): controls the padding method used when
#           :attr:`center` is ``True``. (Default: ``"reflect"``)
#   
#   Example
#       >>> waveform, sample_rate = torchaudio.load("test.wav")
#       >>> plot_specgram_experimental(waveform, sample_rate=sample_rate)  
#
def plot_specgram_experimental(waveform, sample_rate=16000, title="Spectrogram",xlabel='Frame',ylabel='Frequency Bin',
                  n_fft=1024, win_length=None,hop_length=512,center=True,pad_mode="reflect",
                  power=2.0):

    # Define transform
    spectrogram = torchTransforms.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=center,
        pad_mode=pad_mode,
        power=power,
    )

    spec = spectrogram(waveform)

    num_channels, num_frames = waveform.shape
    #figure, axes = plt.subplots(num_channels, 1) #to do: more than 1 channel audio
    figure, axes = plt.subplots(1, 1)
    
    plt.imshow(10*np.log10(spec[0]), origin="lower", aspect="auto")
    figure.suptitle(title)
    axes.set_ylabel(ylabel)
    axes.set_xlabel(xlabel)
    #figure.colorbar(im, ax=axes)
    plt.show(block=False)



#----------------------------------------------------------
# plot_mel_specgram
#
# Description: displays mel-scale frequency bin vs spectrogram-frame bin mel-spectrogram of audio waveform (i.e. a short-time fourier transform where frequencies are converted to the mel scale. Mel Scale is a logarithmic transformation of a signal's frequency. 
#
# Insights: This better mimics filters of the human ear and also makes it easier to resolve many Harmonic Complex Tones (HCTs) that wiggle around in frequency. HCTs are sounds in which frequency components are multiples of a common fundamental frequency (f0). These are ubiquitous in speech, music, reciprocating machinery, and animal vocalization. HCTs tend to move around in frequency. Hence a purturbation of f0 --> f0+del becomes a perturbation at the nth harmonic of fn=n*f0 --> fn'=n*(f0+del)=n*f0 + n*del=fn+n*del. On a regular spectrogram, this n*del extra jump results in very steep jumps at high frequency. These are blurred because of the spectrogram's constant frequency resolution. If frequency resolution is changed to resolve steep jumps, then temporal resolution suffers and lower frequency harmonics suffer, often beating together on the spectrogram. On a mel-scale however, log(f0) --> log(f0+del) and log(fn) --> log(n*(f0+del) = n*log(f0+del). This results in a more stable slope at each harmonic which can be resolved better for all frequencies at the proper resolution. Unfortunately, at high frequencies, the mel-scale compresses HCTs together and they can still become unresolved.  Hence mel-scale filtering (on its own) may not be optimal for all audio problems, especially if frequency resolution at high frequency is important. My intuition is that there is likely a piece of the physiology puzzle that we are missing or misunderstanding -- perhaps in the lateral lemniscus.
#
# Inputs [I] and outputs [O]:
#   [I] waveform: waveform to play, e.g. from waveform, sample_rate = torchaudio.load(filepath)
#   [I] sample_rate (int, optional): sample_rate of waveform, e.g. from waveform, sample_rate = torchaudio.load(filepath) (Default: ``16000``)
#   [I] title (str, optional): title of plot. (Default: ``"Mel Spectrogram"``)
#   [I] xlabel (str, optional): x-axis label. (Default: ``""``)
#   [I] ylabel (str, optional): y-axis label. (Default: ``""``)
#   [I] n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``1024``)
#   [I] win_length (int or None, optional): Window size. (Default: ``None``)
#   [I] hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``512 (win_length // 2)``)
#   [I] n_mels (int, optional): Number of mel filterbanks. (Default: ``128``)
#   [I] power (float, optional): Exponent for the magnitude spectrogram,
#           (must be > 0) e.g., 1 for energy, 2 for power, etc. (Default: ``2``)
#   [I] center (bool, optional): whether to pad :attr:`waveform` on both sides so
#           that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
#           (Default: ``True``)
#   [I] pad_mode (string, optional): controls the padding method used when
#           :attr:`center` is ``True``. (Default: ``"reflect"``)
#   [I] norm (str or None, optional): If "slaney", divide the triangular mel weights by the width of the mel band
#           (area normalization). (Default: ``"slaney"``)
#   [I] mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)
#   
#   Example
#       >>> waveform, sample_rate = torchaudio.load("test.wav")
#       >>> plot_mel_specgram(waveform, sample_rate=sample_rate)  
#
#
#

def plot_mel_specgram(waveform, sample_rate=16000, title="Mel Spectrogram",xlabel='frame bin',ylabel='mel-frequency bin', 
                      n_fft=1024, win_length=None,hop_length=512,center=True,pad_mode="reflect",
                      power=2.0,norm="slaney",onesided=True,n_mels=128,mel_scale="htk"):

    mel_spectrogram = torchTransforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=center,
        pad_mode=pad_mode,
        power=power,
        norm=norm,
        n_mels=n_mels,
        mel_scale=mel_scale,
    )

    melspec = mel_spectrogram(waveform)

    num_channels, num_frames = waveform.shape
    #figure, axes = plt.subplots(num_channels, 1) #to do: more than 1 channel audio
    figure, axes = plt.subplots(1, 1)
    
    plt.imshow(10*np.log10(melspec[0]), origin="lower", aspect="auto")
    figure.suptitle(title)
    axes.set_ylabel(ylabel)
    axes.set_xlabel(xlabel)
    plt.show(block=False)

    
#----------------------------------------------------------
#  plot_mel_specgram_vs_time 
#
# Description: same thing as plot_mel_specgram but plots vs time instead of vs spectrogram frame bin
#
# Inputs [I] and outputs [O]: see plot_mel_specgram for complete list. only additional inputs are listed here
#   [I] secs_label_separation: number of seconds between spectrogram xtick labels (Default: 0.5 sec)
#
#   Example
#       >>> waveform, sample_rate = torchaudio.load("test.wav")
#       >>> plot_mel_specgram_vs_time(waveform, sample_rate=sample_rate,secs_label_separation=.33)  

def plot_mel_specgram_vs_time(waveform, sample_rate=16000, title="Mel Spectrogram",xlabel='time (s)',
                              ylabel='mel frequency bin', n_fft=1024, 
                              win_length=None,hop_length=512,center=True,pad_mode="reflect",
                              power=2.0,norm="slaney",onesided=True,n_mels=128,mel_scale="htk",
                              secs_label_separation=.5):

    mel_spectrogram = torchTransforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=center,
        pad_mode=pad_mode,
        power=power,
        norm=norm,
        n_mels=n_mels,
        mel_scale=mel_scale,
    )

    melspec = mel_spectrogram(waveform)
    melspec_dB = 10*np.log10(melspec[0]) #10*log10 of PSD, since PSD is power


    n_mel_freq_bins = melspec.shape[1] #number of mel-frequency bins
    n_frames = melspec.shape[2] #number spectrogram frames (not audio clip samples)
    #nyquist_frequency = sample_rate / 2

    # find time info
    num_channels, num_samples = waveform.shape #single or duo channel audio, number of samples in audio clip
    time_duration = num_samples/sample_rate #find total audio clip duration in seconds


    #apply time labels (convert from frame bins to seconds)
    n_xticks = int(time_duration / secs_label_separation + 1) #number of ticks to add with secs_label_separation separation
    last_tick_in_secs = (n_xticks-1)*secs_label_separation #note: last tick could be before end of waveform

    xtick_time_labels = np.linspace(0, last_tick_in_secs, n_xticks)
    xtick_time_labels = np.array([round(x,2) for x in xtick_time_labels]) #make sure labels don't get too long

    delta_t = time_duration / n_frames #time (secs) between spectrogram frames (not audio clip samples)
    x_ticklocs =  xtick_time_labels / delta_t #bin locations of where to apply frequency labels


    #make plot
    #figure, axes = plt.subplots(num_channels, 1) #to do: more than 1 channel audio
    figure, axes = plt.subplots(1, 1)

    plt.imshow(melspec_dB, origin="lower", aspect="auto")
    plt.xticks(x_ticklocs,xtick_time_labels)
    #plt.yticks(y_ticklocs,ytick_melfrequency_labels) #to do?: something else for yticks is possible here
    figure.suptitle(title)
    axes.set_ylabel(ylabel)
    axes.set_xlabel(xlabel)

    plt.show(block=False)

