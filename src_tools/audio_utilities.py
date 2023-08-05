############################################################
#
# Methods useful for analyzing audio data.
#
#   play_audio: plays sound file to listen to
#   plot_waveform: plots amplitude vs time of audio waveform
#   plot_specgram: plots frequency vs time spectrogram of audio waveform (i.e. a short-time fourier transform with 
#     sliding window)
#   plot_mel_specgram: plots mel-scale frequency vs time mel-spectrogram of audio waveform (i.e. a short-time 
#     fourier transform where frequencies are converted to the mel scale.
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
# Description: Wrapper that handles single- and duo-channel audio. Displays Audio controls to play the audio 
#   waveform over speakers. Only works in a notebook. 
#              
#
# Inputs [I] and outputs [O]:
#   [I] waveform: waveform to play, e.g. from waveform, sample_rate = torchaudio.load(filepath)
#   [I] sample_rate: sample_rate of waveform, e.g. from waveform, sample_rate = torchaudio.load(filepath)
#


def play_audio(waveform, sample_rate):
    """play_audio
       Description: Wrapper that handles single- and duo-channel audio. Displays Audio controls to play the audio 
          waveform over speakers. Only works in a notebook. 
  
       Args:
          waveform: waveform to play, e.g. from waveform, sample_rate = torchaudio.load(filepath)
          sample_rate: sample_rate of waveform, e.g. from waveform, sample_rate = torchaudio.load(filepath)
     
       Returns:
          None
    """


    waveform = waveform.numpy()
  
    num_channels, num_frames = waveform.shape
    if num_channels == 1:
      display(Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
      #display(Audio((waveform[0], waveform[1]), rate=sample_rate))
      display(Audio(waveform[0], rate=sample_rate))
      display(Audio(waveform[1], rate=sample_rate))
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
    """plot_waveform
       Description: displays input waveform time-series data as time vs amplitude
      
       Args:
          waveform: waveform to play, e.g. from waveform, sample_rate = torchaudio.load(filepath)
          sample_rate: sample_rate of waveform, e.g. from waveform, sample_rate = torchaudio.load(filepath)
     
       Returns:
          None
    """


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
#  amplitude_to_dB 
#
# Description: Converts an amplitude (e.g., a spectrogram) to decibel (dB) units
# 
def amplitude_to_dB(Amp, amin: float = 1e-10):
    """Converts an amplitude (e.g., a spectrogram) to decibel (dB) units

    This computes the scaling ``20 * log10(S)`` in a numerically stable way.

    Args:
        Amp: amplitude signal, e.g., a spectrogram
        amin: float > 0 [scalar].  minimum threshold for ``abs(S)`` 

    Returns:
        Amp_dB : ``Amp_db ~= 20 * log10(Amp)`` or equivalently ``Amp_db ~= 10 * log10(Amp**2)``

    Example:
        $ Amp_dB = amplitude_to_dB(Amp)
    
    """
    Amp = np.asarray(Amp)
    
    if amin <= 0:
        raise ParameterError("amin must be strictly positive")
        
    if np.issubdtype(Amp.dtype, np.complexfloating):
        warnings.warn(
            "power_to_db was called on complex input so phase "
            "information will be discarded. To suppress this warning, "
            "call power_to_db(np.abs(D)**2) instead.",
            stacklevel=2,
        )
        magnitude = np.abs(Amp)
    else:
        magnitude = Amp

    Amp_dB = 20.0 * np.log10(np.maximum(amin, magnitude))

    return Amp_dB

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
#   [I] max_channels_show: max number of channels to plot. (Default: ``1``)
#   [I] cmap: colormap to use (Default: ``'viridis'``). 'magma' is a popular one, and I like 'twilight_shifted'
#
#   Example
#       >>> waveform, sample_rate = torchaudio.load("test.wav")
#       >>> plot_specgram(waveform, sample_rate=sample_rate)  

def plot_specgram(waveform, sample_rate, title="Spectrogram",xlabel='time (sec)',ylabel='frequency (Hz)',
                  n_fft=2048, hop_length=1024, max_channels_show=1, cmap='viridis'):
    """plot_specgram
        Description: displays spectrogram time-frequency plots of single or duo channel input waveform
        Note: this uses the default psd on a dB scale to diplay the spectrogram
       
       Args:
          waveform: waveform to play, e.g. from waveform, sample_rate = torchaudio.load(filepath)
          sample_rate: sample_rate of waveform, e.g. from waveform, sample_rate = torchaudio.load(filepath)
          n_fft (int, optional): Size of FFT, creates ``NFFT // 2 + 1`` bins. (Default: ``2048``)
          hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``1024 (win_length//2)``)
          xlabel: label for x-axis (Default: ``'time (sec)'``)
          ylabel: label for y-axis (Default: ``'frequency (Hz)'``)
          max_channels_show: max number of channels to plot. (Default: ``1``)
          cmap: colormap to use (Default: ``'viridis'``). 'magma' is a popular one, and I like 'twilight_shifted'
       
       Example
           $ waveform, sample_rate = torchaudio.load("test.wav")
           $ plot_specgram(waveform, sample_rate=sample_rate)  

     
       Returns:
          None
    """


    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    nplots = min(max_channels_show,num_channels)

    #figure, axes = plt.subplots(nplots, 1)
    figure, axes = plt.subplots(nplots, 1,figsize=(10, 4))
    if nplots == 1:
        axes = [axes]
    for c in range(nplots):
        axes[c].specgram(waveform[c], Fs=sample_rate, NFFT=n_fft, noverlap=(n_fft-hop_length), cmap=cmap)
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
# Description: displays spectrogram time-frequency plots of audio waveform. This is experimental, to compare with 
#    plot_mel_specgram and also plot_specgram (mainly for debugging or future change testing)
#
# Inputs [I] and outputs [O]:
#   [I] waveform: waveform to play, e.g. from waveform, sample_rate = torchaudio.load(filepath)
#   [I] sample_rate (int, optional): sample_rate of waveform, e.g. from waveform, sample_rate = 
#         torchaudio.load(filepath) (Default: ``16000``)
#   [I] title (str, optional): title of plot. (Default: ``"Mel Spectrogram"``)
#   [I] xlabel (str, optional): x-axis label. (Default: ``""``)
#   [I] ylabel (str, optional): y-axis label. (Default: ``""``)
#   [I] n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``1024``)
#   [I] win_length (int or None, optional): Window size. (Default: ``None``)
#   [I] hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``512 (win_length // 2)``)
#   [I] power (float, optional): Exponent for the magnitude spectrogram,
#         (must be > 0) e.g., 1 for energy, 2 for power, etc. (Default: ``2``)
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
                  n_fft=2048, win_length=None,hop_length=1024,center=True,pad_mode="reflect",
                  power=2.0, cmap='viridis'):
    """plot_specgram_experimental

       Description: displays spectrogram time-frequency plots of audio waveform. This is experimental, to compare 
          with plot_mel_specgram and also plot_specgram (mainly for debugging or future change testing)
       
       Args:
          waveform: waveform to play, e.g. from waveform, sample_rate = torchaudio.load(filepath)
          sample_rate (int, optional): sample_rate of waveform, e.g. from waveform, sample_rate = 
            torchaudio.load(filepath) (Default: ``16000``)
          title (str, optional): title of plot. (Default: ``"Mel Spectrogram"``)
          xlabel (str, optional): x-axis label. (Default: ``""``)
          ylabel (str, optional): y-axis label. (Default: ``""``)
          n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``2048``)
          win_length (int or None, optional): Window size. (Default: ``None`` i.e., n_fft)
          hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``1024 (win_length//2)``)
          power (float, optional): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for energy, 2 for power, etc. (Default: ``2``)
          center (bool, optional): whether to pad :attr:`waveform` on both sides so
              that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
              (Default: ``True``)
          pad_mode (string, optional): controls the padding method used when
                  :attr:`center` is ``True``. (Default: ``"reflect"``)
          cmap: colormap to use (Default: ``'viridis'``). 'magma' is a popular one, and I like 'twilight_shifted'

       Example
           $ waveform, sample_rate = torchaudio.load("test.wav")
           $ plot_specgram_experimental(waveform, sample_rate=sample_rate)  
       
       Returns:
          None
    """



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
    #figure, axes = plt.subplots(1, 1)
    figure, axes = plt.subplots(1, 1,figsize=(10, 4))
    plt.imshow(amplitude_to_dB(spec[0]), origin="lower", aspect="auto", cmap=cmap)    
    #spec_dB = torchTransforms.AmplitudeToDB(top_db=80, stype = "amplitude")(spec[0]) #another option is to use this
    #plt.imshow(spec_dB, origin="lower", aspect="auto", cmap=cmap)
    figure.suptitle(title)
    axes.set_ylabel(ylabel)
    axes.set_xlabel(xlabel)
    #figure.colorbar(im, ax=axes)
    plt.show(block=False)



#----------------------------------------------------------
#  power_to_dB 
#
# Description: Converts a power spectrogram (amplitude squared) to decibel (dB) units
# 
def power_to_dB(PowerSpec, amin: float = 1e-10):
    """Convert a power spectrogram (amplitude squared) to decibel (dB) units

    This computes the scaling ``10 * log10(S)`` in a numerically stable way.

    Args:
        PowerSpec: power spectrogram
        amin: float > 0 [scalar].  minimum threshold for ``abs(S)`` 

    Returns:
        PowerSpec_dB : ``PowerSpec_db ~= 10 * log10(PowerSpec)``

    Example:
        $ PowerSpec_dB = power_to_dB(PowerSpec)
    
    """
    PowerSpec = np.asarray(PowerSpec)
    
    if amin <= 0:
        raise ParameterError("amin must be strictly positive")
        
    if np.issubdtype(PowerSpec.dtype, np.complexfloating):
        warnings.warn(
            "power_to_db was called on complex input so phase "
            "information will be discarded. To suppress this warning, "
            "call power_to_db(np.abs(D)**2) instead.",
            stacklevel=2,
        )
        magnitude = np.abs(PowerSpec)
    else:
        magnitude = PowerSpec

    PowerSpec_dB = 10.0 * np.log10(np.maximum(amin, magnitude))

    return PowerSpec_dB


#----------------------------------------------------------
# plot_mel_specgram
#
# Description: displays mel-scale frequency bin vs spectrogram-frame bin mel-spectrogram of audio waveform 
#   (i.e. a short-time fourier transform where frequencies are converted to the mel scale. Mel Scale is a 
#   logarithmic transformation of a signal's frequency). 
#
# Insights: This somewhat mimics the human ear and also makes it easier to resolve many Harmonic Complex 
#   Tones (HCTs) that wiggle around in frequency. HCTs are sounds in which frequency components are multiples 
#   of a common fundamental frequency (f0). These are ubiquitous in speech, music, reciprocating machinery, 
#   and animal vocalization. HCTs tend to move around in frequency. Hence a purturbation of f0 --> f0+del 
#   becomes a perturbation at the nth harmonic of fn=n*f0 --> fn'=n*(f0+del)=n*f0 + n*del=fn+n*del. On a 
#   regular spectrogram, this n*del extra jump results in very steep jumps at high frequency. These are blurred 
#   because of the spectrogram's constant frequency resolution. If frequency resolution is changed to resolve 
#   steep jumps, then temporal resolution suffers and lower frequency harmonics suffer, often beating together 
#   on the spectrogram. On a mel-scale however, log(f0) --> log(f0+del) and log(fn) --> log(n*(f0+del) = 
#   n*log(f0+del). This results in a more stable slope at each harmonic which can be resolved better for all 
#   frequencies at the proper resolution. Unfortunately, at high frequencies, the mel-scale compresses HCTs 
#   together and they can still become unresolved.  Hence mel-scale filtering (on its own) may not be optimal 
#   for audio problems, especially if frequency resolution at high frequency is important. My intuition is that 
#   there is likely a piece of the physiology puzzle that we are missing or misunderstanding -- perhaps in the 
#   lateral lemniscus.
#
# Inputs [I] and outputs [O]:
#   [I] waveform: waveform to play, e.g. from waveform, sample_rate = torchaudio.load(filepath)
#   [I] sample_rate (int, optional): sample_rate of waveform, e.g. from waveform, sample_rate = 
#       torchaudio.load(filepath) (Default: ``16000``)
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
                      n_fft=2048, win_length=None,hop_length=1024,center=True,pad_mode="reflect",
                      power=2.0,norm="slaney",onesided=True,n_mels=128,mel_scale="htk", cmap='viridis'):
    """plot_mel_specgram
      
       Description: displays mel-scale frequency bin vs spectrogram-frame bin mel-spectrogram of audio waveform 
         (i.e. a short-time fourier transform where frequencies are converted to the mel scale. Mel Scale is a 
         logarithmic transformation of a signal's frequency. 
      
       Insights: This somewhat mimics the human ear and also makes it easier to resolve many Harmonic Complex 
         Tones (HCTs) that wiggle around in frequency. HCTs are sounds in which frequency components are multiples 
         of a common fundamental frequency (f0). These are ubiquitous in speech, music, reciprocating machinery, 
         and animal vocalization. HCTs tend to move around in frequency. Hence a purturbation of f0 --> f0+del 
         becomes a perturbation at the nth harmonic of fn=n*f0 --> fn'=n*(f0+del)=n*f0 + n*del=fn+n*del. On a 
         regular spectrogram, this n*del extra jump results in very steep jumps at high frequency. These are blurred 
         because of the spectrogram's constant frequency resolution. If frequency resolution is changed to resolve 
         steep jumps, then temporal resolution suffers and lower frequency harmonics suffer, often beating together 
         on the spectrogram. On a mel-scale however, log(f0) --> log(f0+del) and log(fn) --> log(n*(f0+del) = 
         n*log(f0+del). This results in a more stable slope at each harmonic which can be resolved better for all 
         frequencies at the proper resolution. Unfortunately, at high frequencies, the mel-scale compresses HCTs 
         together and they can still become unresolved.  Hence mel-scale filtering (on its own) may not be optimal 
         for audio problems, especially if frequency resolution at high frequency is important. My intuition is that 
         there is likely a piece of the physiology puzzle that we are missing or misunderstanding -- perhaps in the 
         lateral lemniscus.
      
       Args:
         waveform: waveform to play, e.g. from waveform, sample_rate = torchaudio.load(filepath)
         sample_rate (int, optional): sample_rate of waveform, e.g. from waveform, sample_rate = 
         torchaudio.load(filepath) (Default: ``16000``)
         title (str, optional): title of plot. (Default: ``"Mel Spectrogram"``)
         xlabel (str, optional): x-axis label. (Default: ``""``)
         ylabel (str, optional): y-axis label. (Default: ``""``)
         n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``2048``)
         win_length (int or None, optional): Window size. (Default: ``None`` i.e. n_fft)
         hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``1024 (win_length//2)``)
         n_mels (int, optional): Number of mel filterbanks. (Default: ``128``)
         power (float, optional): Exponent for the magnitude spectrogram,
             (must be > 0) e.g., 1 for energy, 2 for power, etc. (Default: ``2``)
         center (bool, optional): whether to pad :attr:`waveform` on both sides so
             that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
             (Default: ``True``)
         pad_mode (string, optional): controls the padding method used when
             :attr:`center` is ``True``. (Default: ``"reflect"``)
         norm (str or None, optional): If "slaney", divide the triangular mel weights by the width of the mel band
             (area normalization). (Default: ``"slaney"``)
         mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)
         cmap: colormap to use (Default: ``'viridis'``). 'magma' is a popular one, and I like 'twilight_shifted'
         
       Example
           $ waveform, sample_rate = torchaudio.load("test.wav")
           $ plot_mel_specgram(waveform, sample_rate=sample_rate)  
      
       Returns:
          None
    """



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
    #melspec_dB = power_to_dB(melspec[0]) #make into dB: 10*log10 of PSD, since PSD is power
    melspec_dB = torchTransforms.AmplitudeToDB(top_db=80, stype = "power")(melspec[0]) #another option is to use this



    num_channels, num_frames = waveform.shape
    #figure, axes = plt.subplots(num_channels, 1) #to do: more than 1 channel audio
    #figure, axes = plt.subplots(1, 1)
    figure, axes = plt.subplots(1, 1,figsize=(10, 4))
    
    plt.imshow(melspec_dB, origin="lower", aspect="auto", cmap=cmap)
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
                              ylabel='mel frequency bin', n_fft=2048, 
                              win_length=None,hop_length=1024,center=True,pad_mode="reflect",
                              power=2.0,norm="slaney",onesided=True,n_mels=128,mel_scale="htk",
                              cmap='viridis', secs_label_separation=.5):
    """plot_mel_specgram_vs_time
      
       Description: same thing as plot_mel_specgram but plots vs time instead of vs spectrogram frame bin
      
       Args:
         see plot_mel_specgram for complete list. only additional inputs are listed here
         secs_label_separation: number of seconds between spectrogram xtick labels (Default: ``0.5 sec``)
      
       Example
           $ waveform, sample_rate = torchaudio.load("test.wav")
           $ plot_mel_specgram_vs_time(waveform, sample_rate=sample_rate,secs_label_separation=.33)  

       Returns:
          None
    """



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
    #melspec_dB = power_to_dB(melspec[0]) #make into dB: 10*log10 of PSD, since PSD is power
    melspec_dB = torchTransforms.AmplitudeToDB(top_db=80, stype = "power")(melspec[0]) #another option is to use this


    n_mel_freq_bins = melspec.shape[1] #number of mel-frequency bins
    n_frames = melspec.shape[2] #number spectrogram frames (not audio clip samples)
    #nyquist_frequency = sample_rate / 2

    # find time info
    num_channels, num_samples = waveform.shape #single or duo channel audio, number of samples in audio clip
    time_duration = num_samples/sample_rate #find total audio clip duration in seconds


    #apply time labels (convert from frame bins to seconds)
    n_xticks = int(time_duration / secs_label_separation + 1) #num ticks to add with secs_label_separation separation
    last_tick_in_secs = (n_xticks-1)*secs_label_separation #note: last tick could be before end of waveform

    xtick_time_labels = np.linspace(0, last_tick_in_secs, n_xticks)
    xtick_time_labels = np.array([round(x,2) for x in xtick_time_labels]) #make sure labels don't get too long

    delta_t = time_duration / n_frames #time (secs) between spectrogram frames (not audio clip samples)
    x_ticklocs =  xtick_time_labels / delta_t #bin locations of where to apply frequency labels


    #make plot
    #figure, axes = plt.subplots(num_channels, 1) #to do: more than 1 channel audio
    #figure, axes = plt.subplots(1, 1)
    figure, axes = plt.subplots(1, 1,figsize=(10, 4))
    

    plt.imshow(melspec_dB, origin="lower", aspect="auto", cmap=cmap)
    plt.xticks(x_ticklocs,xtick_time_labels)
    #plt.yticks(y_ticklocs,ytick_melfrequency_labels) #to do?: something else for yticks is possible here
    figure.suptitle(title)
    axes.set_ylabel(ylabel)
    axes.set_xlabel(xlabel)

    plt.show(block=False)
    
    
#----------------------------------------------------------
# hz_to_mel
# 
# see _hz_to_mel https://github.com/pytorch/audio/blob/main/torchaudio/functional/functional.py#L422

def hz_to_mel(freq: float, mel_scale: str = "htk") -> float:
    """Convert Hz to mel(-frenquency).

    Args:
        freqs (float): Frequencies in Hz
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

    Returns:
        mels (float): Frequency in Mels
    """

    if mel_scale not in ["slaney", "htk"]:
        raise ValueError('mel_scale should be one of "htk" or "slaney".')

    if mel_scale == "htk":
        return 2595.0 * np.log10(1.0 + (freq / 700.0))

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (freq - f_min) / f_sp

    # Fill in the log-scale part
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0

    if freq >= min_log_hz:
        mels = min_log_mel + np.log(freq / min_log_hz) / logstep

    return mels

#----------------------------------------------------------
# mel_to_hz
#
# see _mel_to_hz https://github.com/pytorch/audio/blob/main/torchaudio/functional/functional.py#L456

def mel_to_hz(mels, mel_scale: str = "htk"):
    """Convert mel(-frequency) to frequency.

    Args:
        mels (Tensor): Mel frequencies
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

    Returns:
        freqs (Tensor): Mels converted in Hz
    """

    if mel_scale not in ["slaney", "htk"]:
        raise ValueError('mel_scale should be one of "htk" or "slaney".')

    if mel_scale == "htk":
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0

    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hz * torch.exp(logstep * (mels[log_t] - min_log_mel))

    return freqs

#----------------------------------------------------------
# find_first_idx_greater_than_x
#
def find_first_idx_greater_than_x(tensor_array,val):
    """
    Finds the first index with value greater than val in a tensor array of growing numbers.
    Args:
        tensor_array: The PyTorch tensor array of growing numbers.
        val: value to search for or exceed
    Returns: 
        The first index with value greater than val, or None if no such number exists.
        """
    for i in range(len(tensor_array)):
        if tensor_array[i] >= val:
            return i
    return None


#----------------------------------------------------------
# plot_mel_specgram_hz_vs_time
#
# Description: same thing as plot_mel_specgram_vs_time but plots vs frequency instead of vs mel filter number 
#   (aka bin)
#
# Inputs [I] and outputs [O]: see plot_mel_specgram_vs_time for complete list. only additional inputs are listed 
#   here
#   [I] smallest_freq: smallest frequency to show on mel-spectrogram (Default: 500 Hz)
#
#   Example
#       >>> waveform, sample_rate = torchaudio.load("test.wav")
#       >>> plot_mel_specgram_hz_vs_time(waveform, sample_rate=sample_rate,smallest_freq=500) 
#
# To Do: clean this code up a bit more
#

def plot_mel_specgram_hz_vs_time(waveform, sample_rate=16000, title="Mel Spectrogram",xlabel='time (s)',
                                 ylabel='frequency (Hz)', n_fft=2048, 
                                 win_length=None,hop_length=1024,center=True,pad_mode="reflect",
                                 power=2.0,norm="slaney",onesided=True,n_mels=128,mel_scale="htk",
                                 cmap='viridis', secs_label_separation=.5, smallest_freq=500):

    """plot_mel_specgram_hz_vs_time

       Description: same thing as plot_mel_specgram_vs_time but plots vs frequency instead of vs mel filter number  
       
        
       Args:
           see plot_mel_specgram_vs_time for complete list. only additional inputs are listed here
           smallest_freq: smallest frequency to show on mel-spectrogram, i.e. shows from this to 
             nyquist_frequency (Default: 500 Hz)
       
       Example
           $ waveform, sample_rate = torchaudio.load("test.wav")
           $ plot_mel_specgram_hz_vs_time(waveform, sample_rate=sample_rate,smallest_freq=500) 

       Returns:
          None
    """

    #------------------------
    #compute mel-spectra
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
    #melspec_dB = power_to_dB(melspec[0]) #make into dB: 10*log10 of PSD, since PSD is power
    melspec_dB = torchTransforms.AmplitudeToDB(top_db=80, stype = "power")(melspec[0]) #another option is to use this
    
    #------------------------
    # find ancillary parameters to use below
    #n_mel_freq_bins = melspec.shape[1] #number of mel-frequency bins = n_mels
    n_frames = melspec.shape[2] #number spectrogram frames (not audio clip samples)
    nyquist_frequency = sample_rate / 2
    
    #------------------------
    # find time x-axis info
    num_channels, num_samples = waveform.shape #single or duo channel audio, number of samples in audio clip
    time_duration = num_samples/sample_rate #find total audio clip duration in seconds
    
    #apply time labels (convert from frame bins to seconds)
    n_xticks = int(time_duration / secs_label_separation + 1) #num ticks to add with secs_label_separation separation
    last_tick_in_secs = (n_xticks-1)*secs_label_separation #note: last tick could be before end of waveform
    
    xtick_time_labels = np.linspace(0, last_tick_in_secs, n_xticks) 
    xtick_time_labels = np.array([round(x,2) for x in xtick_time_labels]) #make sure labels don't get too long

    delta_t = time_duration / n_frames #time (secs) between spectrogram frames (not audio clip samples)
    x_ticklocs =  xtick_time_labels / delta_t #bin locations of where to apply frequency labels
    
    #------------------------
    #find frequency y-axis info
    #make frequency labels
    ftmp=nyquist_frequency #temporary frequency
    #smallest_freq=500 #smallest frequency to show on plot
    ytick_frequency_labels = []
    while ftmp >= smallest_freq:
        ytick_frequency_labels.append(ftmp)
        ftmp = ftmp / 2 #divide by 2 each time
    ytick_frequency_labels.reverse() #reverse to make list smallest to largest
    
    #find mel-frequency (mel) and frequency (hz) arrays -- used for location calculations next
    m_min = hz_to_mel(0) #mel min
    m_max = hz_to_mel(nyquist_frequency) #mel max
    m_pts = torch.linspace(m_min, m_max, n_mels + 2) #mel-frequency (mel) array
    f_pts = mel_to_hz(m_pts, mel_scale=mel_scale) #frequency (Hz) array
    
    #find location to put ytick labels
    # f_pts frequency-bin corresponds to m_pts mel-bin, which equals mel filter number in plot's y-axis
    ylabel_melbins = [find_first_idx_greater_than_x(f_pts, x) for x in ytick_frequency_labels]
    ylabel_melbins[-1]=n_mels#fix last bin if needed
    
    #finalize
    y_ticklocs = np.array(ylabel_melbins)
    ytick_labels = np.array(ytick_frequency_labels)
    
    #------------------------
    #make plot
    
    #figure, axes = plt.subplots(num_channels, 1) #to do: more than 1 channel audio
    #figure, axes = plt.subplots(1, 1)
    figure, axes = plt.subplots(1, 1,figsize=(10, 4))

    plt.imshow(melspec_dB, origin="lower", aspect="auto", cmap=cmap)
    plt.xticks(x_ticklocs,xtick_time_labels)
    plt.yticks(y_ticklocs,ytick_labels)
    figure.suptitle(title)
    axes.set_ylabel(ylabel)
    axes.set_xlabel(xlabel)

    plt.show(block=False)
    
