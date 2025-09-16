#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 11:49:14 2025

@author: jlab
"""

import numpy as np
import math
import scipy.signal as signal
from scipy.io import wavfile

class FMRadioReceiver:
    """
    A class for FM radio signal reception and demodulation.
    """

    def __init__(self, sample_rate: int, lo_offset: float=1.6e6) -> None:
        """
        class initializer

        Parameters
        ----------
        sample_rate : int
            sample rate of the receive used to capture the data.
        lo_offset : float
            sets the frequency shift value in Hz to move LO out of band

        Returns
        -------
        None

        """
        self.sample_rate = sample_rate
        self.frequency_max = 75e3
        self.audio_rate = 48e3        
        
        self.lo_offset = lo_offset
        
        self.dec_factor  = 10
        self.dec_numtaps = 1927
        self.dec_width   = 25e3
        self.dec_cutoff  = 75e3
        self.dec_rate    = self.audio_rate*self.dec_factor

        self.audio_numtaps = 2048
        self.audio_width   = self.dec_rate/32
        self.audio_cutoff  = self.dec_rate/2 - self.audio_width
        
        
        
    def load_iq_data(self, file_name: str) -> np.ndarray:
        """
        load_iq_data reads IQ samples from a GNU radio file
        
        Parameters
        ----------
        file_name : str
            name (and path) of the file to be loaded.

        Returns
        -------
        iq_data : np.ndarray
            array of samples loaded from the file

        """
        self.file_loaded = file_name
        
        iq_data = np.fromfile(file_name, dtype=np.complex64)
        return iq_data


    def shift_lo(self, iq_data: np.ndarray) -> np.ndarray:
        """
        shift_lo shifts the input signal by the LO offset used during capturing the data

        Parameters
        ----------
        iq_data : np.ndarray
            input signal to be shifted 1xN

        Returns
        -------
        shifted_signal : np.ndarray
            shifted output signal 1xN
        """
        omega = 1j*2*np.pi*(self.lo_offset/self.sample_rate)
        shifted_signal = iq_data*np.exp(omega*np.arange(0,len(iq_data)))
        return shifted_signal


    def get_fir_coeffs(self, numtaps: int, cutoff: float, width: float, fs: float=20e6) -> np.ndarray:
        """
        use scipy signal to design FIr filter coefficients

        Parameters
        ----------
        numtaps : int
            filter order - 1
        cutoff : float
            cutoff frequency in Hz
        width : float
            transition bandwidth in Hz
        fs : float, optional
            sampling rate of the filter design
            default is 20e6.

        Returns
        -------
        coeffs : np.ndarray
            array of filter coefficients

        """
        coeffs = signal.firwin(numtaps=numtaps, cutoff=cutoff, width=width, 
                               window='hamming', pass_zero='lowpass', fs=fs)
        return coeffs    


    def decimate(self, iq_data: np.ndarray, coeffs: np.ndarray, decimate: int=100) -> np.ndarray:
        """
        decimate downsamples and applies LP anti-aliasing filter

        Parameters
        ----------
        iq_data : np.ndarray
            input signal of IQ data
        coeffs: np.ndarray
            filter coefficients for LPF
        decimate : int, optional
            integer decimation factor 
            default is 100.
            
        Returns
        -------
        decimated_signal : np.ndarray
            decimated and filter signal downsampled by decimate value

        """
        decimated_signal = signal.upfirdn(coeffs, iq_data, up=1, down=decimate)
        return decimated_signal


    def rational_resample(self, iq_data: np.ndarray, up: int, down: int, window: np.ndarray) -> np.ndarray:
        """
        rational resample resamples the signal using integer up and down rates

        Parameters
        ----------
        iq_data : np.ndarray
            input iq data signal vector
        up : int
            integer upsample value
        down : int
            integer downsample value

        Returns
        -------
        resampled_data : np.ndarray
            resampled iq data

        """
        resampled_data = signal.resample_poly(iq_data, up=up, down=down, window=window)
        return resampled_data


    def quadrature_demod(self, iq_data: np.ndarray, gain: float=None) -> np.ndarray:
        """
        quadrature demod returns the demodulated FM data signal

        Parameters
        ----------
        iq_data : np.ndarray
            input iq signal
        gain : float
            scalar to apply 'gain' to the demodulated signal

        Returns
        -------
        demod_data : np.ndarray
            quadrature demodulated FM data stream

        """
        if gain == None:
            gain = self.audio_rate/(2*np.pi*self.frequency_max)
        np.append(iq_data,0)
        demod_data = np.angle(iq_data[:-1:]*np.conjugate(iq_data[1::]))
        return gain*demod_data
        
    
    def fm_deemphasis(self, iq_data: np.ndarray, sample_rate: float) -> np.ndarray:
        """
        FM deemphasis applies IIR filter to remove high frequency 
        content of demodulated signal

        Parameters
        ----------
        iq_data : np.ndarray
            input iq data signal
        sample_rate: float
            rate to process the deemphasis filter

        Returns
        -------
        deemphasized_data : TYPE
            output FM deemphasized iq data signal

        """
        # Digital corner frequency
        tau = 75e-6
        
        corner_freq = 1.0 / tau

        # Prewarped analog corner frequency
        analog_corner_freq = 2*self.audio_rate*math.tan(corner_freq/(2*self.audio_rate))

        # Resulting digital pole, zero, and gain term from the bilinear
        # transformation of H(s) = w_ca / (s + w_ca) to
        # H(z) = b0 (1 - z1 z^-1)/(1 - p1 z^-1)
        k = -analog_corner_freq/(2*self.audio_rate)
        b0 = -k/(1 - k)

        # output filter taps
        btaps = [b0*1, b0*1] # [b0*1, b0*z1], z1 = -1
        ataps = [1, -((1 + k)/(1 - k))] # p1 = (1 + k)/(1 - k)

        deemphasized_data = signal.lfilter(btaps, ataps, iq_data)
        return deemphasized_data


    def save_wav(self, audio_signal: np.ndarray) -> None:
        """
        converts IQ data stream to a playable wav file

        Parameters
        ----------
        audio_signal : np.ndarray
            demodulated FM signal

        Returns
        -------
        None
            saves the data to a file for playback

        """
        # Convert to int16 for audio playback
        wav_signal = np.float32(audio_signal/np.max(audio_signal))

        # Write the audio to a WAV file and play it
        wavfile.write("output.wav", int(self.audio_rate), wav_signal)
        print("Audio playback complete. Output saved as output.wav")


if __name__ == "__main__":
    # Example Usage
    
    # create RX instantiation
    sample_rate = 20e6
    receiver = FMRadioReceiver(sample_rate)
    
    # load saved FM signal
    fm_signal = receiver.load_iq_data('fm-radio-106p3MHz')
    
    # shift the LO
    shift_signal = receiver.shift_lo(fm_signal)
    
    # decimate by a factor of 100
    rx_coeffs  = receiver.get_fir_coeffs(numtaps=receiver.dec_numtaps, cutoff=receiver.dec_cutoff, width=receiver.dec_width)
    dec_signal = receiver.decimate(shift_signal,rx_coeffs,100)
    
    # rational resample the signal from 200KHz to 480KHz
    resamp_signal = receiver.rational_resample(dec_signal, 12, 5, window='blackmanharris')
    
    # perform quadrature demod on the audio signal
    demod_signal = receiver.quadrature_demod(resamp_signal)
    
    # filter and downsample the demodulated signal to an audio rate of 48KHz
    audio_coeffs = receiver.get_fir_coeffs(numtaps=receiver.audio_numtaps, cutoff=receiver.audio_cutoff, width=receiver.audio_width)
    audio_signal = receiver.decimate(demod_signal,audio_coeffs,10)
    
    # perform FM deemphasis of the output signal
    deemphasis_signal = receiver.fm_deemphasis(audio_signal, sample_rate=48e3)
    
    # write the output signal to a WAV file for playback
    receiver.save_wav(deemphasis_signal)