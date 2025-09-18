#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 11:44:12 2025

@author: jlab
"""

import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
import os

class FMRadioTransmitter:
    """
    A class for FM radio signal reception and demodulation.
    """

    def __init__(self, sample_rate: int, lo_offset: float=1.4e6) -> None:
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
        
        self.int_factor  = 10
        self.int_numtaps = 289
        self.int_width   = 2e3
        self.int_cutoff  = 16e3
        self.int_rate    = self.audio_rate*self.int_factor

        self.audio_numtaps = 2049
        self.audio_width   = self.int_rate/32
        self.audio_cutoff  = self.int_rate/2 - self.audio_width
        
        self.lpf_factor = int(self.sample_rate/200e3)
        
        
    def save_iq_data(self, iq_data: np.ndarray, file_name: str) -> np.ndarray:
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
        self.file_saved = file_name
        iq_data = iq_data.astype(np.complex64)
        iq_data.tofile(file_name)


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


    def interpolate(self, iq_data: np.ndarray, coeffs: np.ndarray, interpolate: int=10) -> np.ndarray:
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
        interpolated_signal = signal.upfirdn(coeffs, iq_data, up=interpolate, down=1)
        return interpolated_signal


    def quadrature_mod(self, audio_data: np.ndarray) -> np.ndarray:
        """
        quadrature demod returns the demodulated FM data signal

        Parameters
        ----------
        iq_data : np.ndarray
            input iq signal

        Returns
        -------
        moduldated_data : np.ndarray
            quadrature modulated FM data stream

        """
        signal_phase = 2*np.pi*self.frequency_max*np.cumsum(audio_data)/self.sample_rate
        modulated_data = np.exp(1j*signal_phase)
        return modulated_data
        
    
    def fm_preemphasis(self, iq_data: np.ndarray, sample_rate: float) -> np.ndarray:
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
        preemphasized_data : TYPE
            output FM preemphasized iq data signal

        """
        # Digital corner frequency
        tau = 75e-6
        frequency_high_corner = .925*sample_rate/2
        
        w_cl = 1/tau
        w_ch = 2*np.pi*frequency_high_corner

        # Prewarped analog corner frequencies
        w_cla = 2*sample_rate*np.tan(w_cl/(2*sample_rate))
        w_cha = 2*sample_rate*np.tan(w_ch/(2*sample_rate))

        # Resulting digital pole, zero, and gain term from the bilinear
        # transformation of H(s) = (s + w_cla) / (s + w_cha) to
        # H(z) = b0 (1 - z1 z^-1)/(1 - p1 z^-1)
        kl = -w_cla/(2*sample_rate)
        kh = -w_cha/(2*sample_rate)
        z1 = (1+kl)/(1-kl)
        p1 = (1+kh)/(1-kh)
        b0 = (1-kl)/(1-kh)

        # normalized gain factor
        g = np.abs(1-p1)/(b0*np.abs(1-z1))

        btaps = [g*b0, g*b0*-z1]
        ataps = [1, -p1]
        
        preemphasized_data = signal.lfilter(btaps, ataps, iq_data)
        return preemphasized_data


    def load_wav(self, wav_file: str) -> np.ndarray:
        """
        converts IQ data stream to a playable wav file

        Parameters
        ----------
        wav_file : str
            load a wav file

        Returns
        -------
        wav_data : np.ndarray
            saves the data to a file for playback

        """
        # read in wav data from file
        self.audio_rate, wav_data = wavfile.read(wav_file)
        return wav_data
        


if __name__ == "__main__":
    # Example Usage
    
    # create RX instantiation
    sample_rate = 480e3
    transmitter = FMRadioTransmitter(sample_rate)
    
    # empty list of modulated data to be appended to
    modulated_data = np.array([])
    
    # Specify the directory path
    directory_path = './wavs/'

    # List all files in the directory
    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    
    for file in files:
        # load the wav file data
        wav_data = transmitter.load_wav(directory_path + file)
        if wav_data.dtype==np.dtype('uint8'):
            wav_data = (wav_data.astype(np.float32)/2**7 - 1)
        
        if transmitter.audio_rate < 48e3:
            uprate = 48e3/transmitter.audio_rate
            wav_data = signal.resample_poly(wav_data, int(uprate),1)
            transmitter.audio_rate = 48e3;
        
        # interpolate by upsampling and filtering
        interp_coeffs = transmitter.get_fir_coeffs(numtaps=transmitter.int_numtaps, cutoff=transmitter.int_cutoff, width=transmitter.int_width, fs=transmitter.int_rate)
        interp_data = transmitter.interpolate(wav_data, interp_coeffs, 10)
        
        # apply FM preeemphasis filter
        preemphasis_data = transmitter.fm_preemphasis(interp_data, transmitter.int_rate)
        
        # FM modulated the message signal
        modulated_data = np.append(modulated_data, transmitter.quadrature_mod(preemphasis_data))
        
    # save signal for transmission
    transmitter.save_iq_data(modulated_data, 'input.bin')