# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 23:09:58 2025

@author: Ben Carlson

Generates 3 sampled sine signals, the sum the of the 3 signals, and 1 
quadratic chirp signal. Then plots each of the 5 signals vs time, their 
periodograms, and their spectrograms.
"""
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import spectrogram

def sine_signal(t_vec, A, f0, φ0):
    ''' Generate a pure sine signal for given frequency and sampling interval.
    PARAMETERS
        t_vec -> array(float) The array of time values.
        A -> float Scaling factor for signal amplitude.
        f0 -> float The frequency of the sine function.
        φ0 -> float The phase of the signal.
    RETURNS
        signal -> array(float) The array of sine signal values.
    '''
    signal = A * np.sin(2.0*pi*f0*t_vec + φ0)
    return signal

def crcbgenqcsig(t_vec, snr, coefs):
    ''' Generate a quadratic chirp signal for a given sampling interval.
    PARAMETERS
        t_vec -> array(float) The array of time values.
        snr -> float Scaling factor for signal amplitude.
        coefs -> float[:] coefficients for phase polynomial.
    RETURNS
        f_t -> array(float) The array of chirp signal values.
    '''
    # Generate the signal and normalize
    φ_t = coefs[0]*t_vec + coefs[1]*(t_vec**2) + coefs[2]*(t_vec**3)
    f_t = np.sin(2.0*pi*φ_t)
    f_t = snr * f_t / np.linalg.norm(f_t)
    return f_t

def plot_sig(t_vec, f_t, title_str):
    ''' Plots f_t vs t_vec and displays the title string.
    PARAMETERS
        t_vec -> array(float) The array of sampled times.
        f_t -> array(float) The array of function values at t_vec times.
        title_str -> string to print on the graph.
    '''
    plt.figure(figsize=(10, 6))
    plt.plot(t_vec, f_t, marker='.', markersize=12, linestyle='-')
    plt.xlabel('Time (sec)')
    plt.title(title_str)
    plt.grid(True)
    plt.show()
    return

def plot_periodo(signal, samp_interv, title_str):
    ''' Generates the fast fourier transform (fft) of a given signal and plots
    the fft vs frequency.
    PARAMETERS
        signal -> array(float) The array of signal values to transform.
        samp_interv -> float The step size used to choose the frequency values.
        title_str -> string To print on the graph.
    '''
    N_samples = len(signal)
    # DFT sample corresponding to Nyquist frequency
    kNyq = (N_samples // 2) + 1
    # Positive Fourier frequencies
    posFreq = np.arange(0, kNyq) * (1/(N_samples*samp_interv))
    # FFT of the signal
    fftSig = fft(signal)
    # Discard negative frequencies
    fftSig = fftSig[:kNyq]
    # Plot the Periodogram
    plt.figure(figsize=(10, 6))
    plt.plot(posFreq, np.abs(fftSig), marker='.', markersize=12, linestyle='-')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('|FFT|')
    plt.title(title_str)
    plt.grid(True)
    plt.show()
    return

def plot_spectro(signal, samp_interv, title_str):
    ''' Generates the spectrogram of a given signal and plots it.
    PARAMETERS
        signal -> array(float) The array of signal values to use.
        samp_interv -> float The step size.
        title_str -> string To print on the graph.
    '''
    # Spectrogram parameters
    winLen = 0.2  # sec
    ovrlp = 0.1  # sec
    # Convert to integer number of samples
    winLenSmpls = int(winLen / samp_interv)
    ovrlpSmpls = int(ovrlp / samp_interv)
    # Compute spectrogram
    f, t, Sxx = spectrogram(signal, fs=1/samp_interv, nperseg=winLenSmpls, 
                            noverlap=ovrlpSmpls)
    # Plot the Spectrogram
    plt.figure(figsize=(10, 6))
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.pcolormesh(t, f, np.abs(Sxx), shading='auto')
    plt.title(title_str)
    plt.colorbar(label='Magnitude')
    plt.show()
    return

def main():
    '''
    Generates 3 sampled sine signals, the sum the of the 3 signals, and 1 
    quadratic chirp signal. Then plots each of the 5 signals vs time, their 
    periodogram, and their spectrogram.
    '''
    # Generate 3 sine signals and their sum
    N_samples = 2048
    sample_freq = 1024 # Hz
    # Time samples
    t_vec = np.arange(N_samples) / sample_freq
    # Signal parameters (different for each signal)
    A_1, f0_1, φ0_1 = 10.0, 100.0, 0.0  # Signal 1 parameters
    A_2, f0_2, φ0_2 = 5.0,  200.0, pi/6 # Signal 2 parameters
    A_3, f0_3, φ0_3 = 2.5,  300.0, pi/4 # Signal 3 parameters
    # Sine Signals
    signal_1 = sine_signal(t_vec, A_1, f0_1, φ0_1)
    signal_2 = sine_signal(t_vec, A_2, f0_2, φ0_2)
    signal_3 = sine_signal(t_vec, A_3, f0_3, φ0_3)
    # Summed Signal
    sum_signal = signal_1 + signal_2 + signal_3
    
    # Generate 1 Quadratic Chirp Signal
    # Chirp signal parameters
    a1 = 10.0
    a2 = 3.0
    a3 = 3.0
    A = 10.0
    # Instantaneous frequency after 1 second
    maxFreq = a1 + 2*a2 + 3*a3
    # Nyquist frequency guess: 2 * max instantaneous frequency
    nyqFreq = 2 * maxFreq
    # Choose sampling interval to use
    sample_interv = 1/(5*nyqFreq)
    # Time samples from 0 to 1 second
    t_vec_chirp = np.arange(0, 1 + sample_interv, sample_interv)
    # Get the chirp signal
    chirp_sig = crcbgenqcsig(t_vec_chirp, A, [a1, a2, a3])
    
    # Plot the signals
    plot_sig(t_vec, signal_1, 'Signal 1')
    plot_sig(t_vec, signal_2, 'Signal 2')
    plot_sig(t_vec, signal_3, 'Signal 3')
    plot_sig(t_vec, sum_signal, 'Summed Signal')
    plot_sig(t_vec_chirp, chirp_sig, 'Chirp Signal')
    
    # Plot the Periodograms
    plot_periodo(signal_1, 1/sample_freq, 'Periodogram - Signal 1')
    plot_periodo(signal_2, 1/sample_freq, 'Periodogram - Signal 2')
    plot_periodo(signal_3, 1/sample_freq, 'Periodogram - Signal 3')
    plot_periodo(sum_signal, 1/sample_freq, 'Periodogram - Summed Signal')
    plot_periodo(chirp_sig, sample_interv, 'Periodogram - Chirp Signal')

    # Spectrogram Plots
    plot_spectro(signal_1, 1/sample_freq, 'Spectrogram - Signal 1')
    plot_spectro(signal_2, 1/sample_freq, 'Spectrogram - Signal 2')
    plot_spectro(signal_3, 1/sample_freq, 'Spectrogram - Signal 3')
    plot_spectro(sum_signal, 1/sample_freq, 'Spectrogram - Summed Signal')
    plot_spectro(chirp_sig, sample_interv, 'Spectrogram - Chirp Signal')
    
if __name__ == "__main__":
    main()
