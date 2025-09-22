# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 21:29:22 2025

@author: Ben Carlson

Program to generate signals and plots:
    Part 1:  Generate sampled signal plots and periodograms for a pure sinusoid
        where the frequency is varied in small steps. For a frequency shift 
        less than 1/T, the peak location doesn't move, but the leakage
        increases. The periodogram becomes sinc-shaped.
    Part 2:  Generate and plot a quadratic chirp signal over sampling intervals
        of 0, 0.5, and 1 times the Nyquist frequency and the corresponding 
        periodograms.
"""
import numpy as np
from math import sin, pi
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.fft import fft

def pure_sine_signal(freq, samp_interv):
    ''' Generate a pure sine signal for given frequency and sampling interval.
    PARAMETERS
        freq -> float The frequency of the sine function.
        sample_interv -> float The step size used to choose the time values.
    RETURNS
        t_vec -> array(float) The array of time values.
        signal -> array(float) The array of sine signal values.
    '''
    t_vec = np.arange(0, 1.0, samp_interv)
    signal = np.sin(2.0 * pi * freq * t_vec)
    return t_vec, signal

def crcbgenqcsig(sample_interv, snr, coefs):
    ''' Generate a quadratic chirp signal for a given sampling interval.
    PARAMETERS
        sample_interv -> float The step size used to choose the time values.
        snr -> float Scaling factor for signal amplitude.
        coefs -> float[:] coefficients for phase polynomial.
    RETURNS
        t_vec -> array(float) The array of time values.
        f_t -> array(float) The array of chirp signal values.
    '''
    # Time samples from 0 to 1 second
    t_vec = np.arange(0, 1 + sample_interv, sample_interv)
    # Generate the signal and normalize
    φ_t = coefs[0]*t_vec + coefs[1]*(t_vec**2) + coefs[2]*(t_vec**3)
    f_t = np.array([ sin(2.0*pi*φ) for φ in φ_t ])
    f_t = snr * f_t / np.linalg.norm(f_t)
    return t_vec, f_t

def plot_sig(t_vec, f_t, title_str):
    ''' Plots f_t vs t_vec and displays the sampling interval.
    PARAMETERS
        t_vec -> float[:] The array of sampled times.
        f_t -> float[:] The array of function values at t_vec times.
        title_str -> string to print on the graph.
    '''
    plt.figure(figsize=(10, 6))
    plt.plot(t_vec, f_t, marker='.', markersize=12, linestyle='-')
    plt.xlabel('Time (sec)')
    plt.title(title_str)
    plt.grid(True)
    plt.show()
    return

def gen_periodo(signal, samp_interv, N_samples, title_str):
    ''' Generates the fast fourier transform (fft) of a given signal and plots
    the fft vs frequency.
    PARAMETERS
        signal -> array(float) The array of signal values to transform.
        samp_interv -> float The step size used to choose the frequency values.
        N_samples -> int The number of samples.        
        title_str -> string To print on the graph.
    '''
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

def gen_spectro(signal, samp_interv):
    ''' Generates the spectrogram of a given signal and plots it.
    PARAMETERS
        signal -> array(float) The array of signal values to use.
        samp_interv -> float The step size.
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
    plt.title(f'Spectrogram {samp_interv}')
    plt.colorbar(label='Magnitude')
    plt.show()
    return

def main():
    '''
    Part 1:  Generate sampled signal plots and periodograms for a pure sinusoid
        where the frequency is varied in small steps. For a frequency shift 
        less than 1/T, the peak location doesn't move, but the leakage
        increases. The periodogram becomes sinc-shaped.
    Part 2:  Generate and plot a quadratic chirp signal over sampling intervals
        of 0, 0.5, and 1 * the Nyquist frequency and the corresponding 
        periodograms.
    '''
    # Signal parameters
    a1 = 10.0
    a2 = 3.0
    a3 = 3.0
    A = 10.0

    # PART 1
    # Parameters for the pure sinusoid
    base_freq = 25              # in Hz
    intv = 1 / (10 * base_freq)
    delta_f = 0.5               # less than 1/T, (T=1.0)
    # Generate the plots for each frequency shift.
    for df in [0.0, delta_f/2.0, delta_f]:
        freq = base_freq + df
        # generate the signal
        ts, s = pure_sine_signal(freq, intv)
        # get max frequency
        max_f = max([(a1 + 2*a2*t + 3*a3*t*t) for t in ts])
        # plot the sampled signal
        plot_sig(ts, s, f'Frequency = {freq:.2f}, Δf = {df:.2f}, max f = {max_f:.2f}')
        # plot the periodogram
        gen_periodo(s, intv, len(ts), f'Frequency = {freq:.2f}, Δf = {df:.2f} (Hz)')

    # PART 2
    # Instantaneous frequency after 1 second
    maxFreq = a1 + 2*a2 + 3*a3
    # Nyquist frequency guess: 2 * max instantaneous frequency
    nyqFreq = 2 * maxFreq
    # Choose sampling intervals to use
    sample_intervals = [1/nyqFreq,
                        1/(2*nyqFreq), 
                        1/(10*nyqFreq)
                        ]
    for samp in sample_intervals:
        # Generate the signal
        ts, fs = crcbgenqcsig(samp, A, [a1, a2, a3])
        # get max frequency
        max_f = max([(a1 + 2*a2*t + 3*a3*t*t) for t in ts])
        # plot the sample
        plot_sig(ts, fs, f'Signal, Interval = {samp}, max f = {max_f:.2f}')
        # Generate the periodograms
        gen_periodo(fs, samp, len(ts), f'Periodogram, Interval = {samp}')
        # Generate the spectrograms
        #gen_spectro(fs, samp)
    
if __name__ == "__main__":
    main()
