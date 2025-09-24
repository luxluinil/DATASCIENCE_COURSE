# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 03:13:06 2025

@author: Ben Carlson

Generates 3 sampled signals and the sum the of the 3 signals. Then creates
3 filters which are designed to pass 1 of the signals from the summed 
signal.
"""
import numpy as np
from math import sin, pi
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import firwin, fftconvolve, freqz, find_peaks

def estimate_peak_freqs(signal, sample_freq):
    ''' Estimates the periodogram peaks for a given time-domain signal.
    PARAMETERS
        signal -> array(float) The array of function values over time.
        sample_freq -> float The frequency used to generate the time samples.
    RETURNS
        peak_freqs -> array(float) The array of estimated peak values.
    '''
    N_samples = len(signal)
    # FFT of the signal
    fftSig = fft(signal)
    fftMag = np.abs(fftSig[:N_samples // 2])  # positive freqs only
    freqs = np.linspace(0, sample_freq / 2, N_samples // 2)
    peaks, _ = find_peaks(fftMag, height=np.max(fftMag)*0.1)  # threshold at 10% max height
    peak_freqs = freqs[peaks]
    return peak_freqs

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

def plot_sig(t_vec, f_t, title_str):
    ''' Plots f_t vs t_vec and displays the title string.
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
    
def plot_filter_response(b, sample_freq, title_str):
    ''' Plots the filter response vs freqency for a given filter (set of b params)
    PARAMETERS
        b -> array(float) The array of parameters which describe the filter.
        sample_freq -> float The frequency used to generate the time samples.
        title_str -> string To print on the graph.
    '''
    w, h = freqz(b, worN=8000)  # Frequency response
    freqs = ( w/(2.0*pi) ) * sample_freq  # Convert from rad/sample to Hz
    # Make the plot
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, 20.0 * np.log10(abs(h)), 'b')  # magnitude in dB
    plt.title(title_str)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.ylim(-100, 5)
    plt.show()
    return

def main():
    '''
    Generates 3 sampled signals and the sum the of the 3 signals. Then creates
    3 filters which are designed to pass 1 of the signals from the summed 
    signal.
    '''
    # Generate 3 signals and their sum
    N_samples = 2048
    sample_freq = 1024 # Hz
    # Time samples
    t_vec = np.arange(N_samples) / sample_freq
    # Signal parameters (different for each signal)
    A_1 = 10.0
    A_2 = 5.0
    A_3 = 2.5
    f0_1 = 100.0
    f0_2 = 200.0
    f0_3 = 300.0
    φ0_1 = 0.0
    φ0_2 = pi/6
    φ0_3 = pi/4
    # Sine Signals
    signal_1 = sine_signal(t_vec, A_1, f0_1, φ0_1)
    signal_2 = sine_signal(t_vec, A_2, f0_2, φ0_2)
    signal_3 = sine_signal(t_vec, A_3, f0_3, φ0_3)
    # Summed Signal
    sum_signal = signal_1 + signal_2 + signal_3
    
    # Plot the signals
    plot_sig(t_vec, signal_1, 'Signal 1')
    plot_sig(t_vec, signal_2, 'Signal 2')
    plot_sig(t_vec, signal_3, 'Signal 3')
    plot_sig(t_vec, sum_signal, 'Summed Signal')
    
    # Now create filters for the signals
    # I chose filter ranges manually, but this is an interesting alternative 
        # to automate the process: 
    peak_freqs = estimate_peak_freqs(sum_signal, sample_freq)
    print("Estimated peak frequencies:", peak_freqs)
    # Filter 1 (Lowpass)
    # Filter order
    fN_1 = 30
    # FIR filter design using firwin (like matlab's fir1)
    b_1 = firwin(fN_1+1, 150/(sample_freq/2))
    # Apply filter using fft-based filtering (like fftfilt)
    filt_sig_1 = fftconvolve(sum_signal, b_1, mode='same')
    
    # Filter 2 (Bandpass)
    fN_2 = 30
    b_2 = firwin(fN_2+1, [150/(sample_freq/2), 250/(sample_freq/2)], pass_zero='bandpass')
    filt_sig_2 = fftconvolve(sum_signal, b_2, mode='same')
    
    # Filter 3 (Highpass)
    fN_3 = 30
    b_3 = firwin(fN_3+1, 250/(sample_freq/2), pass_zero=False)
    filt_sig_3 = fftconvolve(sum_signal, b_3, mode='same')
    
    # Plot the Periodograms
    plot_periodo(sum_signal, 1/sample_freq, 'Periodogram - Summed Signal')
    plot_periodo(signal_1, 1/sample_freq, 'Periodogram - Signal 1')
    plot_periodo(filt_sig_1, 1/sample_freq, 'Periodogram - Filter 1')
    plot_periodo(signal_2, 1/sample_freq, 'Periodogram - Signal 2')
    plot_periodo(filt_sig_2, 1/sample_freq, 'Periodogram - Filter 2')
    plot_periodo(signal_3, 1/sample_freq, 'Periodogram - Signal 3')
    plot_periodo(filt_sig_3, 1/sample_freq, 'Periodogram - Filter 3')
    
    # Also plot the filter responses to visualize.
    plot_filter_response(b_1, sample_freq, 'Filter 1 Frequency Response (Lowpass)')
    plot_filter_response(b_2, sample_freq, 'Filter 2 Frequency Response (Bandpass)')
    plot_filter_response(b_3, sample_freq, 'Filter 3 Frequency Response (Highpass)')
    
if __name__ == "__main__":
    main()
