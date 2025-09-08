# -*- coding: utf-8 -*-
"""
Created on Mon Sept 8 01:57:22 2025

@author: Ben Carlson

Program to generate and plot a quadratic chirp signal over 
chosen sampling intervals.

"""

import numpy as np
from math import sin, pi
import matplotlib.pyplot as plt

def plot_func(t_vec, f_t, figNum):
    ''' Plots t_vec vs f_t and displays the sampling interval.
    parameters
        t_vec -> float[:] The array of sampled times.
        f_t -> float[:] The array of function values at t_vec times.
        figNum -> float The sampling interval used to make t_vec.
    '''
    plt.figure(figsize=(10, 6))
    plt.plot(t_vec, f_t, marker='.', markersize=12, linestyle='-')
    plt.xlabel('Time (sec)')
    plt.title(f'Interval = {figNum}')
    plt.grid(True)
    plt.show()
    return

def crcbgenqcsig(sample_interv, snr, coefs):
    ''' Generate a quadratic chirp signal for a given sampling interval.
    parameters
        sample_interv -> float The step size used to choose the time values.
        snr -> float Scaling factor for signal amplitude.
        coefs -> float[:] coefficients for phase polynomial.
    '''
    # Time samples from 0 to 1 second
    t_vec = np.arange(0, 1 + sample_interv, sample_interv)
    # Generate the signal and normalize
    φ_t = coefs[0]*t_vec + coefs[1]*(t_vec**2) + coefs[2]*(t_vec**3)
    f_t = np.array([ sin(2.0*pi*φ) for φ in φ_t ])
    f_t = snr * f_t / np.linalg.norm(f_t)
    return t_vec, f_t

def main():
    '''
    Program to generate and plot a quadratic chirp signal over 
    chosen sampling intervals.
    '''
    # Signal parameters
    a1 = 10.0
    a2 = 3.0
    a3 = 3.0
    A = 10.0
    # Instantaneous frequency after 1 second
    maxFreq = a1 + 2*a2 + 3*a3
    # Nyquist frequency guess: 2 * max instantaneous frequency
    nyqFreq = 2 * maxFreq

    # Choose sampling intervals to try
    sample_intervals = [1/nyqFreq,
                        1/(2*nyqFreq), 
                        1/(5*nyqFreq),
                        1/(10*nyqFreq)
                        ]
    
    # Generate the signal plots
    for samp in sample_intervals:
        ts, fs = crcbgenqcsig(samp, A, [a1, a2, a3])
        plot_func(ts, fs, samp)
        

if __name__ == "__main__":
    main()
