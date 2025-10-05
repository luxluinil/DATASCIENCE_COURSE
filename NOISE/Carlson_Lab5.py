# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 02:14:11 2025

@author: Ben Carlson

Generates random values using both uniform and normal distributions and
both PCG64 and MT19937 generators. Then plots histograms of these generated
values. Finally, generate 2 sets of random values with the same seed and 
compare.
"""
import numpy as np
import matplotlib.pyplot as plt
from math import isclose

def plot_histogram(vals, title_str):
    ''' Plots the histogram of a given set of values.
    PARAMETERS
        vals -> array(float) The array of values to plot.
        title_str -> string To print on the graph.
    '''
    plt.figure(figsize=(10, 6))
    plt.hist(vals, bins=50, color='skyblue', edgecolor='black')
    plt.title(title_str)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    return

def main():
    ''' Generates random values using both uniform and normal distributions and
    both PCG64 and MT19937 generators. Then plots histograms of these generated
    values. Finally, generate 2 sets of random values with the same seed and 
    compare.'''
    # Get 10,000 values from uniform and normal distributions and 2 generators
    # Generate values using PCG64
    rand_gen = np.random.default_rng(0)
    vals_u_1 = rand_gen.uniform(0, 1, 10000) # low boundary, high boundary, num values
    vals_n_1 = rand_gen.normal(0, 1, 10000) # mean, st deviation, num values
    # Generate values using MT19937 (Mersenne Twister) same as Matlab
    np.random.seed(0)
    vals_u_2 = np.random.rand(10000) # from uniform distribution
    vals_n_2 = np.random.randn(10000) # from normal distribution
    
    # Create Histograms of the random values
    plot_histogram(vals_u_1, 'random values from a uniform distribution using PCG64')
    plot_histogram(vals_n_1, 'random values from a normal distribution using PCG64')
    plot_histogram(vals_u_2, 'random values from a uniform distribution using MT19937 (Mersenne Twister)')
    plot_histogram(vals_n_2, 'random values from a normal distribution using MT19937 (Mersenne Twister)')
    
    # Generate 2 sets of 5,000 values from randn with the same seed and compare
    np.random.seed(0)
    vals_comp_1 = np.random.randn(5000)
    np.random.seed(0)
    vals_comp_2 = np.random.randn(5000)

    # Print each pair of values and compare to see if they are the same.
    for v1,v2 in zip(vals_comp_1, vals_comp_2):
        print(v1,'\t',v2)
        if not isclose(v1, v2):
            print(v1, 'and', v2, 'are different')
    else:
        print('\nThe two sets are the same.')

if __name__ == "__main__":
    main()
