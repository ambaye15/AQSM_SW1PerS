# Copyright 2025 Austin Amadou MBaye
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ripser import ripser
from persim import plot_diagrams
from scipy.interpolate import CubicSpline
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy import signal
from scipy import interpolate
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import numpy.matlib
import sympy
import multiprocessing as multi


# Section: Utility Functions for Algorithm
# -------------------------------------------------

def find_prominent_peaks(magnitude_spectrum, threshold, prominence):
    '''
    Tool to find the prominent peaks to be considered for period calculation 
    '''
    peaks, _ = find_peaks(magnitude_spectrum, height=threshold, prominence=prominence)
    sorted_peaks = sorted(peaks, key=lambda p: -magnitude_spectrum[p])
    cumulative_sum = np.cumsum(magnitude_spectrum[sorted_peaks])
    total_sum = np.sum(magnitude_spectrum[sorted_peaks])
    n_peaks = np.searchsorted(cumulative_sum, 0.7 * total_sum, side='right')
    return sorted_peaks[:n_peaks]



def knn_density(point_cloud, k = 20):
    '''
    Filters out sparse regions of the point cloud that could be artifacts of noise
    '''

    # Compute k-NN
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(point_cloud)
    distances, indices = nbrs.kneighbors(point_cloud)
    densities = 1 / np.nanmedian(distances, axis=1)
    
    retain_fraction = 0.8
    num_points_to_keep = int(retain_fraction * len(point_cloud))
    
    most_dense_indices = np.argsort(densities)[-num_points_to_keep:]
    
    subsampled_point_cloud = point_cloud[most_dense_indices]
    
    return subsampled_point_cloud


def SW_cloud_nD(cs, x_vals, tau, d, n_data, n_dims):
    '''
    Computes the sphere normalized sliding window embedding for an n-dimensional time series.
    '''
    t_vals = np.linspace(np.min(x_vals), np.max(x_vals) - (d * tau), n_data)
    SW = np.zeros((n_data, (d + 1) * n_dims))
    
    for i, t in enumerate(t_vals):
        for j in range(n_dims):
            SW_f_t = cs[j](t + np.arange(0, d + 1) * tau)
            SW[i, j * (d + 1):(j + 1) * (d + 1)] = SW_f_t
            
    SW = knn_density(SW, k = 20)
    # Subtract the mean from each row (mean centering)
    SW_mean = np.mean(SW, axis=1)
    SW_mean = numpy.matlib.repmat(SW_mean,  np.shape(SW)[1],1).T
    SW_centered = SW - SW_mean
    
    # Perform L2 normalization (row-wise)
    SW_norm = np.linalg.norm(SW_centered, axis=1, keepdims=True)
    SW_normalized = SW_centered / SW_norm
  
    return SW_normalized


def next_prime(d):
    '''
    Needed for ripser's coeff parameter
    '''
    return sympy.nextprime(d)


# Section: Period Estimation
# We have two ways to estimate the period:   
# - The primary method used with the pose estimation (x,y) coordinates is via the continuous discrete fourier transfrom presented in the paper. 
# - The alternative is presented in Supplamentrary Note. 4 for >= 3 component signals
# -------------------------------------------------

def estimate_period(keypoint_x, keypoint_y, sampling_rate):
    '''
    Estimate period using Complex Fast Fourier Transform
    '''
    F = keypoint_x + 1j * keypoint_y

    try:
        dft = np.fft.fft(F)
        magnitude_spectrum = np.abs(dft)
        frequencies = np.fft.fftfreq(len(F), 1/sampling_rate)
        
        pk = find_prominent_peaks(magnitude_spectrum, 0, 0) 
        peak_center = frequencies[pk][0]
        period = np.abs((1/peak_center))
        
        window_size = 4

        if period > window_size / 2: #Cutoff Period, we would need at least 2 oscillations to detect periodic motion
            epsilon = 1e-10  
            
            frequencies = np.where(frequencies == 0, epsilon, frequencies)
        
            cutoff_period = window_size / 2
        
            periods = np.abs(1/frequencies)
            
            # Apply cutoff to magnitude spectrum
            idx = np.where(periods < cutoff_period) 
            magnitude_spectrum = magnitude_spectrum[idx]
            frequencies = frequencies[idx]
            pk = find_prominent_peaks(magnitude_spectrum, 0, 0) 
        
            peak_center = frequencies[pk][0]
            period = np.abs((1/peak_center))
    except:
        period = np.nan
        
    return period


from sklearn.linear_model import Lasso

def generate_filtered_ramanujan_dictionary(T, fs, f_min=0.7, f_max=6.0):
    min_p = int(np.ceil(fs / f_max))
    max_p = int(np.floor(fs / f_min))
    atoms = []
    periods = []

    for g in range(min_p, max_p + 1):
        for d in range(1, g + 1):
            if g % d == 0:
                atom = np.cos(2 * np.pi * (np.arange(T) % d) / d)
                atom /= np.linalg.norm(atom)
                atoms.append(atom)
                periods.append(d)

    return np.column_stack(atoms), periods


def estimate_periods_lapis(Y, fs, f_min=0.7, f_max=6.0, alpha=0.001):
    T, N = Y.shape
    D, all_periods = generate_filtered_ramanujan_dictionary(T, fs, f_min, f_max)
    D = np.nan_to_num(D)
    y_flat = np.nanmean(Y, axis=1)
    model = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
    model.fit(D, y_flat)
    coefs = model.coef_
    selected_indices = np.where(np.abs(coefs) > 1e-4)[0]
    selected_periods = [all_periods[i] for i in selected_indices]
    selected_weights = [coefs[i] for i in selected_indices]
    return selected_periods, selected_weights, coefs, all_periods


def estimate_period_LAPIS(X_interp, sampling_rate):
    selected_periods, selected_weights, coefs, all_periods = estimate_periods_lapis(
        X_interp, sampling_rate
    )
    if selected_periods:
        top_idx = np.argmax(np.abs(selected_weights))
        top_period = selected_periods[top_idx]
        estimated_freq = sampling_rate / top_period
        return 1 / estimated_freq, top_period, estimated_freq
    else:
        return None, None, None

    
def compute_PS(dgm1, method = 'PS1'):
    try:
        #Sort the points to get the 10 most persistent points
        sorted_points = sorted(dgm1, key=lambda x: x[1] - x[0], reverse=True)
        n_most_persistent_points = np.array(sorted_points[:10])
        zeros_array = np.zeros((10 - len(n_most_persistent_points), 2))
        filled_points = np.vstack([n_most_persistent_points, zeros_array])
        persistences = np.abs(filled_points[:, 0] - filled_points[:, 1]) 
        weights =  np.exp(-0.5 * filled_points[:, 0])
    except:
        persistences = np.zeros(10)
        weights =  np.zeros(10)
        
    if method == 'PS10':
        return (persistences * weights) / np.sqrt(3)
    elif method == 'PS1':
        return (persistences[0] * weights[0]) / np.sqrt(3) - (persistences[1] * weights[1]) / np.sqrt(3)
    else:
        print('Not a valid mathod. Choose either 10MPS or 1PS')

        
# Section: Main Algorithm
# ---------------------------

def getFeatures(spline_funcs, segments, num_points = 1000, method = 'PS10'):
    '''
    Compute periodicity score of collection of time intervals
    :Param spline_funcs: Spline functions for given sensor. Supports (x,y), (x,y,z), and even multi(>= 4) dimensional signals
    :Param segments: n-second time intervals
    :Param num_points: controls the sampling rate of the signal 
    :Param method: Decision on how periodicity score is computed (Default = PS10)
    
    :Return periodicity_scores: Array of TDA scores
    '''

    periodicity_scores = []  

    d = 23

    prime_coeff = next_prime(2 * d)

    #Iterate through each time segment
    for i,segment in enumerate(segments):
        try:
            start=np.min(segment)
            end=np.max(segment)
            
            #For each time sgement, interolate values to increase resolution
            t_vals=np.linspace(start, end, num_points) 

            detrended_signals = [
                signal.detrend(f(t_vals)) for f in spline_funcs
            ]
            
            X_detrended = np.column_stack(detrended_signals)

            component_splines = [
                CubicSpline(t_vals, detrended_signals[i]) for i in range(len(detrended_signals))
            ]

            num_components = int(len(component_splines))
            
            sampling_rate = num_points / (end - start)

            if num_components == 2:
                period = estimate_period(X_detrended[:,0], X_detrended[:,1], sampling_rate)
            else:
                period, _ = estimate_period_LAPIS(X_detrended, sampling_rate)

            tau = period / (d + 1)    

            SW = SW_cloud_nD(component_splines, t_vals, tau, d, 300, num_components)
                
            result = ripser(SW, coeff = prime_coeff, maxdim = 1) 
            dgm1 = np.array(result['dgms'][1])

            score = compute_PS(dgm1, method = method)
            
            periodicity_scores.append(score)
            
        except Exception as e:
            if method == 'PS10':
                periodicity_scores.append(np.zeros(10))
            else:
                periodicity_scores.append(np.zeros(1))

    return np.vstack(periodicity_scores)


def processFeatures(args):
    '''
    Tool for multiprocessing getFeatures because computations across sensors are easily parallelizable 
    '''
    return getFeatures(*args)

#Example signal for demonstration
def run_periodicity_demo():
    import matplotlib.pyplot as plt
    
    t_vals = np.linspace(0, 4, 150)
    sampling_rate = 150 / 4
    x = np.cos(2 * np.pi * t_vals)
    y = np.sin(2 * np.pi *t_vals)
    X = np.column_stack((x, y))
    X += np.random.normal(scale=0.1, size=X.shape) #Add some noise
    period = estimate_period(X[:,0], X[:,1], sampling_rate)

    d = 23
    tau = period / (d + 1)
    spline_x = CubicSpline(t_vals, X[:,0])
    spline_y = CubicSpline(t_vals, X[:,1])
    spline = [spline_x, spline_y]

    SW = SW_cloud_nD(spline, t_vals, tau, d, 300, 2)

    prime_coeff = next_prime(2 * d)
    results = ripser(SW, coeff = prime_coeff, maxdim = 1) 
    diagrams = results['dgms']
    dgm1 = np.array(diagrams[1])
    score = compute_PS(dgm1, method = 'PS1')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    ax1.plot(t_vals,X[:, 0],color='r',label = 'X')
    ax1.plot(t_vals,X[:, 1],color='g',label = 'Y')
    ax1.set_title("Generated 2D Time Series")
    ax1.set_yticks([])
    ax1.axis("equal")

    plot_diagrams(diagrams, plot_only=[1], xy_range=[0, 2, 0, 2], ax = ax2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title(fr'Persistence Diagram')
    
    ax3.bar(range(1), score, alpha=0.5)
    ax3.set_title(fr'Periodicity Score')
    ax3.set_xlim(-0.5, 0.5)
    ax3.set_ylim(0, 1)
    ax3.set_xticks([])
    
    plt.tight_layout()
    plt.show()    

if __name__ == "__main__":
    run_periodicity_demo()
