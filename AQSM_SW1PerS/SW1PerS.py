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
from scipy import signal
from scipy import interpolate
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import numpy.matlib
import sympy
import multiprocessing as multi
from AQSM_SW1PerS.utils.period_estimation import *


# Section: Utility Functions for Algorithm
# -------------------------------------------------

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
        print('Not a valid mathod. Choose either PS10 or PS1')

# Section: SW1PerS Algorithm
# ---------------------------

class SW1PerS:

    def __init__(self, start_time = 0, end_time = 4, num_points = 1000, method = 'PS1', d = 23, prime_coeff = 47):
        self.start_time = start_time
        self.end_time = end_time
        self.num_points = num_points
        self.time_values = np.linspace(start_time, end_time, num_points) 
        self.method = method
        self.d = d
        self.prime_coeff = prime_coeff

        self.SW = None
        self.X_detrended = None
        self.num_components = None
        self.period = None
        self.periodicity_score = None

    def _detrend_and_convert(self, spline_funcs):
        
        detrended_signals = [
            signal.detrend(f(self.time_values)) for f in spline_funcs
        ]

        self.X_detrended = np.column_stack(detrended_signals)

        component_splines = [
            CubicSpline(self.time_values, detrended_signals[i]) for i in range(len(detrended_signals))
        ]

        self.num_components = int(len(component_splines))
        
        return component_splines
        
        
    def _estimate_period(self):

        sampling_rate = self.num_points / (self.end_time - self.start_time)

        period_estimator = PeriodEstimator(sampling_rate, self.num_components, f_min = 0.5, f_max = 2.0, window_size = (self.end_time - self.start_time))
        period = period_estimator.estimate_period(self.X_detrended)
        self.period = period
    
    def _sliding_windows(self, component_splines):
        
        tau = self.period / (self.d + 1)    

        self.SW = SW_cloud_nD(component_splines, self.time_values, tau, self.d, 300, self.num_components)

    def _1PerS(self):
        
        result = ripser(self.SW, coeff = self.prime_coeff, maxdim = 1) 
        dgm1 = np.array(result['dgms'][1])

        self.periodicity_score = compute_PS(dgm1, method = self.method)

    def compute_score(self, spline_funcs):

        component_splines = self._detrend_and_convert(spline_funcs)
        self._estimate_period()
        SW = self._sliding_windows(component_splines)
        self._1PerS()

        
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
            scoring_pipeline = SW1PerS(start_time = np.min(segment), end_time = np.max(segment), num_points = num_points, method = method, d = d, prime_coeff = prime_coeff)

            scoring_pipeline.compute_score(spline_funcs)
            
            periodicity_scores.append(scoring_pipeline.periodicity_score)

            #If higher temporal rate videos are available, you may also extract period as frequency-domain feature
            
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

    period_estimator = PeriodEstimator(sampling_rate, num_components = 2, f_min = 0.5, f_max = 2.0, window_size = 4.0)
    period = period_estimator.estimate_period(X)
    
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
