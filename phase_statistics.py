# %%
import cmath
from curses.ascii import RS
from math import *

import colorednoise as cn
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.signal as sig
from fooof import FOOOF
from scipy import stats
from scipy.fft import fft, fftshift
from scipy.signal import savgol_filter
import tqdm

plt.style.use('dark_background')

from plotting_tools import *
from sim_tools import *

# %% class definition
class PhaseSimulator:
    def __init__(self,oscillation_freq = 10, # frequency of oscillation
                 FS                    = 1000, # sampling frequency
                 WIN_SIZE              = 100,  # Hann window size
                 window_stop           = 500,  # sample at which to stop window, i.e. critical time or time of event onset
                 PROP_LENGTH           = 500,  # length of phase propagation
                 eps                   = 0.1,  # phase diffusion strength
                 niter                 = 1000, # number of iterations to simulate experiment
                 random_start          = True, # if True, start at random phase to each iteration
                 pink_noise_level      = 0.5, # pink noise level of lfp
                 k                     = 0):  # asymmetry of taper. posivie = right skewerd, negative = left skewed, 0 = no skew              

        self.oscillation_freq = oscillation_freq
        self.FS               = FS
        self.WIN_SIZE         = WIN_SIZE
        self.window_stop      = window_stop
        self.PROP_LENGTH      = PROP_LENGTH
        self.eps              = eps
        self.niter            = niter
        self.random_start     = random_start
        self.pink_noise_level = pink_noise_level
        self.k                = k


    def generate_lfp(self):
        """ generates lfp and ground truth phase data """        
        lfp, pdata = sim_lfp(self.oscillation_freq,nChannels=1,
                             eps=self.eps,FS=self.FS,nSamples=1000,
                             pink_noise_level=self.pink_noise_level,
                             random_start=self.random_start)
        self.lfp   = lfp
        self.pdata = pdata

    def generate_gtruth_and_pphase(self):
        """ generates ground truth and propagated phase data """
        _, taper = get_w_at(self.k,self.WIN_SIZE)

        self.taper = taper # this is symmetric if k = 0, so its a hann, asymmetric if k != 0

        prop_phase = propagate_phase(self.lfp,WIN_SIZE=self.WIN_SIZE,
                            TARGET_FREQ=self.oscillation_freq,
                            FS=self.FS,
                            NPAD=1000,
                            PROP_LENGTH=self.PROP_LENGTH,
                            window_stop=self.window_stop,
                            taper=self.taper)

        self.gtruth     = self.pdata[self.window_stop-self.WIN_SIZE: (self.window_stop-self.WIN_SIZE+self.PROP_LENGTH)]
        self.prop_phase = prop_phase

    def pairwise_circ_distance(self):
        """ calculates pairwise circular distance between ground truth and propagated phase """
        circ_distances = np.zeros(len(self.gtruth))
        for i in range(len(self.gtruth)):
            circ_distances[i] = np.angle(cmath.exp(1j*self.gtruth[i])/cmath.exp(1j*self.prop_phase[i])) * np.pi/180
        self.circ_distances = circ_distances

    def get_stats(self):
        """ returns the mean and std of the pairwise distance """
        self.mean_distance = stats.circmean(self.distances,axis=0)
        self.circ_stds     = stats.circstd(self.distances,axis=0)

    def get_R(self):
        R = np.empty(self.PROP_LENGTH,dtype=complex)
        for i in range(self.PROP_LENGTH):
            R[i] = cmath.exp(1j * (self.gtruth[i] - self.prop_phase[i]))
            self.R = R


    def experiment(self):
        """ runs the experiment """
        distances = np.empty((self.niter,self.PROP_LENGTH))
        Rs        = np.empty((self.niter,self.PROP_LENGTH),dtype=complex)

        for i in range(self.niter):
            self.generate_lfp() # make lfp simulation and ground truth phase data
            self.generate_gtruth_and_pphase() # cut the ground truth vector and propagate the phase
            self.pairwise_circ_distance() # calculate the pairwise distance between ground truth and propagated phase for this iteration
            self.get_R()

            distances[i,:] = self.circ_distances # attach the pairwise distance to the experiment
            Rs[i,:] = self.R # attach the R to the experiment

        self.distances = distances # attach the distances to the experiment
        self.Rs = Rs # attach the Rs to the experiment

        # get the stats on the experiment
        self.get_stats()


    def plot_phdiffusion(self):
        """ plots the phase diffusion """
        plot_phase_diffusion(self.lfp,
                             self.pdata,
                             DIFFSTART=self.window_stop-self.WIN_SIZE,
                             PROP_LENGTH=self.PROP_LENGTH,
                             prop_phase=self.prop_phase)

    def plot_stats(self):
        """ plots the mean and std of the pairwise distance """
        fig, ax = plt.subplots(1,1,figsize=(12,5))
        ax.plot(self.mean_distance,label='sin(mean_distance)',color="orange")
        ax.fill_between(range(len(self.mean_distance)),self.mean_distance-self.circ_stds,self.mean_distance+self.circ_stds,alpha=0.6,label='circ std')
        ax.set_title(f'Mean distance from propgated to true phase with phase diffusion strength of {str(self.eps)}', color="white")

        ax.legend()
        ax.set_xlabel('Sample')
        ax.set_ylabel('Sin(distance [rad])')
        plt.show()
        return fig, ax


# %% Analysis 1

eps   = 0.1
niter = 5000

def analysis_1(ks):
    """ analysis 1 """
    fig, ax = plt.subplots(len(ks),1,figsize=(12,18))


    for i,k in enumerate(ks):
        sim = PhaseSimulator(oscillation_freq=10,FS=1000,WIN_SIZE=100,window_stop=500,PROP_LENGTH=500,eps=eps,niter=niter,random_start=True,pink_noise_level=0.5,k=k)
        sim.experiment()
        ax[i].plot(abs(sum(sim.Rs)/niter),label='R')
        ax[i].plot(sim.taper/np.max(sim.taper),alpha=0.5,label='normalized taper')
        ax[i].set_title(f'k = {str(k)}', color="white")
        ax[i].set_ylabel('R')
        ax[i].legend()
    plt.show()
    return fig, ax

analysis_1([-5,-2,0,2,5])



 # %%
sim = PhaseSimulator(eps=eps,niter=niter,oscillation_freq=10,PROP_LENGTH=500,random_start=True,k=5)
sim.experiment()
fig, ax = plt.subplots(3,1,figsize=(8,8),sharex=True)
ax[0].plot(abs(sum(sim.Rs)/niter),label='R')
ax[0].set_title(f'Diffusion strength of {str(sim.eps)}', color="white")
ax[0].set_ylabel('R')
ax[0].plot(sim.taper/np.max(sim.taper),alpha=0.5,label='normalized taper')
ax[0].legend()

sim = PhaseSimulator(eps=eps,niter=niter,oscillation_freq=10,PROP_LENGTH=500,random_start=True,k=0)
sim.experiment()
ax[1].plot(abs(sum(sim.Rs)/niter),label='R')
ax[1].set_title(f'Diffusion strength of {str(sim.eps)}', color="white")
ax[1].set_ylabel('R')
ax[1].plot(sim.taper/np.max(sim.taper),alpha=0.5,label='normalized taper')
ax[1].legend()


sim = PhaseSimulator(eps=eps,niter=niter,oscillation_freq=10,PROP_LENGTH=500,random_start=True,k=-5)
sim.experiment()
ax[2].plot(abs(sum(sim.Rs)/niter),label='R')
ax[2].set_title(f'Diffusion strength of {str(sim.eps)}', color="white")
ax[2].set_xlabel('Samples')
ax[2].set_ylabel('R')
ax[2].plot(sim.taper/np.max(sim.taper),alpha=0.5,label='normalized taper')
ax[2].legend()


# %%

