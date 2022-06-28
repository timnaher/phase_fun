# %%
from curses.ascii import RS
import numpy as np
import scipy
from scipy.fft import fft, fftshift
import scipy.signal as sig
from scipy import stats
from math import *
import cmath
from fooof import FOOOF
from scipy.signal import savgol_filter
import colorednoise as cn

import matplotlib.pyplot as plt

plt.style.use('dark_background')


from sim_tools import *
from plotting_tools import *


# %%
oscillation_freq = 10
FS               = 1000
WIN_SIZE         = 100
window_stop      = 500
PROP_LENGTH      = 500

# create lfp data and groundtruth phase
lfp, pdata = sim_lfp(oscillation_freq,nChannels=1,eps=0.1,FS=FS,nSamples=1000,pink_noise_level=0.5)


# this function propagates the phase based on a selected fourier coefficient of the lfp
prop_phase = propagate_phase(lfp,WIN_SIZE=WIN_SIZE,
                            TARGET_FREQ=oscillation_freq,
                            FS=FS,
                            NPAD=1000,
                            PROP_LENGTH=PROP_LENGTH,
                            window_stop=window_stop)
# plot both signal and phase
#plot_phase_diffusion(lfp,pdata,DIFFSTART=window_stop-WIN_SIZE,PROP_LENGTH=PROP_LENGTH,prop_phase=prop_phase)



# sample to sample distance between groundtruth and propagated phase
gtruth = pdata[window_stop-WIN_SIZE: (window_stop-WIN_SIZE+PROP_LENGTH)]

circ_distances = np.zeros(len(gtruth))
for i in range(len(gtruth)):
    circ_distances[i] = np.angle(cmath.exp(1j*gtruth[i])/cmath.exp(1j*prop_phase[i])) * np.pi/180

plot_phase_diffusion(lfp,pdata,DIFFSTART=window_stop-WIN_SIZE,PROP_LENGTH=PROP_LENGTH,prop_phase=prop_phase)


fig, ax = plt.subplots(2,1,figsize=(10,10))
ax[0].plot(circ_distances, label='circ_distances',color="orange")
ax[1].plot(gtruth, label='gtruth',color="orange")
ax[1].plot(prop_phase, label='prop_phase',color="red")
ax[0].set_title('circ_distances',color="white")
ax[1].set_title('gtruth',color="white")
ax[1].set_title('prop_phase',color="white")


def generate_gtruth_and_pphase(oscillation_freq = 10,
                               FS = 1000,
                               WIN_SIZE = 100,
                               window_stop = 500,
                               PROP_LENGTH = 500,
                               eps = 0.1):

    lfp, pdata = sim_lfp(oscillation_freq,nChannels=1,eps=eps,FS=FS,nSamples=1000,pink_noise_level=0.5)
    prop_phase = propagate_phase(lfp,WIN_SIZE=WIN_SIZE,
                            TARGET_FREQ=oscillation_freq,
                            FS=FS,
                            NPAD=1000,
                            PROP_LENGTH=PROP_LENGTH,
                            window_stop=window_stop)
    gtruth = pdata[window_stop-WIN_SIZE: (window_stop-WIN_SIZE+PROP_LENGTH)]
   

    return gtruth, prop_phase

def pairwise_circ_distance(gtruth,prop_phase):

    circ_distances = np.zeros(len(gtruth))
    for i in range(len(gtruth)):
        circ_distances[i] = np.angle(cmath.exp(1j*gtruth[i])/cmath.exp(1j*prop_phase[i])) * np.pi/180

    return circ_distances

# %%

niter = 1000
distances = np.empty((500,len(gtruth)))
for i in range(500):
    gtruth, prop_phase = generate_gtruth_and_pphase(eps =0.1)
    circ_distances     = pairwise_circ_distance(gtruth,prop_phase)
    distances[:,i] = circ_distances

from statistics import stdev
stds = stats.circstd(distances,axis=1)

mean_distance = np.sin(stats.circmean(distances,axis=0))

fig, ax = plt.subplots(2,1,figsize=(10,10))
ax[0].plot(mean_distance,label='sin(mean_distance)',color="orange")
ax[0].fill_between(range(len(mean_distance)),mean_distance-stds,mean_distance+stds,alpha=0.6,label='circ std')
ax[0].set_title('Mean distance from propgated to true phase with phase diffusion strength of 0.1',color="white")
ax[0].legend()
ax[0].set_xlabel('Sample')
ax[0].set_ylabel('Sin(distance [rad])')
distances = np.empty((500,len(gtruth)))
for i in range(500):
    gtruth, prop_phase = generate_gtruth_and_pphase(eps =0.9)
    circ_distances     = pairwise_circ_distance(gtruth,prop_phase)
    distances[:,i] = circ_distances

from statistics import stdev
stds = stats.circstd(distances,axis=1)

mean_distance = np.sin(stats.circmean(distances,axis=0))

ax[1].plot(mean_distance,label='sin(mean_distance)',color="orange")
ax[1].fill_between(range(len(mean_distance)),mean_distance-stds,mean_distance+stds,alpha=0.6,label='circ std')
ax[1].set_title('Mean distance from propgated to true phase with phase diffusion strength of 0.9',color="white")
ax[1].legend()
ax[1].set_xlabel('Sample')
ax[1].set_ylabel('Sin(distance [rad])')







# %%

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
# %%
sim_eps0_1 = PhaseSimulator(eps=0.1,niter=1000,oscillation_freq=10,PROP_LENGTH=500,random_start=True,use_asym_taper=True,k=2)
sim_eps0_1.experiment()

fig, ax = plt.subplots(3,1,figsize=(8,8))
ax[0].plot(abs(sum(sim_eps0_1.Rs)/1000),label='R')
ax[0].set_title(f'R with phase diffusion strength of {str(sim_eps0_1.eps)}', color="white")
ax[0].set_xlabel('Sample')
ax[0].set_ylabel('R')
ax[0].plot(sim_eps0_1.taper,alpha=0.5,label='taper')
ax[0].legend()

sim_eps0_1 = PhaseSimulator(eps=0.1,niter=1000,oscillation_freq=10,PROP_LENGTH=500,random_start=True,use_asym_taper=False,k=2)
sim_eps0_1.experiment()

ax[1].plot(abs(sum(sim_eps0_1.Rs)/1000),label='R')
ax[1].set_title(f'R with phase diffusion strength of {str(sim_eps0_1.eps)}', color="white")
ax[1].set_xlabel('Sample')
ax[1].set_ylabel('R')
ax[1].plot(sim_eps0_1.taper,alpha=0.5,label='taper')
ax[1].legend()


sim_eps0_1 = PhaseSimulator(eps=0.1,niter=1000,oscillation_freq=10,PROP_LENGTH=500,random_start=True,use_asym_taper=True,k=-2)
sim_eps0_1.experiment()

ax[2].plot(abs(sum(sim_eps0_1.Rs)/1000),label='R')
ax[2].set_title(f'R with phase diffusion strength of {str(sim_eps0_1.eps)}', color="white")
ax[2].set_xlabel('Sample')
ax[2].set_ylabel('R')
ax[2].plot(sim_eps0_1.taper,alpha=0.5,label='taper')
ax[2].legend()


# %%

