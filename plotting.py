import numpy as np
import matplotlib as plt
import matplotlib as mpl
from tqdm import tqdm, trange
import pickle
from quasar_class import Quasar
from quasar_stack_class import *
import lmfit
from helper_functions import *
import gc
import emcee

plt.rc('font', family='serif')
plt.rc('text', usetex = True)

def plot_boot(qstack):
    plt.figure()
    flux_covar = qstack.flux_covar

    std = np.sqrt(np.diagonal(flux_covar))
    ax = plt.axes(None, label = str(bin_size))
    
    num = 100
    ws_boot = qstack.ws_boot[:100]
    fs_boot = qstack.fs_boot[:100]
    
    plt.plot(ws_boot.T,fs_boot.T, alpha = 0.1, color = 'orange')
    plt.plot(ws_boot[0],fs_boot[0], alpha = 0.1, color = 'orange', label = 'Bootstrap Samples')
    plt.plot(qstack.wave_stack, qstack.flux_stack, label = 'Stacked Flux')
    #plt.title("Stacked Continuum Normalized Flux Near Ly-$\\alpha$ Transition")
    plt.xlabel("Wavelength (Angstroms)")
    plt.ylabel("Stacked Continuum Normalized Flux")
    plt.axvline(x=1215.67, color = 'red', linestyle = '--')
    plt.legend()

def plot_std(qstack):
    plt.figure()
    flux_covar = qstack.flux_covar

    std = np.sqrt(np.diagonal(flux_covar))
    ax = plt.axes(None, label = str(bin_size))
    plt.plot(qstack.wave_stack, qstack.flux_stack, label = 'Stacked Flux')
    ax.fill_between(qstack.wave_stack, qstack.flux_stack-std, qstack.flux_stack+std, alpha = 0.25, label = "1-$\\sigma$ Uncertainty Range")
    #plt.title("Stacked Continuum Normalized Flux Near Ly-$\\alpha$ Transition")
    plt.xlabel("Wavelength (Angstroms)")
    plt.ylabel("Stacked Continuum Normalized Flux")
    plt.axvline(x=1215.67, color = 'red', linestyle = '--')
    plt.legend()

def plot_cov(qstack):
    fig = plt.figure()
    ax = plt.axes(None, label = str(qstack.bin_size))
    flux_corr = np.corrcoef(qstack.fs_boot, rowvar = False)
    flux_covar = qstack.flux_covar
    im = ax.pcolormesh(qstack.wave_grid, qstack.wave_grid, flux_corr, vmin = -1,vmax = 1)
    plt.ylim(qstack.wave_grid[-1], qstack.wave_grid[0])
    cbar = fig.colorbar(im, ax = ax)
    cbar.set_label('Correlation')
    #plt.title("Continuum Normalized Flux Covariance Matrix Across Wavelength Bins")
    plt.xlabel("Wavelength Bin (Angstroms)")
    plt.ylabel("Wavelength Bin (Angstroms)")



bin_list = [0.25, 0.3, 0.35, 0.4]



for bin_size in bin_list:
    qstack11 = pickle.load(open('q_11_stack_{}.pickle'.format(bin_size), 'rb'))
    qstack14 = pickle.load(open('q_14_stack_{}.pickle'.format(bin_size), 'rb'))
    qstack15 = pickle.load(open('q_15_stack_{}.pickle'.format(bin_size), 'rb'))

    plot_std(qstack11)
    plt.savefig('stacking/bootstrap/11_stack_flux_var_'+str(bin_size)+'.pdf')
    plot_boot(qstack11)
    plt.savefig('stacking/bootstrap/11_stack_flux_samp_'+str(bin_size)+'.pdf')


    plot_std(qstack14)
    plt.savefig('stacking/bootstrap/14_stack_flux_var_'+str(bin_size)+'.pdf')
    plot_boot(qstack14)
    plt.savefig('stacking/bootstrap/14_stack_flux_samp_'+str(bin_size)+'.pdf')
    plot_std(qstack15)
    plt.savefig('stacking/bootstrap/15_stack_flux_var_'+str(bin_size)+'.pdf')
    plot_boot(qstack15)
    plt.savefig('stacking/bootstrap/15_stack_flux_samp_'+str(bin_size)+'.pdf')
    
    plot_cov(qstack11)
    plt.savefig('stacking/covar/11_flux_covar_bin_'+str(bin_size)+'.pdf')
    plot_cov(qstack14)
    plt.savefig('stacking/covar/14_flux_covar_bin_'+str(bin_size)+'.pdf')
    plot_cov(qstack15)
    plt.savefig('stacking/covar/15_flux_covar_bin_'+str(bin_size)+'.pdf')
    plt.close('all')
