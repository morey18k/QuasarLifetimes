import numpy as np
from quasar_class import Quasar
from quasar_stack_class import QuasarStack
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.signal
import os
import time
import pickle

plt.rc('font', family='serif')
plt.rc('text', usetex = True)


mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

bin_size  = 0.25
tqs = np.linspace(1, 8.9, 80)
wave_grid = np.linspace(1177, 1217.75, 164)
covars = np.array([np.load('covar_model/covar_matrix_tq{1:.2}_{0}.npy'.format(bin_size, t))[:-1, :-1] for t in tqs])

invcovs = [cov/np.sqrt(np.outer(np.diag(cov), np.diag(cov))) for cov in covars]

element_diag = np.array([cov[30,30] for cov in covars])
element_off = np.array([cov[30,34] for cov in covars])
tracer = np.array([np.trace(cov) for cov in covars])
invtracer = np.array([np.trace(np.linalg.inv(cov)) for cov in covars])
fig = plt.figure()
plt.xlabel("$\\log_{10}(t_Q)$ [years]")
plt.ylabel("Trace of Inverse Covariance")
plt.plot(tqs, invtracer-np.amin(invtracer)+10000, '.-')
plt.yscale('log')
plt.savefig('model/inv_covar_trace.pdf')


fig = plt.figure()
plt.xlabel("$\\log_{10}(t_Q)$ [years]")
plt.ylabel("Covariance")
plt.plot(tqs, element_diag, '.-')
plt.savefig('model/covar_variation_diag.pdf')

fig = plt.figure()
plt.xlabel("$\\log_{10}(t_Q)$ [years]")
plt.ylabel("Covariance")
plt.plot(tqs, element_off, '.-')
plt.savefig('model/covar_variation_off.pdf')

fig = plt.figure()
plt.xlabel("$\\log_{10}(t_Q)$ [years]")
plt.ylabel("Trace of Covariance")
plt.plot(tqs, tracer, '.-')
plt.savefig('model/covar_trace.pdf')


k = 50

fig = plt.figure()
ax = plt.axes(None, label = str(k))
inv = invcovs[k]
im = ax.pcolormesh(wave_grid, wave_grid, inv)
plt.ylim(wave_grid[-1], wave_grid[0])
cbar = fig.colorbar(im, ax = ax)
cbar.set_label('Correlation')
#plt.title("Continuum Normalized Flux Covariance Matrix Across Wavelength Bins")
plt.xlabel("Wavelength Bin (Angstroms)")
plt.ylabel("Wavelength Bin (Angstroms)")
plt.savefig('noise_modeling/all_tq/corr_{1}_tq{0}.pdf'.format(tqs[k], bin_size))
plt.close('all')


