import numpy as np
from quasar_class import Quasar
from quasar_stack_class import QuasarStack
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.signal
import os
import time
import pickle


mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

def ivarsmooth(flux, ivar, window):
    nflux = (flux.shape)[0]
    halfwindow = int(np.floor((np.round(window) - 1) / 2))
    shiftarr = np.zeros((nflux, 2 * halfwindow + 1))
    shiftivar = np.zeros((nflux, 2 * halfwindow + 1))
    shiftindex = np.zeros((nflux, 2 * halfwindow + 1))
    indexarr = np.arange(nflux)
    indnorm = np.outer(indexarr, (np.zeros(2 * halfwindow + 1) + 1))
    for i in np.arange(-halfwindow, halfwindow + 1, dtype=int):
        shiftarr[:, i + halfwindow] = np.roll(flux, i)
        shiftivar[:, i + halfwindow] = np.roll(ivar, i)
        shiftindex[:, i + halfwindow] = np.roll(indexarr, i)
    wh = (np.abs(shiftindex - indnorm) > (halfwindow + 1))
    shiftivar[wh] = 0.0
    outivar = np.sum(shiftivar, axis=1)
    nzero, = np.where(outivar > 0.0)
    zeroct = len(nzero)
    smoothflux = np.sum(shiftarr * shiftivar, axis=1)
    if (zeroct > 0):
        smoothflux[nzero] = smoothflux[nzero] / outivar[nzero]
    else:
        smoothflux = np.roll(flux, 2 * halfwindow + 1)  # kill off NAN's
    return smoothflux, outivar


plt.rc('font', family='serif')
plt.rc('text', usetex = True)

vals = np.genfromtxt("complete_sample/quasar_types.csv", skip_header = 1,dtype = [object, object, object, float, float, float, object], delimiter = ',', encoding = "latin1")

quas12 = []
quas15 = []
quas16 = []

smpixes = [501, 301, 501, 301, 301, 201, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301]
names = ["PSOJ323+12", "PSOJ231-20", "PSOJ065-26", "PSOJ011+09"]
names1 = ["PSOJ323+12", "PSOJ065-26", "PSOJ011+09"]

for k in range(vals.shape[0]):
    decoded = [el.decode() if type(el)==type(vals[k][0]) else el for el in vals[k]]
    name, file_nir, file_vis, redshift, dv, magnitude, inst = decoded
    quas16.append(Quasar(name, "complete_sample/spectra/"+file_nir, "complete_sample/spectra/"+file_vis, magnitude, redshift,draw = False, load = True, typeof = inst, sm_pix = smpixes[k])) 
    if name == names[1]:
        continue
    quas15.append(Quasar(name, "complete_sample/spectra/"+file_nir, "complete_sample/spectra/"+file_vis, magnitude, redshift,draw = False, load = True, typeof = inst, sm_pix = smpixes[k])) 
    if name in names:
        continue
    quas12.append(Quasar(name, "complete_sample/spectra/"+file_nir, "complete_sample/spectra/"+file_vis, magnitude, redshift,draw = False, load = True, typeof = inst, sm_pix = smpixes[k])) 


fig = plt.figure(figsize = (8,10))


vals = []
for q in (quas16):
    if q.name != "SDSSJ1148+5251":
        vals.append(np.amin(q.wav_obs))

minner = min(vals)
ax = [plt.subplot(811)]

for k, q in enumerate(quas16[:8]):
    smoothed_other, oivar = ivarsmooth(q.un_flux, q.un_ivar, 21)
    smoothed_other = np.ma.array(smoothed_other, mask = q.wav_obs.mask)
    ax[k].plot(q.wav_obs, smoothed_other, linewidth = 0.3, drawstyle = 'steps', color = 'black')
    ax[k].plot(q.wav_obs, oivar**(-0.5), linewidth = 0.3, drawstyle = 'steps', color = 'gray')
    ax[k].plot(q.obs_wav_model, q.pca_model*q.normalization, linewidth = 0.3, color = 'red')
    ax[k].text(0.01, 0.7, q.name+"\n"+"$z={}$".format(q.redshift), transform = ax[k].transAxes)
    ax[k].set_ylim(-0.5,top = np.amax(q.pca_model*q.normalization)*1.3)
    if k!=7:
        plt.setp(ax[k].get_xticklabels(), visible = False)
        ax.append(plt.subplot(812+k, sharex = ax[0]))
    ax[k].set_xlim(6000,22000)
plt.subplots_adjust(wspace=None, hspace=0)
plt.xlabel("Observed Wavelength [\\AA]")
fig.text(0.005, 0.5, "$f_{\\lambda} [10^{-16}\\mathrm{ergs}^{-1}\\mathrm{cm}^{-2}\mathrm{\\AA}^{-1}]$", va= 'center', rotation = 'vertical')
fig.tight_layout()
plt.savefig('spectra_part_1.pdf')



fig = plt.figure(figsize = (8,10))
ax = [plt.subplot(811)]

for k, q in enumerate(quas16[8:]):
    smoothed, oivar = ivarsmooth(q.un_flux, q.un_ivar, 21)
    smoothed_std = oivar**(-0.5)
    top = np.amax(q.pca_model*q.normalization)*1.3
    bot = -0.2
    if top<0.01:
        top*=1e16 
        smoothed*=1e16 
        smoothed_std *= 1e16
        q.pca_model*=1e16
        bot = -0.03
    ax[k].set_ylim(bot, top = top)
    ax[k].plot(q.wav_obs, smoothed, linewidth = 0.3, color = 'black', drawstyle = 
            'steps')
    ax[k].plot(q.wav_obs, smoothed_std, linewidth = 0.3, color= 'gray', drawstyle = 'steps')
    ax[k].plot(q.obs_wav_model, q.pca_model*q.normalization, linewidth = 0.3, color = 'red', drawstyle = 'steps')
    ax[k].text(0.01, 0.7, q.name+"\n"+"$z={}$".format(q.redshift), transform = ax[k].transAxes)

    if k!=7:
        plt.setp(ax[k].get_xticklabels(), visible = False)
        ax.append(plt.subplot(812+k, sharex = ax[0]))
    ax[k].set_xlim(6000, 22000)
plt.subplots_adjust(wspace=None, hspace=0)
plt.xlabel("Observed Wavelength [\\AA]")
fig.text(0.005, 0.5, "$f_{\\lambda} [10^{-16}\\mathrm{ergs}^{-1}\\mathrm{cm}^{-2}\mathrm{\\AA}^{-1}]$", va= 'center', rotation = 'vertical')
fig.tight_layout()
plt.savefig('spectra_part_2.pdf')
plt.show()

bin_size = 0.25

lower = 1188
upper = 1217

bin_list=  [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]

