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

quas12 = []
quas15 = []
quas16 = []

vals = np.genfromtxt("complete_sample/quasar_types.csv", skip_header = 1,dtype = [object, object, object, float, float, float, object], delimiter = ',', encoding = "latin1")

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


fig, axes = plt.subplots(8, 2, sharex = True, figsize = (7.2,9))


for k, q in enumerate(quas15):
    smoothed, oivar = ivarsmooth(q.cont_norm, q.cont_std**(-2.0), 3)
    smoothed_std = oivar**(-0.5)
    axes[(k+1)%8, k//8].plot(q.cont_wav, smoothed, linewidth = 0.3, drawstyle = 'steps', color = 'black')
    axes[(k+1)%8, k//8].text(0.01, 0.7, q.name+"\n"+"$z={}$".format(q.redshift), transform = axes[(k+1)%8, k//8].transAxes)
    top = 1.5
    bot = -0.2
    axes[(k+1)%8, k//8].set_ylim(bot, top = top)
    if k==7 or k==13:
        plt.setp(axes[(k+1)%8, k//8].get_xticklabels(), visible = True)
    axes[(k+1)%8, k//8].set_xlim(1177,1220)

axes[0][1].set_axis_off()
plt.subplots_adjust(wspace=None, hspace=0)
#plt.xlabel("Rest Wavelength [\\AA]")
fig.text(0, 0.5, "Continuum Normalized Flux", va= 'center', rotation = 'vertical')
fig.text(0.5, 0.0, "Rest Wavelength [\\AA]", ha= 'center')
fig.tight_layout()
plt.savefig('cont_all.pdf')
plt.show()

bin_size = 0.25

lower = 1188
upper = 1217

bin_list=  [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
