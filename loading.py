import numpy as np
from quasar_class import Quasar
from quasar_stack_class import QuasarStack
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import time
import pickle

mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


plt.rc('font', family='serif')
plt.rc('text', usetex = True)


vals = np.genfromtxt("complete_sample/quasar_types.csv", skip_header = 1,dtype = [object, object, object, float, float, float, object], delimiter = ',', encoding = "latin1")

quas = []
quas1 = []
quas2 = []

smpixes = [501, 301, 501, 301, 301, 201, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301]
names = ["PSOJ323+12", "PSOJ231-20", "PSOJ065-26", "PSOJ011+09"]
names1 = ["PSOJ323+12", "PSOJ065-26", "PSOJ011+09"]

for k in range(vals.shape[0]):
    decoded = [el.decode() if type(el)==type(vals[k][0]) else el for el in vals[k]]
    name, file_nir, file_vis, redshift, dv, magnitude, inst = decoded
    #quas2.append(Quasar(name, "complete_sample/spectra/"+file_nir, "complete_sample/spectra/"+file_vis, magnitude, redshift,draw = False, load = True, typeof = inst, sm_pix = smpixes[k])) 
    if name == names[1]:
        continue
    quas1.append(Quasar(name, "complete_sample/spectra/"+file_nir, "complete_sample/spectra/"+file_vis, magnitude, redshift,draw = False, load = True, typeof = inst, sm_pix = smpixes[k])) 
    if name in names:
        continue
    #quas.append(Quasar(name, "complete_sample/spectra/"+file_nir, "complete_sample/spectra/"+file_vis, magnitude, redshift,draw = False, load = True, typeof = inst, sm_pix = smpixes[k])) 


bin_size = 0.25

lower = 1177
upper = 1218

bin_list=  [0.1, 0.25, 0.4, 0.5, 0.6]

print(len(quas), len(quas1), len(quas2))

fig = plt.figure()


ax = [plt.subplot(511)]


for k, bin_size in enumerate(bin_list):
    qstack1 = QuasarStack(quas1, lower, upper, bin_size)
    
    with open('q_15_stack_{}.pickle'.format(qstack1.bin_size), 'wb') as filename:
        pickle.dump(qstack1, filename)
    
    ax[k].plot(qstack1.wave_stack, qstack1.flux_stack, linewidth = 0.3, drawstyle = 'steps', color = 'black', label = f"{bin_size} {{\\AA}} Bins")
    top = 1.2
    bot = -0.1
    ax[k].set_ylim(bot, top = top)
    
    plt.legend()
    if k!=4:
        plt.setp(ax[k].get_xticklabels(), visible = False)
        ax.append(plt.subplot(512+k, sharex = ax[0]))
    ax[k].set_xlim(1177,1221)


plt.subplots_adjust(wspace=None, hspace=0)
plt.xlabel("Rest Wavelength [\\AA]")
fig.text(0.00, 0.5, "Continuum Normalized Flux", va= 'center', rotation = 'vertical')
fig.tight_layout()
plt.savefig('stack_plot.pdf')


