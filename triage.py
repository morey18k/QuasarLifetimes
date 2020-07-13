import numpy as np
from quasar_class import Quasar
import matplotlib.pyplot as plt

vals = np.genfromtxt("complete_sample/quasar_types.csv", skip_header = 1,dtype = [object, object, object, float, float, float, object], delimiter = ',', encoding = "latin1")

quas = []

smpixes = [501, 301, 501, 301, 301, 201, 301, 301, 301, 301, 301, 301, 301, 301, 301,301]
names1 = ["PSOJ323+12"]
names2 = ["PSOJ065-26", "PSOJ011+09"]
names3 = ["PSOJ231-20"]
for k in range(vals.shape[0]):
    decoded = [el.decode() if type(el)==type(vals[k][0]) else el for el in vals[k]]
    name, file_nir, file_vis, redshift, dv, magnitude, inst = decoded
    if name == 'J1509-1749':
        quas.append(Quasar(name, "complete_sample/spectra/"+file_nir, "complete_sample/spectra/"+file_vis, magnitude, redshift, typeof = inst, sm_pix = smpixes[k], load = False, draw= True)) 

for quasar in quas:
    quasar.fit_pipeline()
    plt.close('all')

