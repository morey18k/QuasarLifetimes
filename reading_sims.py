import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import os 
from quasar_stack_class import QuasarStack
import pickle

path = "/mnt/quasar/aceilers/rtcode/"

names = ["J1148","J036", "J0842", "J158-14", "J159", "J1143", "J1306", "J1319", "J1335", "J217", "J1509", "J359-06", "J011+09", "J065-26", "J323", "J231"]

names_prox_long = ['SDSSJ1143+3803', 'SDSSJ1335+3533', 'SDSSJ1148+5251']
names_prox_short = ['J1143', 'J1335', 'J1148']

names_long = ['SDSSJ1148+5251', 'PSOJ036+03',  'J0842+1218', 'PSOJ158-14', 'PSOJ159-02', 'SDSSJ1143+3803','J1306+0356', 'J1319+0950', 'SDSSJ1335+3533', 'PSOJ217-16', 'J1509-1749', 'PSOJ359-06', 'PSOJ011+09','PSOJ065-26', 'PSOJ323+12', 'PSOJ231-20']

redshift = [6.4189, 6.5412, 6.0763, 6.0681, 6.3809, 5.8367, 6.0337, 6.133, 5.9012, 6.1498, 6.1225, 6.1719, 6.4693, 6.1877, 6.5872, 6.5864]


tqs = np.linspace(1, 8.9, 80)
pixel_scale = 4 # km/s
dv = np.arange(-500, 11500, pixel_scale) # in km/s
lya = 1215.6701
c = 299792.458

N_skew = 1000

dlambda = lya - dv/c * lya 

freq = 1/np.abs(dlambda[2]-dlambda[1])

snradj = np.load('snradj.npy')

stack11_flux =[]
stack14_flux = []
stack15_flux = []



for k, t in enumerate(tqs):
	flux = np.zeros_like(dlambda)
	for j in range(16):	
		#name = names[j]
		#z = redshift[j]
		for i in range(N_skew):
			flux += np.fromfile(path+'{2}/rt_{2}_{0}_tq{1}_tlya.bin'.format(i, '%.3f' %t, name))        
		if j==14:	
			stack15_flux.append(flux/15000)

tot_models = np.array(stack15_flux)
np.save("all_models.npy", tot_models)
np.save("wavelength_models.npy", dlambda)



	
