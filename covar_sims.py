import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import os 
from quasar_stack_class import QuasarStack
import pickle
import os


np.random.seed(2)
path = "/mnt/quasar/aceilers/rtcode/"

names = ["J1148","J1143","J1335","J036", "J0842", "J158-14", "J159",  "J1306", "J1319",  "J217", "J1509", "J359-06", "J011+09", "J065-26", "J323", "J231"]

names_prox_long = ['SDSSJ1143+3803', 'SDSSJ1335+3533', 'SDSSJ1148+5251']
names_prox_short = ['J1143', 'J1335', 'J1148']

names_long = ['SDSSJ1148+5251', 'SDSSJ1143+3803','SDSSJ1335+3533','PSOJ036+03',  'J0842+1218', 'PSOJ158-14', 'PSOJ159-02', 'J1306+0356', 'J1319+0950',  'PSOJ217-16', 'J1509-1749', 'PSOJ359-06', 'PSOJ011+09','PSOJ065-26', 'PSOJ323+12', 'PSOJ231-20']

redshift = [6.4189, 5.8367, 5.9012, 6.5412, 6.0763, 6.0681, 6.3809, 6.0337, 6.133, 6.1498, 6.1225, 6.1719, 6.4693, 6.1877, 6.5872, 6.5864]

lower = 1177
higher = 1218 
bin_size = 0.6
num_bins = int((higher-lower)/bin_size) 
wave_grid = np.linspace(lower, higher, num_bins+1) 


tqs = np.linspace(1, 8.9, 80)
pixel_scale = 4 # km/s
dv = np.arange(-500, 11500, pixel_scale) # in km/s
lya = 1215.6701
c = 299792.458

N_skew = 1000

dlambda = lya - dv/c * lya 
freq = 1/np.abs(dlambda[2]-dlambda[1])
snradj = dict(np.load('snradj.npy'))
snr = np.array([float(snradj[q]/np.sqrt(freq)) for q in names_long])

redoff = dict(np.load('redoff.npy'))

num_stack = 10000

name_long = names_long[0]
zq = redshift[0]
dz_q = redoff[name_long]
num_samples = 10

w1 = np.load('covar_prox_truncated_10r_6b_wav.npy')
c1 = np.load('covar_prox_truncated_10r_6b_covar.npy')

w2 = np.load('covar_prox_10r_6b_wav.npy')
c2= np.load('covar_prox_10r_6b_covar.npy')

cont_err_b = np.random.multivariate_normal(np.zeros(c2.shape[0]), c2, (num_stack,12))
cont_err_b_prox = np.random.multivariate_normal(np.zeros(c1.shape[0]), c1, (num_stack,3))

cont_err_full = np.concatenate([cont_err_b_prox, cont_err_b], axis  = 1)


multiplier = 1.0 + 100*np.random.randn(num_stack,16)/(299792.458*(1.0+np.array(redshift)[None,:]))

flux = np.zeros_like(dlambda)
skws = np.random.randint(1000, size = (num_stack, 15))
low = np.argmin(np.abs(dlambda - 1177.2))
high = np.argmin(np.abs(dlambda - 1219.9))
lam_trunc = dlambda[high:low]
idcs = np.argsort(lam_trunc)
noise = np.random.normal(loc = 0.0, scale = 1.0/snr[None,:,None], size = (num_stack,snr.shape[0], lam_trunc.shape[0]))
#fig = plt.figure(figsize = (10, 35))


waves_flat = np.zeros((num_stack, lam_trunc.shape[0]*15))
fluxes_flat = np.zeros((num_stack, lam_trunc.shape[0]*15))

ws = np.ma.empty((num_stack, num_bins))
fs = np.ma.empty((num_stack, num_bins))

#plt.figure()
t =  tqs[-1]

covar_path = 'covar_all_matrix_{0}.npy'.format(bin_size)
fs_path ='fs_all_{0}.npy'.format(bin_size) 
load_covar = os.path.isfile(covar_path)
load_fs = os.path.isfile(fs_path)

if load_fs:
	all_fs = np.load(fs_path)
else:
	all_fs = np.zeros((80, num_stack, num_bins))

if load_covar:
	all_covar = np.load(covar_path)
else:
	all_covar = np.zeros((80, num_bins, num_bins))

print('load_fs', load_fs)
print('load_covar', load_covar)

for k,t in enumerate(tqs):
	for i in range(num_stack):
		for j in range(15):	
			name_short = names[j]
			name_long = names_long[j]
			zq = redshift[j]
			dz_q = redoff[name_long]
			

			if name_long in names_prox_long:             
				wave_pca_b, covar_pca_b = w1, c1
			else:	
				wave_pca_b, covar_pca_b = w2, c2
			bmask = np.where((wave_pca_b*(1+zq+dz_q)/(1+zq)<1220) & (wave_pca_b>1177))
			nb_wav = wave_pca_b[bmask]*(1+zq+dz_q)/(1+zq)
			dCont = cont_err_full[i][j][bmask]
			dC = np.interp(lam_trunc,nb_wav, dCont)
			
			flux = np.fromfile(path+'{2}/rt_{2}_{0}_tq{1}_tlya.bin'.format(skws[i][j], '%.3f' %t, name_short))[high:low]        
			flux_int =np.interp(nb_wav, lam_trunc[idcs], flux[idcs])
			new_flux = flux*(1+dC)+ noise[i][j]	
			#ax = fig.add_subplot(15, 1, j)
			#ax.plot(lam_trunc*multiplier[i][j], new_flux, drawstyle = 'steps', label = name_short)	
			#ax.plot(lam_trunc, flux, drawstyle = 'steps', linestyle = '-.')	
			#ax.legend(loc = 'lower left')
			waves_flat[i][j*lam_trunc.shape[0]:(j+1)*lam_trunc.shape[0]] = lam_trunc
			fluxes_flat[i][j*lam_trunc.shape[0]:(j+1)*lam_trunc.shape[0]] = flux*(1+dC)

		idx = np.searchsorted(wave_grid, waves_flat[i], 'right')-1

		bad_mask = ((idx == (-1)) | (idx == (num_bins)) ) 

		idx[bad_mask]=num_bins

		nused = (np.bincount(idx, minlength = num_bins+1)[:-1])

		ws_total = (np.bincount(idx, minlength = num_bins+1, weights = waves_flat[i])[:-1])
		fs_total = (np.bincount(idx, minlength = num_bins+1, weights = fluxes_flat[i])[:-1])

		wave_stack = (nused>0.0)*ws_total/(nused+(nused==0.0))
		flux_stack = (nused>0.0)*fs_total/(nused+(nused==0.0))
		mask_stack = (nused==0)

		ws[i] = np.ma.array(wave_stack, mask = mask_stack) 
		fs[i] = np.ma.array(flux_stack, mask = mask_stack)
		#plt.plot(ws[i], fs[i], color = 'steelblue', alpha = 0.2)
	print(t)


	#plt.xlim(1177, 1220)
	covar = np.cov(fs, rowvar = False) 
	
	all_fs[k] = fs.data
	
	act_covar = covar[:-1, :-1]
	
	all_covar[k] = covar


	#plt.savefig('stack_tq{1}_{0}.pdf'.format(bin_size, t))
	#fig = plt.figure()
	#ax = plt.axes()
	#im = ax.pcolormesh(wave_grid, wave_grid, covar)
	#plt.ylim(wave_grid[-1], wave_grid[0])
	#cbar = fig.colorbar(im, ax = ax)
	#cbar.set_label('Covariance')

#	plt.savefig('covar_tq{1}_{0}.pdf'.format(bin_size, t))
	np.save('covar_all_matrix_{0}.npy'.format(bin_size), all_covar)
	np.save('fs_all_{0}.npy'.format(bin_size), all_fs)
		
#	fig = plt.figure()
#	ax = plt.axes()
#	im = ax.pcolormesh(wave_grid, wave_grid, corr)
#	plt.ylim(wave_grid[-1], wave_grid[0])
#	cbar = fig.colorbar(im, ax = ax)
#	cbar.set_label('Correlation')

#	plt.savefig('corr_tq{1}_{0}.pdf'.format(bin_size, t))

