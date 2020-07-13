from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import math
from _cont_src import fit_sdss_cont
import emcee
import corner
import scipy
import scipy.interpolate as interpol
from astropy.stats import sigma_clip
import pickle
import lmfit
from multiprocessing import Pool
from helper_functions import *
import os
import time

class Quasar:
    def __init__(self, name, nir_file, vis_file, magnitude, redshift, typeof= 'xshooter', load = False, draw = True, sm_pix = 301, order = 3):
        self.type = typeof
        self.sm_pix = sm_pix #number of smoothing pixels in smoothing function
        self.order = order # order of polynomial filter
        self.name = name
        self.vis_file = vis_file
        self.nir_file = nir_file
        self.magnitude = magnitude
        self.exists = os.path.exists(self.name)
        if not self.exists:
            os.mkdir(self.name)
            os.chdir(self.name)
            os.mkdir("plots")
            os.chdir("..")

        self.redshift = redshift
        self.load = load
        self.draw = draw
        l = self.load
        if self.load:
            new_obj = pickle.load(open(self.name+"/"+self.name+'.pickle','rb'))
            self.__dict__.update(new_obj.__dict__)
            self.load = l
            self.mcmc_load()
            self.sample_analysis()
        else:
            self.read()
            self.match()
            self.mask()
            self.normalize()
            self.clip()
            self.pca_load()
            self.mcmc_init()
            with open(self.name+"/"+self.name+'.pickle', 'wb') as filename:
                   pickle.dump(self, filename)
    
    def fit_pipeline(self):
            #for item in self.__dict__.items():
                #print(item[0], type(item[1]))
            self.mcmc_fit()
            self.sample_analysis()
            self.plot_mc_results()
            #self.autocorr()
    
    def read(self):
        if self.type == 'fire':
            self.fire_path  = self.nir_file
            hdu = fits.open(self.fire_path) 
            header = hdu[0].header
            self.um_flux = hdu[0].data
            self.um_std = fits.getdata(self.fire_path[:-6]+'E'+self.fire_path[-5:])
            self.um_ivar = 1./self.um_std**2
            self.um_wavelengths = 10. ** (header['CRVAL1'] + header['CDELT1'] * np.arange(0, header['NAXIS1']))
            #self.smoothed = scipy.signal.savgol_filter(self.um_flux, self.sm_pix, self.order)
            self.smoothed = smooth(self.um_flux, self.um_ivar)
            return
        
        if self.type == 'solo':
            vis_hdul = fits.open(self.vis_file)
            self.vis_data = vis_hdul[1].data
            self.um_wavelengths = vis_hdul[2].data
            self.um_std= vis_hdul[1].data
            self.um_flux= vis_hdul[0].data
            self.um_ivar = self.um_std**(-2.0)
            self.smoothed = smooth(self.um_flux, self.um_ivar)
            return
            
        vis_hdul = fits.open(self.vis_file)
        nir_hdul = fits.open(self.nir_file)
        self.vis_data = vis_hdul[1].data
        self.vis_columns = vis_hdul[1].columns
        
        self.vis_wavstr  = self.vis_columns[0].name
        self.vis_flstr = self.vis_columns[1].name
        self.vis_ivarstr = self.vis_columns[2].name
        try:
            self.vis_stdstr = self.vis_columns[3].name 
            labler = 'Noise'
        except:
            labler = "1/\\sqrt(\\sigma^{-2})"
        vwav = self.vis_data[self.vis_wavstr]
        vivar = self.vis_data[self.vis_ivarstr]
        vflux = self.vis_data[self.vis_flstr]
        vsmooth,o  = ivarsmooth(vflux, vivar, 21)
        vstdsm, o= ivarsmooth(vivar**(-0.5), vivar, 21)
        try:
            vstd = self.vis_data[self.vis_stdstr]
            vstdsm,o = ivarsmooth(vstd, vivar, 21)
        except:
            pass
        print("vis ivarstring", self.vis_ivarstr)     
        self.nir_data = nir_hdul[1].data
        self.nir_columns = nir_hdul[1].columns
        
        self.nir_wavstr = self.nir_columns[0].name
        self.nir_flstr = self.nir_columns[1].name
        self.nir_ivarstr = self.nir_columns[2].name
        print("nir ivarstring", self.nir_ivarstr)     

        self.vis_wav = self.vis_data[self.vis_wavstr]
        self.nir_wav = self.nir_data[self.nir_wavstr]

        self.vis_data_smoothed = scipy.signal.savgol_filter(self.vis_data[self.vis_flstr], self.sm_pix, self.order)
        self.vis_data_smoothed = smooth(self.vis_data[self.vis_flstr], self.vis_data[self.vis_ivarstr])

        self.nir_data_smoothed = scipy.signal.savgol_filter(self.nir_data[self.nir_flstr], self.sm_pix,self.order)
        self.nir_data_smoothed = smooth(self.nir_data[self.nir_flstr], self.nir_data[self.nir_ivarstr])
        # plots of visible spectrum and smoothed version, along with nir spectrum and
        # smoothed version
        if self.draw:
            plt.figure()
            plt.plot(self.vis_wav, self.vis_data[self.vis_flstr])
            plt.plot(self.vis_wav, self.vis_data_smoothed)
            plt.ylim(-6,6)
            plt.title("Visible Spectrum")
            plt.figure()
            plt.title("NIR Spectrum")
            plt.plot(self.nir_wav, self.nir_data[self.nir_flstr])
            plt.plot(self.nir_wav, self.nir_data_smoothed)
            plt.ylim(-6,6)
            plt.show()

    def match(self):
        if self.type=='fire' or self.type == 'solo':
            if self.draw: 
                #plots the full spectrum, now continuously matched
                plt.plot(self.um_wavelengths, self.um_flux)
                um_flux_smoothed = scipy.signal.savgol_filter(self.um_flux, self.sm_pix,
                        self.order)
                um_flux_smoothed = smooth(self.um_flux, self.um_ivar)
                plt.plot(self.um_wavelengths, um_flux_smoothed)
                plt.title(self.name +" Full Smoothed Spectrum")
                plt.show()
            return



        last_vis = self.vis_data[self.vis_wavstr][-1]
        first_nir = self.nir_data[self.nir_wavstr][0]

        index_nir = np.argmin(np.abs(self.nir_data[self.nir_wavstr]-last_vis))
        index_vis = np.argmin(np.abs(self.vis_data[self.vis_wavstr]-first_nir))

        
        special_wavelength = 10170
        if self.type=='deimos': 
            special_wavelength =self.nir_data[self.nir_wavstr][index_nir//2] 

        nir_init = np.argmin(np.abs(self.nir_data[self.nir_wavstr]-special_wavelength))
        opt_fin = np.argmin(np.abs(self.vis_data[self.vis_wavstr]-special_wavelength))
        
        if self.type == 'deimos':
            self.um_wavelengths = np.concatenate([self.vis_data[self.vis_wavstr][:opt_fin],self.nir_data[self.nir_wavstr][nir_init:]])
            self.um_flux = np.concatenate([self.vis_data[self.vis_flstr][:opt_fin],self.nir_data[self.nir_flstr][nir_init:]])
            self.um_ivar = np.concatenate([self.vis_data[self.vis_ivarstr][:opt_fin],
            self.nir_data[self.nir_ivarstr][nir_init:]])
            self.um_std = np.concatenate([(self.vis_data[self.vis_ivarstr][:opt_fin])**(-0.5),(self.nir_data[self.nir_ivarstr][nir_init:])**(-0.5)])
            if self.draw: 
                #plots the full spectrum, now continuously matched
                plt.plot(self.um_wavelengths, self.um_flux)
                um_flux_smoothed = scipy.signal.savgol_filter(self.um_flux, self.sm_pix,
                        self.order)
                um_flux_smoothed = smooth(self.um_flux, self.um_ivar)
                plt.plot(self.um_wavelengths, um_flux_smoothed)
                plt.ylim(-1,4)
                plt.title(self.name + " Full Smoothed Spectrum")
                plt.show()
            return
        
        nir_indices = np.where(np.abs(self.nir_data[self.nir_wavstr]-special_wavelength)<100)
        vis_indices = np.where(np.abs(self.vis_data[self.vis_wavstr]-special_wavelength)<100)
        nir_mean = np.mean(self.nir_data_smoothed[nir_indices])
        vis_mean = np.mean(self.vis_data_smoothed[vis_indices])
        
        self.nirnoise = smooth((vis_mean/nir_mean)*(self.nir_data[self.nir_ivarstr]**(-0.5)), self.nir_data[self.nir_ivarstr])
        self.visnoise = smooth((self.vis_data[self.vis_ivarstr]**(-0.5)), self.vis_data[self.vis_ivarstr])
        
        
        vis_interpfunc = scipy.interpolate.interp1d(self.vis_data[self.vis_wavstr][index_vis:], self.visnoise[index_vis:])
        nir_interpfunc = scipy.interpolate.interp1d(self.nir_data[self.nir_wavstr][:index_nir], self.visnoise[:index_nir])
        
        wav_grid = np.linspace(first_nir+10, last_vis-10, 5000)
        interped_vis = vis_interpfunc(wav_grid)
        interped_nir = nir_interpfunc(wav_grid)

        
        if self.draw:
            #plots of the 1 micron of overlap between visible and nir 
            plt.plot(self.nir_data[self.nir_wavstr][:index_nir], self.nir_data_smoothed[:index_nir],label = "NIR")
            plt.plot(self.vis_data[self.vis_wavstr][-index_nir:], self.vis_data_smoothed[-index_nir:], label = "VIS")
            plt.plot(self.nir_data[self.nir_wavstr][:index_nir], smooth(self.nir_data[self.nir_ivarstr]**(-0.5),self.nir_data[self.nir_ivarstr], pix = 20)[:index_nir],label = "NIR STD")
            plt.plot(self.vis_data[self.vis_wavstr][-index_nir:], smooth(self.vis_data[self.vis_ivarstr]**(-0.5),self.vis_data[self.vis_ivarstr], pix = 20)[-index_nir:],label = "VIS STD")
            
            plt.legend()
            plt.title(self.name+' Overlap Region')
            plt.show()
        if self.draw:
            #plots the data to make sure things are matched up right
            plt.plot(self.vis_data[self.vis_wavstr][vis_indices], self.vis_data_smoothed[vis_indices],label = "VIS")
            plt.plot(self.vis_data[self.vis_wavstr][vis_indices], self.visnoise[vis_indices],label = "vis unc")
            plt.plot(self.nir_data[self.nir_wavstr][nir_indices],
                    (vis_mean/nir_mean)*self.nir_data_smoothed[nir_indices],label = "NIR ADJ")
            plt.plot(self.nir_data[self.nir_wavstr][nir_indices],
                   (vis_mean/nir_mean)*self.nirnoise[nir_indices],label = "nir unc")
            plt.legend()
            plt.title(self.name +" Adjusted Smoothed Flux for both NIR and VIS")
            plt.ylim(-1,5)
            plt.figure()
            #plots the matching conditions between the sides of the spectra
            plt.plot(self.vis_data[self.vis_wavstr][:opt_fin],self.vis_data_smoothed[:opt_fin])
            plt.plot(self.nir_data[self.nir_wavstr][nir_init:],(vis_mean/nir_mean)*self.nir_data_smoothed[nir_init:])
            #plt.plot(self.vis_data[self.vis_wavstr][:opt_fin], visnoise[:opt_fin],label = "vis unc")
            #plt.plot(self.nir_data[self.nir_wavstr][nir_init:],nirnoise[nir_init:],label = "nir unc")
            plt.ylim(-1,4)
            plt.title(self.name +" Plotting the Matched Parts of the Spectra")
            plt.show()

        #concatenates the data into one big array
        self.um_wavelengths = np.concatenate([self.vis_data[self.vis_wavstr][:opt_fin],self.nir_data[self.nir_wavstr][nir_init:]])
        self.um_flux = np.concatenate([self.vis_data[self.vis_flstr][:opt_fin],
        (vis_mean/nir_mean)*self.nir_data[self.nir_flstr][nir_init:]])
        self.um_ivar = np.concatenate([self.vis_data[self.vis_ivarstr][:opt_fin],
        ((vis_mean/nir_mean)**(-2.0))*self.nir_data[self.nir_ivarstr][nir_init:]])
        self.um_std = np.concatenate([(self.vis_data[self.vis_ivarstr][:opt_fin])**(-0.5),
        (vis_mean/nir_mean)*(self.nir_data[self.nir_ivarstr][nir_init:])**(-0.5)])
        if self.draw: 
            #plots the full spectrum, now continuously matched
            um_flux_smoothed = scipy.signal.savgol_filter(self.um_flux, self.sm_pix,
                    self.order)
            um_flux_smoothed = smooth(self.um_flux, self.um_ivar)
            filtering = np.ones(20)
            weights = np.correlate(self.um_ivar, filtering, mode = 'same')
            smoothed = np.correlate(self.um_flux*self.um_ivar, filtering, mode = 'same')/weights
            smoothed_std = np.correlate(self.um_std*self.um_ivar, filtering, mode = 'same')/weights
            plt.plot(self.um_wavelengths, smoothed, color = 'black', drawstyle = 'steps')
            plt.plot(self.um_wavelengths, smoothed_std, color = 'gray', drawstyle = 'steps')
            plt.ylim(-1,4)
            plt.title(self.name +" Full Smoothed Spectrum")
            plt.show()

    def mask(self):
        mask1 = np.logical_and(self.um_wavelengths>13000,self.um_wavelengths<15200)
        mask2 = np.logical_and(self.um_wavelengths>17500,self.um_wavelengths<20000)
        if self.type=='solo':
            mask3 = (self.um_wavelengths > 22500) | ((np.abs(self.um_flux)/np.amax(self.um_flux))<1e-15)
        else:
            mask3  = (self.um_wavelengths >22500)

        tot_mask = np.logical_or(np.logical_or(mask1, mask2),mask3)

        #masks the unnormalized flux in the telluric regions using numpy masked arrays
        self.wav_obs = np.ma.masked_where(tot_mask, self.um_wavelengths)
        self.un_flux = np.ma.masked_where(tot_mask, self.um_flux)
        self.un_ivar = np.ma.masked_where(tot_mask, self.um_ivar)
        self.un_std = np.ma.masked_where(tot_mask, self.um_std)
        self.wav = np.ma.masked_where(tot_mask, self.um_wavelengths/(1+self.redshift))


        
        #smooths the data again, using the same filter, but now with the data masked
        self.un_flux_smoothed = scipy.signal.savgol_filter(self.un_flux,
                self.sm_pix, self.order)
        if self.draw:
            #plots the masked, unnormalized flux in the rest frame of the quasar
            plt.plot(self.wav,self.un_flux)
            plt.plot(self.wav,self.un_flux_smoothed)
            if self.type!='fire':
                plt.ylim(-1,5) 
            plt.title(self.name +" Masked, Un-Normalized Flux for Full Spectrum")
            plt.show()

    def normalize(self):
#normalizes the flux at 1290 angstroms
        self.normalization = np.median(np.array(self.un_flux[np.argmin(np.abs(1290-2.5-self.wav)):np.argmin(np.abs(1290+2.5-self.wav))]))
        self.n_flux = self.un_flux/self.normalization
        self.n_std = self.un_std/self.normalization
        self.n_ivar = self.un_ivar * self.normalization**2.0
        self.n_flux_smoothed = self.un_flux_smoothed/self.normalization

        #focuses on the part of the spectrum between 1220 and 2850 Angstroms
        low_red = np.argmin(np.abs(self.wav.data-1220))
        if self.type == 'deimos' or self.type == 'solo':
            high_red = np.argmin(np.abs(self.wav.data-1490))
        else:
            high_red = np.argmin(np.abs(self.wav.data-2850))
        self.red_obs = self.wav_obs[low_red:high_red]
        self.red_wav = self.wav[low_red:high_red]
        self.red_flux = self.n_flux[low_red:high_red]
        self.red_std = self.n_std[low_red:high_red]
        self.red_ivar = self.n_ivar[low_red:high_red]
        self.red_smoothed = self.n_flux_smoothed[low_red:high_red]

    def clip(self):
        #smoothing spline to sigma clip at 3 sigma
        #see fit_sdss_cont.c for more documentation on each of these parameters
        #may need to mess around with these parameters
        deltapix1 = 150
        deltapix2 = 150
        minpix = 501
        slopethresh = 0.033
        fluxthresh = 0.99
        Fscale = 1.0
        cont = np.zeros_like(self.red_flux)
        nfit = self.red_flux.shape[0]
        fit_sdss_cont(self.red_wav, self.red_flux, self.red_std, nfit, self.redshift, deltapix1, deltapix2, minpix, slopethresh, fluxthresh, cont, Fscale)
        if self.draw:
            #plots red-side flux, along with the smoothed version using savgol filter and
            #using the smoothing spline

            plt.plot(self.red_obs, self.red_flux, label = 'Raw Flux')
            plt.plot(self.red_obs, cont, label='Cubic Spline')
            plt.plot(self.red_obs, self.red_smoothed, label = 'Savgol Smoothed')
            plt.legend()
            plt.ylim(-1,4)
            plt.title(self.name +" Cubic Spline Used for Sigma Clipping")
            plt.show()
            plt.close()
            time.sleep(1)
#sigma clips the data to exclude outliers at 3sigma, and then gets only the
#signficant data from the masked arrays

        self.clipped = sigma_clip(self.red_flux-self.red_smoothed, sigma = 3)

        self.rc_flux = self.red_flux[~self.clipped.mask].data
        self.rc_obs = self.red_obs[~self.clipped.mask].data
        self.rc_wav = self.red_wav[~self.clipped.mask].data
        self.rc_std = self.red_std[~self.clipped.mask].data
        self.rc_ivar = self.red_ivar[~self.clipped.mask].data
        rc_smoothed = self.red_smoothed[~self.clipped.mask].data

    def pca_load(self):
#loads in the pca components, where each of the components in pca_comp_r and
#pca_comp_b are 2d numpy arrays 
        if self.type == 'deimos' or self.type == 'solo':
            self.wave_pca_r, self.pca_comp_r, self.wave_pca_b, self.pca_comp_b, self.X_proj = pickle.load(open('complete_sample/pca_prox_truncated_10r_6b.pckl','rb'))
        else:
            self.wave_pca_r, self.pca_comp_r, self.wave_pca_b, self.pca_comp_b, self.X_proj = pickle.load(open('pca_prox_10r_6b.pckl','rb'))

        #print('min of interpol', np.amin(self.wave_pca_b))
#performs linear interpolation between the points in the PCA components to get
#accurate model function
        self.pca_comp_r_int = interpol.interp1d(self.wave_pca_r, self.pca_comp_r, kind = 'linear')
        self.pca_comp_b_int = interpol.interp1d(self.wave_pca_b, self.pca_comp_b, kind = 'linear')

    def mcmc_init(self):
        #sets up the dimensions, number of walkers, number of steps, and bounds on the
        #coefficients
        self.nwalkers = 100
        self.nsteps = 7000
        xx = np.array([(-56.059703706225235, 56.059703706225235), (-14.78144804812042, 14.78144804812042), (-7.067877973585391, 7.067877973585391), (-6.11902457355148, 6.11902457355148), (-4.4107048087866225, 4.4107048087866225), (-3.779206428050559, 3.779206428050559), (-2.2731500297660006, 2.2731500297660006), (-3.354336106021761, 3.354336106021761), (-1.343510638564665, 1.343510638564665), (-1.4161209039712808, 1.4161209039712808),(-1.4161209039712808, 1.4161209039712808),(-1.4161209039712808, 1.4161209039712808),(-1.4161209039712808, 1.4161209039712808),(-1.4161209039712808, 1.4161209039712808),(-1.4161209039712808, 1.4161209039712808)])
        self.ndim = 11
        self.coef_min = np.ones((self.ndim,))
        self.coef_max = np.ones((self.ndim,))
        self.coef_min[:10] = xx[:10, 0]
        self.coef_max[:10] = xx[:10, 1]
        # prior on z
        self.coef_min[-1] = -0.01
        self.coef_max[-1] = 0.03
        if self.name == "PSOJ323+12":
            self.coef_min[-1] = -0.5
            self.coef_max[-1] = 0.5


    def mcmc_fit(self):
        p0 = GetInitialPositions(self.nwalkers, self.coef_min, self.coef_max, self.ndim)

        self.max_l = maximum_likelihood(self.redshift, self.pca_comp_r_int, self.rc_obs, self.rc_flux, self.rc_ivar, self.coef_min, self.coef_max)

        if not np.isfinite(lnprior_pca(self.max_l, self.coef_min, self.coef_max)):
            print(self.max_l)
            print(self.coef_max)
            raise ValueError("Maximum Likelihood Estimate Violates 3 Sigma Bounds")

        #max_l = np.load("means.npy")

        p0 = self.max_l + 1e-5*np.random.normal(size = (self.nwalkers, self.ndim))

        filename = self.name+"/"+self.name+".h5"
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(self.nwalkers, self.ndim)


        # We'll track how the average autocorrelation time estimate changes
        index = 0
        max_checks = int(self.nsteps/5000)
        autocorr = np.empty((max_checks, self.ndim))

        # This will be useful to testing convergence
        old_tau = np.inf
        with Pool() as pool:
            self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, lnprob_pca, args =
                    (self.coef_min, self.coef_max, self.redshift,
                        self.pca_comp_r_int, self.rc_obs, self.rc_flux,
                        self.rc_ivar), pool = pool, backend= backend)
            # Now we'll sample for up to max_n steps
            for sample in self.sampler.sample(p0,
                iterations=self.nsteps,rstate0=np.random.get_state(), progress=True):
                # Only check convergence every 100 steps
                if self.sampler.iteration % 5000:
                    continue

                # Compute the autocorrelation time so far
                # Using tol=0 means that we'll always get an estimate even
                # if it isn't trustworthy
                tau = self.sampler.get_autocorr_time(tol=0)
                autocorr[index] = tau
                index += 1

                # Check convergence
                converged = np.all(tau * 100 < self.sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                if converged:
                    break
                old_tau = tau    

    def mcmc_load(self):
        self.sampler = emcee.backends.HDFBackend(self.name+"/"+self.name+".h5", read_only = True)
    
    def autocorr(self):
        print("Calculating Autocorrelation Time")
        tau = self.sampler.get_autocorr_time()
        print("Integrated Autocorrelation Time", tau)



    def sample_analysis(self):
        self.walker_samples = self.sampler.get_chain()
        self.samples = self.sampler.get_chain(flat = True, discard =
                int(self.nsteps/10), thin = 10)


        #blue side model
 #       print('min',np.amin(self.wav))
        low_blue = np.argmin(np.abs(self.wav-1177))
        high_blue = np.argmin(np.abs(self.wav-1220))
        bv_obs = self.wav_obs[low_blue:high_blue].data
        rv_obs = self.red_obs.data
        #shifts the 
        z_pca = self.redshift + np.mean(self.samples[:, -1])
        self.redoff = np.mean(self.samples[:, -1])
        pca_coeff_red = np.mean(self.samples[:, :-1], axis = 0)
        pca_coeff_blue = 1.0 * np.dot(np.append(1.0, pca_coeff_red), self.X_proj)
  #      print(np.amax(rv_obs)/(1+z_pca))
        pca_model_blue, pca_model_red = ModelSpectra(z_pca, bv_obs,
                pca_coeff_blue, self.pca_comp_b_int, rv_obs, pca_coeff_red,
                self.pca_comp_r_int)
        log_pca_model = np.hstack([pca_model_blue, pca_model_red])
        self.pca_model = np.exp(log_pca_model)
        self.obs_wav_model = np.hstack([bv_obs, rv_obs])

        low_model = np.argmin(np.abs(self.wav_obs.data - np.amin(self.obs_wav_model.data)))
        high_model = np.argmin(np.abs(self.wav_obs.data - np.amax(self.obs_wav_model.data)))

        pca_int = interpol.interp1d(self.obs_wav_model, self.pca_model, kind = 'linear')
        

        self.cont_norm = self.n_flux[low_model:high_model]/pca_int(self.wav_obs.data[low_model:high_model])
        self.cont_std  = self.n_std[low_model:high_model]/pca_int(self.wav_obs.data[low_model:high_model])
        self.cont_wav = self.wav[low_model:high_model]
        if not self.load: 
            plt.figure(figsize = (300, 10))
            ax1 = plt.subplot(211)
            plt.plot(self.wav_obs/(1+self.redshift), self.n_flux, label = 'Raw Flux')
            plt.plot(self.obs_wav_model/(1+self.redshift), self.pca_model, label = 'PCA Model')
            plt.ylim(-3,6)
            plt.legend()
            plt.subplot(212, sharex = ax1)
            plt.plot(self.cont_wav, self.cont_norm)
            plt.ylim(-0.5,2.0)
            plt.grid()
            plt.savefig(self.name+"/"+"plots/"+self.name+"_fit.pdf")
            #plt.savefig("continuum_fit_results/fit_plots/"+self.name+"_fit.pdf")

    def plot_mc_results(self):
        fig = plt.figure()



        for k in range(self.ndim):
            if k == self.ndim-1:
                plt.ylabel("Redshift Offset")
            else:
                plt.ylabel("PCA Component "+str(k+1))
            plt.xlabel("Number of Steps")
            plt.plot(self.walker_samples[:,:,k],'k', alpha = 0.3)
            plt.savefig(self.name+"/"+'plots/mc'+str(k+1)+'analysis.pdf')
            plt.figure()

        corner.corner(self.samples, labels = ['PCA Component '+str(k+1) for k in range(self.samples.shape[1]-1)]+['Reshift Offset'], bins = 50)
        plt.savefig(self.name+"/plots/"+self.name+"_corner.pdf")
        plt.savefig("continuum_fit_results/corner_plots/"+self.name+"_corner.pdf")
