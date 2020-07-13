import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import os
import time

class QuasarStack:
    def __init__(self, quasar_list, low, high, bin_size, wave_list = None, flux_list = None):
        if quasar_list is not None:
            self.num_quasar = len(quasar_list)
            lengths = np.array([q.cont_wav.shape[0] for q in quasar_list])
            max_len =  lengths.max()
            padding = max_len - lengths
            wavs = [q.cont_wav for q in quasar_list]
            self.namelist = [q.name for q in quasar_list]
            flxs = [q.cont_norm for q in quasar_list]
            varis = [q.cont_std**2.0 for q in quasar_list]
            masks = [q.cont_wav.mask for q in quasar_list]
        if wave_list is not None and flux_list is not None:
            self.wave_list = wave_list
            self.flux_list = flux_list
            self.num_quasar = len(self.wave_list)
            lengths = np.array([wav.shape[0] for wav in self.wave_list])
            max_len =  lengths.max()
            padding = max_len - lengths
            wavs = self.wave_list
            flxs = self.flux_list
            masks = [np.zeros_like(wv, dtype = bool) for wv in wavs]
            varis = [np.zeros_like(wv) for wv in wavs]
            
        self.lower = low
        self.upper = high
        self.bin_size = bin_size
        self.num_bins = int((self.upper-self.lower)/self.bin_size)
        waves = []
        fluxes = []
        variances = []
        maskers = []
        self.wave_grid = np.linspace(self.lower, self.upper, self.num_bins+1)
        for i in range(self.num_quasar):
            #quasar = self.quasar_list[i]
            #plt.plot(quasar.cont_wav, quasar.cont_norm)
            waves.append(np.pad(wavs[i],pad_width = (0,padding[i]), mode = 'constant'))
            fluxes.append(np.pad(flxs[i],pad_width = (0,padding[i]), mode = 'constant'))
            variances.append(np.pad(varis[i],pad_width = (0,padding[i]), mode = 'constant'))
            maskers.append(np.pad(np.logical_not(masks[i]), pad_width = (0,padding[i]),mode = 'constant', constant_values=False))
	
        self.waves =np.array(waves)
        ascending = np.all(self.waves[:, 0]<self.waves[:,1])
        descending = np.all(self.waves[:, 0]>self.waves[:,1])
        if ascending:
            hs = abs(np.amax(np.argmin(np.abs(self.waves-self.upper), axis = 1)) + 5)
            ls = abs(np.amin(np.argmin(np.abs(self.waves-self.lower), axis = 1)) - 5)	
        if descending:
            ls = abs(np.amin(np.argmin(np.abs(self.waves-self.upper), axis = 1)) + 5)
            hs = abs(np.amax(np.argmin(np.abs(self.waves-self.lower), axis = 1)) - 5)
        self.waves = self.waves[:, ls:hs]
        self.fluxes = np.array(fluxes)[:, ls:hs]
        self.variances = np.array(variances)[:, ls:hs]
        self.masks = np.array(maskers)[:, ls:hs]
        self.wave_stack, self.flux_stack = self.stack(self.waves, self.fluxes, self.masks, self.wave_grid)
    
    def bootstrap(self, num_trials):
        resampler = np.random.choice(self.num_quasar, (num_trials, self.num_quasar), replace = True)
        masks_resampled = self.masks[resampler].reshape((num_trials, -1))
        waves_resampled = self.waves[resampler].reshape((num_trials,-1))
        fluxes_resampled = self.fluxes[resampler].reshape((num_trials, -1))
        
        upper, lower, bin_size = self.upper, self.lower, self.bin_size
        
        
        ws_boot = np.empty((num_trials, self.num_bins))
        fs_boot = np.empty((num_trials, self.num_bins))
        
        for i in range(num_trials):
            ws_boot[i], fs_boot[i] = self.stack(waves_resampled[i], fluxes_resampled[i], masks_resampled[i], self.wave_grid)
        
        self.wave_covar = np.cov(ws_boot, rowvar = False)
        self.flux_covar = np.cov(fs_boot, rowvar = False)
        
        self.ws_boot = ws_boot
        self.fs_boot = fs_boot
        return ws_boot, fs_boot
    

    def stack(self, waves, fluxes, masks, bins):

        waves_flat = waves[masks].flatten()
        fluxes_flat = fluxes[masks].flatten()
        
        idx = np.searchsorted(bins, waves_flat, 'right')-1

        bad_mask = ((idx == (-1)) | (idx == (self.num_bins)) ) 

        limit = self.num_bins
        idx[bad_mask]=limit
        
        nused = (np.bincount(idx, minlength = limit+1)[:-1])
        ws_total = (np.bincount(idx, minlength = limit+1, weights = waves_flat)[:-1])
        fs_total = (np.bincount(idx, minlength = limit+1, weights = fluxes_flat)[:-1])
        
        wave_stack = (nused>0.0)*ws_total/(nused+(nused==0.0))
        flux_stack = (nused>0.0)*fs_total/(nused+(nused==0.0))
        mask_stack = (nused==0)

        return np.ma.array(wave_stack, mask = mask_stack), np.ma.array(flux_stack, mask = mask_stack)


