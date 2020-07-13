#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 16:09:18 2020

@author: eilers
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 10:12:05 2020

@author: eilers
"""

import os
import numpy as np
import astropy.units as u
from astropy.table import Table, Column
from astropy.io import fits, ascii
from astropy.convolution import convolve, Box1DKernel, Gaussian1DKernel
import astropy.constants as const
import emcee
import corner
import scipy.interpolate as interpol
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import astropy.cosmology as cosmo
from astropy.coordinates import SkyCoord
import scipy.optimize as op
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import pickle
from _cont_src import fit_sdss_cont
from astropy.stats import sigma_clip

planck = cosmo.Planck13


# -------------------------------------------------------------------------------
# plotting settings
# -------------------------------------------------------------------------------

matplotlib.rcParams['ytick.labelsize'] = 18
matplotlib.rcParams['xtick.labelsize'] = 18
matplotlib.rc('text', usetex=True)
fsize = 18

colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple", "pale red", "dark red"]
colors = sns.xkcd_palette(colors)

np.random.seed(42)

# -------------------------------------------------------------------------------
# constants
# -------------------------------------------------------------------------------

mgii = 2798.7
civ_a, civ_b = 1548., 1550.
lya, lyb = 1215.6701, 1025.7223
lyc, lyd, lye, lycont = 972.5368, 949.7, 937.8, 911.7

nu_rest_cii = 1900.548
nu_rest_co65 = 691.473
nu_rest_co54 = 576.267

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# quasar class
# -------------------------------------------------------------------------------  
# -------------------------------------------------------------------------------

class quasar:
       
    def __init__(self, name, M1450, z, dv=None, notes=None, xshooter_path_vis=None, xshooter_path_nir=None, xshooter_path=None, deimos_path_1=None, deimos_path_2=None, fire_path=None):
        
        self.name = name
        self.M1450 = M1450
        self.notes = notes
        self.z = z
        self.z_err = dv / const.c.to(u.km/u.s).value * (1 + self.z)
        self.xshooter_path_vis = xshooter_path_vis
        self.xshooter_path_nir = xshooter_path_nir
        self.xshooter_path = xshooter_path
        self.deimos_path_1 = deimos_path_1
        self.deimos_path_2 = deimos_path_2
        self.fire_path = fire_path
                
    # -------------------------------------------------------------------------------
    # read data
    # -------------------------------------------------------------------------------

    def read_data(self, folder):  

        self.folder = folder
         
        '''
        read all available data files
        '''       
        if self.xshooter_path_vis is not None:
            print('reading in X-Shooter VIS data for {}...'.format(self.name))
            self.xshooter_data_vis = fits.getdata(self.xshooter_path_vis)
        if self.xshooter_path_nir is not None:
            print('reading in X-Shooter NIR data for {}...'.format(self.name))
            self.xshooter_data_nir = fits.getdata(self.xshooter_path_nir)
        if self.xshooter_path is not None:
            print('reading in X-Shooter NIR data for {}...'.format(self.name))
            self.xshooter_data = fits.getdata(self.xshooter_path)
            self.wave = self.xshooter_data['x'] * 10 # u.AA
            self.flux = self.xshooter_data['y']
            self.noise = self.xshooter_data['dy']
            self.ivar = 1. / self.noise**2
        
        # stitch NIR and VIS together
        if self.xshooter_path_nir is not None:
            try: 
                wl_nir = self.xshooter_data_nir['OPT_WAVE']
                flux_nir = self.xshooter_data_nir['OPT_FLAM']
                noise_nir = self.xshooter_data_nir['OPT_FLAM_SIG']
            except:
                wl_nir = self.xshooter_data_nir['WAVE']
                flux_nir = self.xshooter_data_nir['FLUX']
                noise_nir = self.xshooter_data_nir['ERROR']
            try:
                try:
                    try:
                        wl_vis = self.xshooter_data_vis['OPT_WAVE']
                        flux_vis = self.xshooter_data_vis['OPT_FLAM']
                        noise_vis = self.xshooter_data_vis['OPT_FLAM_SIG']
                    except:
                        wl_vis = self.xshooter_data_vis['wave']
                        flux_vis = self.xshooter_data_vis['flux']
                        noise_vis = 1./np.sqrt(self.xshooter_data_vis['ivar'])
                except: 
                    wl_vis = self.xshooter_data_vis['WAVE']
                    flux_vis = self.xshooter_data_vis['FLUX']
                    noise_vis = self.xshooter_data_vis['ERROR']
            except:
                   self.deimos_data = fits.getdata(self.deimos_path_2)
                   wl_vis = self.deimos_data['OPT_WAVE']
                   flux_vis = self.deimos_data['OPT_FLAM']
                   noise_vis = self.deimos_data['OPT_FLAM_SIG']
            ind_vis = wl_vis < 10100
            ind_nir = wl_nir >= 10100
            match_pix= 1000
            mean_flux_vis = np.median(convolve(flux_vis[ind_vis], Box1DKernel(500))[-match_pix:])
            mean_flux_nir = np.median(convolve(flux_nir[ind_nir], Box1DKernel(500))[:match_pix])
            correction = mean_flux_nir/mean_flux_vis

            plt.plot(wl_vis[ind_vis], convolve(flux_vis[ind_vis], Box1DKernel(2)), c='k', lw = .8)
            plt.plot(wl_nir[ind_nir], convolve(flux_nir[ind_nir], Box1DKernel(2)), c='k', lw = .8)            
            plt.plot(wl_vis[ind_vis], convolve(flux_vis[ind_vis], Box1DKernel(500)), c='r')
            plt.plot(wl_nir[ind_nir], convolve(flux_nir[ind_nir], Box1DKernel(500)), c='b')
            plt.plot(wl_nir[ind_nir], convolve(flux_nir[ind_nir], Box1DKernel(500)) * correction, c='g')
            plt.xlim(8000, 12000)
            plt.title('correction factor: {}'.format(correction))
            plt.axvline(wl_vis[ind_vis][-match_pix], color = colors[1], linestyle = '--')
            plt.axvline(wl_nir[ind_nir][match_pix], color = colors[1], linestyle = '--') 
            plt.ylim(-.5, 3)
            plt.savefig('{}/tests/{}_test_correction_mult.pdf'.format(self.folder, self.name))  
            plt.close()
            
            self.wave = np.hstack([wl_vis[ind_vis], wl_nir[ind_nir]])
            self.flux = np.hstack([flux_vis[ind_vis] * correction, flux_nir[ind_nir]])
            self.noise = np.hstack([noise_vis[ind_vis] * correction, noise_nir[ind_nir]]) 
            try:
                try:
                    try:
                        self.ivar = np.hstack([self.xshooter_data_vis['OPT_FLAM_IVAR'][ind_vis] /correction**2, self.xshooter_data_nir['OPT_FLAM_IVAR'][ind_nir]])
                    except:
                        self.ivar = np.hstack([self.xshooter_data_vis['ivar'][ind_vis] / correction**2, self.xshooter_data_nir['OPT_FLAM_IVAR'][ind_nir]])
                except:
                    self.ivar = 1./(np.hstack([self.xshooter_data_vis['ERROR'][ind_vis] * correction, self.xshooter_data_nir['ERROR'][ind_nir]]))**2
            except:
                self.ivar = np.hstack([self.deimos_data['OPT_FLAM_IVAR'][ind_vis] /correction**2, self.xshooter_data_nir['OPT_FLAM_IVAR'][ind_nir]])
        if self.deimos_path_1 is not None:
            print('reading in DEIMOS data for {}...'.format(self.name))
            deimos_data_b = fits.getdata(self.deimos_path_1)            
            deimos_data_r = fits.getdata(self.deimos_path_2)     
            
            wl_b = deimos_data_b['OPT_WAVE']
            flux_b = deimos_data_b['OPT_FLAM']
            noise_b = deimos_data_b['OPT_FLAM_SIG']
            ivar_b = deimos_data_b['OPT_FLAM_IVAR']            
            wl_r = deimos_data_r['OPT_WAVE']
            flux_r = deimos_data_r['OPT_FLAM']
            noise_r = deimos_data_r['OPT_FLAM_SIG']
            ivar_r = deimos_data_r['OPT_FLAM_IVAR']
                        
            self.wave = np.hstack([wl_b, wl_r])
            self.flux = np.hstack([flux_b, flux_r])
            self.noise = np.hstack([noise_b, noise_r])            
            self.ivar = np.hstack([ivar_b, ivar_r])
            
        if self.fire_path is not None:
            print('reading in FIRE data for {}...'.format(self.name))
            hdu = fits.open(self.fire_path) 
            header = hdu[0].header
            self.flux = hdu[0].data
            self.noise = fits.getdata(self.fire_path[:-6]+'E'+self.fire_path[-5:])
            self.ivar = 1./self.noise**2
            self.wave = 10. ** (header['CRVAL1'] + header['CDELT1'] * np.arange(0, header['NAXIS1']))
            
            
            
