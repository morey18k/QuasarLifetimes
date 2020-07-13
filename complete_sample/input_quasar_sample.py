#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 16:07:26 2020

@author: eilers
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import astropy.units as u
import astropy.constants as const
from scipy.stats import binned_statistic_2d
from astropy.table import Table
from astropy.io import fits
from astropy.convolution import convolve, Box1DKernel, Gaussian1DKernel
from astropy.coordinates import SkyCoord
import astropy.cosmology as cosmo
planck = cosmo.Planck13

from functions_quasars import quasar

# -------------------------------------------------------------------------------
# plotting settings
# -------------------------------------------------------------------------------

matplotlib.rcParams['ytick.labelsize'] = 18
matplotlib.rcParams['xtick.labelsize'] = 18
matplotlib.rc('text', usetex=True)
fsize = 18

np.random.seed(42)

# -------------------------------------------------------------------------------
# quasar sample
# -------------------------------------------------------------------------------

quasars = {}

    
quasars['PSOJ011+09'] = quasar(name='PSOJ011+09', M1450=-26.85, z=6.4693, dv=100, \
                       xshooter_path_nir = '/Users/eilers/Dropbox/XShooter/complete_sample/spec1d_coadd_PSOJ011+09_tellcorr_nir.fits', \
                       xshooter_path_vis = '/Users/eilers/Dropbox/XShooter/complete_sample/spec1d_stack_PSOJ011+09_tellcorr.fits')

quasars['PSOJ036+03'] = quasar(name='PSOJ036+03', M1450=-26.28, z=6.5412, dv=100, \
                       xshooter_path_nir = '/Users/eilers/Dropbox/XShooter/complete_sample/spec1d_coadd_J0226+0302_tellcorr_nir_01.fits', \
                       xshooter_path_vis = '/Users/eilers/Dropbox/XShooter/complete_sample/J0226+0302_XSHOOTER_VIS.fits' )

quasars['PSOJ065-26'] = quasar(name='PSOJ065-26', M1450=-27.25, z=6.1877, dv=100, \
                       xshooter_path_nir = '/Users/eilers/Dropbox/XShooter/complete_sample/spec1d_coadd_J0421-2657_tellcorr_nir_02.fits', \
                       xshooter_path_vis = '/Users/eilers/Dropbox/XShooter/complete_sample/J065_coadd.fits' )

quasars['J0842+1218'] = quasar(name='J0842+1218', M1450=-26.91, z=6.0763, dv=100, \
                       xshooter_path_nir = '/Users/eilers/Dropbox/XShooter/complete_sample/spec1d_coadd_J0842+1218_tellcorr_nir_final_02.fits', \
                       xshooter_path_vis = '/Users/eilers/Dropbox/XShooter/complete_sample/J0842_coadd.fits')

quasars['PSOJ158-14'] = quasar(name='PSOJ158-14', M1450=-27.41, z=6.0681, dv=100, \
                       xshooter_path_nir = '/Users/eilers/Dropbox/XShooter/complete_sample/spec1d_coadd_J158-14_tellcorr_nir.fits', \
                       xshooter_path_vis = '/Users/eilers/Dropbox/XShooter/complete_sample/spec1d_stack_PSOJ158-14_tellcorr.fits')

quasars['PSOJ159-02'] = quasar(name='PSOJ159-02', M1450=-26.59, z=6.3809, dv=100, \
                       xshooter_path_nir = '/Users/eilers/Dropbox/XShooter/complete_sample/spec1d_coadd_J1036-0232_tellcorr_nir_final_02.fits', \
                       xshooter_path_vis = '/Users/eilers/Dropbox/XShooter/complete_sample/J1036-0232_XSHOOTER_VIS.fits')

quasars['SDSSJ1143+3803'] = quasar(name='SDSSJ1143+3803', M1450=-26.69, z=5.8367, dv=100, \
                       deimos_path_1 = '/Users/eilers/Dropbox/XShooter/complete_sample/J1143p3807_det3.fits', \
                       deimos_path_2 = '/Users/eilers/Dropbox/XShooter/complete_sample/J1143p3807_det7.fits')

quasars['J1306+0356'] = quasar(name='J1306+0356', M1450=-26.81, z=6.0337, dv=100, \
                       xshooter_path_nir = '/Users/eilers/Dropbox/XShooter/complete_sample/spec1d_coadd_J1306+0356_tellcorr_nir_01.fits', \
                       xshooter_path_vis = '/Users/eilers/Dropbox/XShooter/complete_sample/J1306_coadd.fits')

quasars['J1319+0950'] = quasar(name='J1319+0950', M1450=-26.88, z=6.1330, dv=100, \
                       xshooter_path_nir = '/Users/eilers/Dropbox/XShooter/complete_sample/spec1d_coadd_J1319+0950_tellcorr_nir_05.fits', \
                       xshooter_path_vis = '/Users/eilers/Dropbox/XShooter/complete_sample/J1319_coadd.fits')

quasars['SDSSJ1335+3533'] = quasar(name='SDSSJ1335+3533', M1450=-26.67, z=5.9012, dv=100, \
                       deimos_path_1 = '/Users/eilers/Dropbox/XShooter/complete_sample/J1335p3533_det2.fits', \
                       deimos_path_2 = '/Users/eilers/Dropbox/XShooter/complete_sample/J1335p3533_det6.fits')

quasars['PSOJ217-16'] = quasar(name='PSOJ217-16', M1450=-26.93, z=6.1498, dv=100, \
                       fire_path = '/Users/eilers/Dropbox/XShooter/complete_sample/P217-16_F.fits')

quasars['J1509-1749'] = quasar(name='J1509-1749', M1450=-27.14, z=6.1225, dv=100, \
                       xshooter_path_nir = '/Users/eilers/Dropbox/XShooter/complete_sample/spec1d_coadd_J1509-1749_tellcorr_nir_01.fits', \
                       xshooter_path_vis = '/Users/eilers/Dropbox/XShooter/complete_sample/J1509_coadd.fits')

quasars['PSOJ231-20'] = quasar(name='PSOJ231-20', M1450=-27.20, z=6.5864, dv=100, \
                       xshooter_path_nir = '/Users/eilers/Dropbox/XShooter/complete_sample/spec1d_coadd_J1526-2050_tellcorr_nir_01.fits', \
                       xshooter_path_vis = '/Users/eilers/Dropbox/XShooter/complete_sample/J1526-2049_XShooter_VIS_tellcorr.fits')

quasars['PSOJ323+12'] = quasar(name='PSOJ323+12', M1450=-27.06, z=6.5872, dv=100, \
                       xshooter_path_nir = '/Users/eilers/Dropbox/XShooter/complete_sample/spec1d_coadd_J2132+1217_tellcorr_nir_02.fits', \
                       deimos_path_2 = '/Users/eilers/Dropbox/XShooter/complete_sample/J2132+1217_DEIMOS_830G_8100_tellcorr.fits')

quasars['PSOJ359-06'] = quasar(name='PSOJ359-06', M1450=-26.79, z=6.1719, dv=100, \
                       xshooter_path_nir = '/Users/eilers/Dropbox/XShooter/complete_sample/spec1d_coadd_PSOJ359-06_tellcorr_nir.fits', \
                       xshooter_path_vis = '/Users/eilers/Dropbox/XShooter/complete_sample/spec1d_stack_PSOJ359-06_tellcorr.fits')


# -------------------------------------------------------------------------------
# analysis
# -------------------------------------------------------------------------------

for q in quasars.keys():
    quasars[q].read_data()
    
    
    
    
    
