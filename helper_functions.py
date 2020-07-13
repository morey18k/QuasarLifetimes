from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import math
from _cont_src import fit_sdss_cont
import emcee
import corner
import scipy.interpolate as interpol
from astropy.stats import sigma_clip
import pickle
import lmfit
from multiprocessing import Pool

def pca_model_fn10(wavs, pca1, pca2, pca3, pca4, pca5, pca6, pca7, pca8, pca9, pca10, redoff, interp_function, redshift):
    
    z_test = redshift + redoff
    lams = wavs/(1+z_test)
    parms = np.array([pca1, pca2, pca3, pca4, pca5, pca6, pca7, pca8, pca9, pca10])
    model = np.exp(np.dot(np.append(1.0, parms),interp_function(lams)))
    return model

def pca_model_fn15(wavs, pca1, pca2, pca3, pca4, pca5, pca6, pca7, pca8, pca9, pca10, pca11, pca12, pca13, pca14, pca15, redoff, interp_function, redshift):
    
    z_test = redshift + redoff
    lams = wavs/(1+z_test)
    parms = np.array([pca1, pca2, pca3, pca4, pca5, pca6, pca7, pca8, pca9,
        pca10, pca11, pca12, pca13, pca14, pca15])
    model = np.exp(np.dot(np.append(1.0, parms),interp_function(lams)))
    return model

def maximum_likelihood(redshift, pca_comp_r_int, wave, flux, ivar, coeff_min,
        coeff_max):
    lams = wave/(1+redshift)
    ncomp = pca_comp_r_int(lams[0]).shape[0]-1
    fit_params = lmfit.Parameters()
    for k in range(ncomp):
        fit_params.add('pca'+str(k+1), value=1.0, max = coeff_max[k], min = coeff_min[k])
    fit_params.add('redoff', value = 0.01, max = coeff_max[-1], min = coeff_min[-1])
    
    if ncomp == 10:
        pcm = lmfit.Model(pca_model_fn10, independent_vars = ['wavs'], param_names = ['pca'+str(k+1) for k in range(ncomp)]+['redoff']) 
    if ncomp == 15:
        pcm = lmfit.Model(pca_model_fn15, independent_vars = ['wavs'], param_names = ['pca'+str(k+1) for k in range(ncomp)]+['redoff']) 

    pcm.opts = {'redshift':redshift,'interp_function':pca_comp_r_int}
    
    fit_params = lmfit.Parameters()
    for k in range(ncomp):
        fit_params.add('pca'+str(k+1), value=1.0, max = coeff_max[k], min = coeff_min[k])
    fit_params.add('redoff', value = 0.01, max = coeff_max[-1], min = coeff_min[-1])

    result = pcm.fit(flux, fit_params, wavs = wave, weights = np.sqrt(ivar))

    max_likelihood = np.array(list(result.params.valuesdict().values()))

    return max_likelihood

#MCMC functions imported from Christina's code
def GetInitialPositions(nwalkers, coef_min, coef_max, ndim):
    p0 = np.random.uniform(size = (nwalkers, ndim))
    for i in range(0, ndim):
        p0[:, i] = coef_min[i] + p0[:, i] * (coef_max[i] - coef_min[i])
    return p0

def lnlike_pca(theta, z, pca_comp_r_int, wave, flux, ivar):
    z_test = z + theta[-1]
    lam = wave/(1.+z_test)
    C_dec = np.exp(np.dot(np.append(1.0, theta[:-1]), pca_comp_r_int(lam))) #np.exp(np.dot(theta[:-1],interp_pca_r(lam)))
    chi2 = ivar * np.power(flux-C_dec, 2.0)
    return -np.sum(chi2)

def lnprob_pca(theta, coef_min, coef_max, z, pca_comp_r_int, wave, flux, ivar):
    lp = lnprior_pca(theta, coef_min, coef_max)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_pca(theta, z, pca_comp_r_int, wave, flux, ivar)

def lnprior_pca(theta, coef_min, coef_max):
    if coef_min[0] < theta[0] < coef_max[0] and coef_min[1] < theta[1] < coef_max[1] and coef_min[2] < theta[2] < coef_max[2] and coef_min[3] < theta[3] < coef_max[3] and coef_min[4] < theta[4] < coef_max[4] and coef_min[5] < theta[5] < coef_max[5] and coef_min[6] < theta[6] < coef_max[6] and coef_min[7] < theta[7] < coef_max[7] and coef_min[8] < theta[8] < coef_max[8] and coef_min[9] < theta[9] < coef_max[9] and coef_min[10] < theta[10] < coef_max[10]:
            return 0.0
    return -np.inf


def ModelSpectra(z, wl_blue, pca_coeff_blue, pca_comp_b_int, wl_red, pca_coeff_red, pca_comp_r_int):  
   
    lam_blue = wl_blue/(1.+z)
    model_blue = np.dot(pca_coeff_blue, pca_comp_b_int(lam_blue))

    lam_red = wl_red/(1.+z)
    model_red = np.dot(np.append(1.0, pca_coeff_red), pca_comp_r_int(lam_red))
    
    return model_blue, model_red


def smooth(quant, ivar, pix = 150):
    filtering = np.ones(pix)
    weights = np.correlate(ivar, filtering, mode = 'same')
    return np.correlate(quant*ivar, filtering, mode = 'same')/weights



def feq(float1, float2):
    return (np.abs(float1-float2)<1e-1)

def sigma2fwhm(sigma):
    return sigma * np.sqrt(8 * np.log(2))


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
    print(smoothflux/outivar)
    if (zeroct > 0):
        smoothflux[nzero] = smoothflux[nzero] / outivar[nzero]
    else:
        smoothflux = np.roll(flux, 2 * halfwindow + 1)  # kill off NAN's
    return smoothflux, outivar


def like_computation(covars, x, y, models, all_pixels = True, diag = True, lower = 1190):
    
    if not all_pixels:
        low_pixel = np.argmin(np.abs(x - lower))
    else:
        low_pixel = 0

    x = x[low_pixel:]
    y = y[low_pixel:]    
    
    covars = [c[low_pixel:, low_pixel:] for c in covars]

    if diag:
        covars = [np.diag(np.diag(c)) for c in covars]

    invs = [np.linalg.inv(c) for c in covars]
    like_grid = np.array([log_like_grid(x, y, invs[k], models[k]) for k in range(80)])
    return like_grid

def log_like_grid(x, y, invcov, modler):
    model = modler(x)
    k = invcov.shape[0]
    sign, lndet = np.linalg.slogdet(invcov)
    ynew  = np.array([model - y]).T
    chisq= float(ynew.T@invcov@ynew)
    return -0.5*k*np.log(2*np.pi)-0.5*chisq+0.5*lndet


def rangetup(array, ranger = 100):
    maxer = np.amax(array)
    return (maxer - ranger, maxer+ranger*0.1)
