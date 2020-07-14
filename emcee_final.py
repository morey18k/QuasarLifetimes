import numpy as np
import matplotlib as plt
import matplotlib as mpl
from tqdm import tqdm, trange
import pickle
from quasar_class import Quasar
from quasar_stack_class import *
import lmfit
from helper_functions import *
import gc
import emcee
import scipy.integrate
import scipy.special


mpl.rcParams['text.latex.preamble'] = [r'\usepackage{mathrsfs}']

plt.rc('font', family='serif')
plt.rc('text', usetex = True)

def log_likelihood(tq, x, y, invcov, modler):
    model = modler(x, tq)
    k = invcov.shape[0]
    sign, lndet = np.linalg.slogdet(invcov)
    ynew  = np.array([model - y]).T
    chisq= float(ynew.T@invcov@ynew)
    return -0.5*k*np.log(2*np.pi)+0.5*lndet-0.5*chisq 


def log_prior(tq):
    if  1 < tq < 8.9:
        return 0.0
    else:
        return -np.inf

def log_probability(tq, x, y, modler, likeinterp):
    lp = log_prior(tq)
    if not np.isfinite(lp):
        return -np.inf
    return lp + likeinterp(tq)


all_models = np.load('all_models.npy')[:,::-1]
waves = np.load("wavelength_models.npy")[::-1]
high_wave = np.amax(waves)
low_wave = np.amin(waves)
waves_fine = np.linspace(1215.67, 1215.9, 1000)

bin_list=  [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
bin_list = [0.5]

load = False
fixed = False
all_pixels = True
diag_only = False

if all_pixels:
    lower = 1177
else:
    lower = 1200

stack = 15
k = 0


flxx = all_models


tqs = np.linspace(1, 8.9, 80)

flux_interpolated = [interpol.interp1d(waves, flux) for flux in flxx]



model = lambda x, tq: interpol.interp2d(waves, tqs, flxx)(x,tq)

i = 10
tq_dense = np.linspace(tqs[i], tqs[i+1], 10)
plt.figure()    
plt.plot(waves_fine, model(waves_fine, tq_dense).T)
plt.legend(["$t_Q={:.3}$".format(t) for t in tq_dense])
plt.xlabel("Wavelength (Angstroms)")
plt.ylabel("Continuum Normalized Flux")
plt.savefig('model/interpolation_model/tqmodel_{0}_interpolation.pdf'.format(stack))

lmodel = lmfit.Model(model, independent_vars = 'x')
for bin_size in bin_list: 
    qstack = pickle.load(open('q_{1}_stack_{0}.pickle'.format(bin_size, stack), 'rb'))


    wave_stack = qstack.wave_stack[~qstack.wave_stack.mask].data
    flux_stack = qstack.flux_stack[~qstack.wave_stack.mask].data
    idx = np.where(wave_stack<high_wave)
    last_in = np.where(wave_stack<high_wave)[0][-1]
    
    ws = wave_stack[:last_in+1]
    fs = flux_stack[:last_in+1]    


    try:
        mod_covars = np.array([np.load('covar_model/covar_matrix_tq{1:.2}_{0}.npy'.format(bin_size, t))[:last_in+1,:last_in+1] for t in tqs])
    except:
        mod_covars = np.load(f'covar_all_matrix_{bin_size:.1f}.npy')[:, :last_in+1, :last_in+1]
    


    lower_all = 1190
    reg = 1177

    l_diag_short = like_computation(mod_covars, ws, fs, flux_interpolated, all_pixels = False, diag = True, lower = lower_all)

    l_diag_full = like_computation(mod_covars, ws, fs, flux_interpolated, all_pixels = True, diag = True, lower = reg)

    l_nondiag_short = like_computation(mod_covars, ws, fs, flux_interpolated, all_pixels = False, diag = False, lower = lower_all)

    l_nondiag_full = like_computation(mod_covars, ws, fs, flux_interpolated, all_pixels = True, diag = False, lower = reg)

    fig, axes = plt.subplots(2, 2)

    axes[0,0].plot(tqs, l_diag_short, '.-')
    axes[0,0].set_ylim(*rangetup(l_diag_short))
    axes[0,0].set_title(f'Diagonal Covariance ({lower_all}-1220)')

    axes[1,0].plot(tqs, l_diag_full, '.-')
    axes[1,0].set_ylim(*rangetup(l_diag_full))
    axes[1,0].set_title(f'Diagonal Covariance ({reg}-1220)')

    axes[0,1].plot(tqs, l_nondiag_short, '.-')
    axes[0,1].set_ylim(*rangetup(l_nondiag_short))
    axes[0,1].set_title(f'Non-Diagonal Covariance ({lower_all}-1220)')

    axes[1,1].plot(tqs, l_nondiag_full, '.-')
    axes[1,1].set_ylim(*rangetup(l_nondiag_full))
    axes[1,1].set_title(f'Non-Diagonal Covariance ({reg}-1220)')
    fig.tight_layout(pad=2.0)
    plt.savefig(f'model/like_variations_{lower_all}_{bin_size}.pdf')


    tryidx = np.argmin(np.abs(tqs - 6.0))
    std = np.sqrt(np.diag(mod_covars[tryidx]))


    mod_covars_diag = [np.diag(np.diag(c)) for c in mod_covars]
    mod_covars_non = mod_covars

    if diag_only:
        mod_covars = [np.diag(np.diag(c)) for c in mod_covars]
    
    invcovs_diag = [np.linalg.inv(c) for c in mod_covars_diag]

    if fixed:
        invcovs = [np.linalg.inv(mod_covars[tryidx]) for cov in mod_covars]
    else:
        invcovs = [np.linalg.inv(cov) for cov in mod_covars]
   


    logdets = np.array([np.linalg.slogdet(inv)[1] for inv in invcovs])
    #invcovs = [np.diag(np.diag(np.linalg.inv(cov))) for cov in mod_covars]
    like_grid = l_nondiag_short 
    
    maxl = tqs[np.argmax(like_grid)]

    tq_interpol = interpol.interp1d(tqs, like_grid, kind = 'cubic')
    tq_fine = np.linspace(1, 8.9, 1000)
    fine_like = tq_interpol(tq_fine) - np.amax(like_grid)

    cumul = scipy.integrate.cumtrapz(np.exp(fine_like), tq_fine, initial = 0)

    cdf = cumul/cumul[-1]
    pdf = np.exp(fine_like)/cumul[-1]

    quantile = interpol.interp1d(cdf, tq_fine)



    plt.figure()
    plt.plot(tqs, like_grid, 'o', label = 'Model Likelihood Evaluation')
    plt.plot(tq_fine, tq_interpol(tq_fine), label = 'Interpolation')
    plt.xlabel('$\\log_{10}(t_Q)$ (years)')
    plt.ylabel("Log Likelihood")
    #plt.yscale('symlog', linthreshy=250, linscaley = 5)
    plt.ylim(*rangetup(like_grid))
    plt.legend()
    plt.grid()
    plt.savefig('model/interpolation_like/{0}_stack_{1}_bin_likelihood_interpolation.pdf'.format(stack, bin_size))
    invtracer = np.array([np.trace(np.linalg.inv(cov)) for cov in mod_covars])
    fig = plt.figure()
    plt.xlabel("$\\log_{10}(t_Q)$ [years]")
    plt.plot(tqs, invtracer, '.-', label = "$\\mathrm{tr}(\\Sigma(t_Q)^{-1})+c_1$")
    if fixed:
        plt.plot(tqs, -like_grid, '.-', label = "$-\\log(\\mathscr{L_{\mathrm{fixed}}})+c_3$")
    else:
        plt.plot(tqs, -like_grid, '.-', label = "$-\\log(\\mathscr{L_{\mathrm{dep}}})+c_2$")
    plt.legend()
    #plt.yscale('log')
    plt.savefig('model/inv_covar_trace_like.pdf')



    res = lmodel.fit(fs, x= ws, weights = 1/std, tq=5.0)
    
    tq_best = res.best_values['tq']
    pos = tq_best + 0.05*np.random.randn(100, 1)
    nwalkers, ndim = pos.shape

    #initializes and runs an MCMC sampler with the log_probability function to fit the model to the data
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args = (ws, fs, model, tq_interpol))
    #if not load:
        #sampler.run_mcmc(pos, 15000, progress = True)
        #tau = sampler.get_autocorr_time()
        #print('autocorr',tau)
        #flattens the data according to burning/thinning from autocorrelation time
        #flat_samples = sampler.get_chain(discard=1000, thin=1, flat=True)
        #np.save('tq_{1}_samples_{0}.npy'.format(stack, bin_size),flat_samples)
    #else:
        #flat_samples = np.load('tq_{1}_samples_{0}.npy'.format(stack, bin_size))
    
    cv = lambda z: (1/2)*(1 + scipy.special.erf(z/np.sqrt(2)))    
    

    plt.figure()
    plt.plot(tq_fine, pdf)
    plt.xlim(quantile(cv(-4)), quantile(cv(4)))
    plt.xlabel("$\\log_{10}(t_Q)$ (years)")
    plt.ylabel("Probability Density")


    tqmed = quantile(0.5)
    one_sig_upper = quantile(cv(1))
    one_sig_lower = quantile(cv(-1))

    two_sig_upper = quantile(cv(2))
    two_sig_lower = quantile(cv(-2))



    plt.axvline(x=one_sig_upper, color = 'red', linestyle = '--')
    plt.axvline(x=one_sig_lower, color = 'red', linestyle = '--')
    plt.axvline(x=two_sig_upper, color = 'blue', linestyle = '--')
    plt.axvline(x=two_sig_lower, color = 'blue', linestyle = '--')
    plt.axvline(x=tqmed, color = 'orange', linestyle = '--')
    plt.savefig("model/hist/{0}_tq_{1}_hist.pdf".format(stack, bin_size))

    u_var = one_sig_upper - tqmed
    l_var = tqmed - one_sig_lower
        
    print("{0} stack, bin size = {1}, tq =".format(stack, bin_size), tqmed, '+', u_var, '-', l_var)
    
    #tqmed = 5.9

    cov_interp = interpol.interp1d(tqs,mod_covars, axis = 0)
    
    #flux_covar = cov_interp(8.7)
    flux_covar = cov_interp(tqmed)
    
    fig = plt.figure()
    ax = plt.axes(None, label = str(k))
    corr = flux_covar/np.sqrt(np.outer(np.diag(flux_covar), np.diag(flux_covar)))
    im = ax.pcolormesh(qstack.wave_grid[:last_in+1], qstack.wave_grid[:last_in+1], corr)
    plt.ylim(qstack.wave_grid[-2], qstack.wave_grid[0])
    cbar = fig.colorbar(im, ax = ax)
    cbar.set_label('Correlation')
    #plt.title("Continuum Normalized Flux Covariance Matrix Across Wavelength Bins")
    plt.xlabel("Wavelength Bin (Angstroms)")
    plt.ylabel("Wavelength Bin (Angstroms)")
    plt.savefig('model/corr_{1}_tq{0:.2f}.pdf'.format(tqmed, bin_size))






    plt.figure()
    std = np.sqrt(np.diagonal(flux_covar))
    mean_mod = model(waves, tqmed) 
    grid_mod = model(ws, tqmed) 

    ax = plt.axes(None, label = str(bin_size))
    plt.plot(waves, flxx[10], label = "$t_Q=10^2$ years", drawstyle = 'steps')
    plt.plot(waves, flxx[30], label = "$t_Q=10^4$ years", drawstyle = 'steps')
    plt.plot(waves, flxx[70], label = "$t_Q=10^{8}$ years", color = 'red', drawstyle = 'steps')
    plt.xlabel("Rest Frame Wavelength [\\AA]")
    plt.ylabel("Stacked Continuum Normalized Flux")
    plt.axvline(x=1215.67, color = 'grey', linestyle = '--')
    plt.legend()
    plt.savefig('model/diff_models.pdf')
    plt.plot(waves, mean_mod, label = f"$t_Q=10^{{{tqmed:.2f}}}$ years", color = 'purple', drawstyle = 'steps')
    plt.plot(qstack.wave_stack, qstack.flux_stack, label = 'Stacked Flux', color = 'black', drawstyle = 'steps')
    ax.fill_between(ws, grid_mod-std, grid_mod+std, alpha = 0.25, label = f"1-$\\sigma$ Range for $\\Sigma_{{\\mathrm{{model}}}}(t_Q=10^{{{tqmed:.2f}}})$", color = 'purple')
    plt.legend()
    plt.ylim(-0.1, 1.1)
    plt.xlim(lower_all, 1222)
    plt.savefig('model/stack_model/{0}_stack_{1}_model.pdf'.format(stack, bin_size))
    
    plt.figure()
    flux_covar = cov_interp(one_sig_upper)
    std = np.sqrt(np.diagonal(flux_covar))
    mean_mod = model(waves, one_sig_upper) 
    grid_mod = model(ws, one_sig_upper) 

    ax = plt.axes(None, label = str(bin_size))
    plt.plot(waves, flxx[10], label = "$t_Q=10^2$ years", drawstyle = 'steps')
    plt.plot(waves, flxx[30], label = "$t_Q=10^4$ years", drawstyle = 'steps')
    plt.plot(waves, flxx[70], label = "$t_Q=10^{8}$ years", color = 'red', drawstyle = 'steps')
    plt.xlabel("Rest Frame Wavelength [\\AA]")
    plt.ylabel("Stacked Continuum Normalized Flux")
    plt.axvline(x=1215.67, color = 'grey', linestyle = '--')
    plt.legend()
    plt.savefig('model/diff_models.pdf')
    plt.plot(waves, mean_mod, label = f"$t_Q=10^{{{one_sig_upper:.2f}}}$ years", color = 'purple', drawstyle = 'steps')
    plt.plot(qstack.wave_stack, qstack.flux_stack, label = 'Stacked Flux', color = 'black', drawstyle = 'steps')
    ax.fill_between(ws, grid_mod-std, grid_mod+std, alpha = 0.25, label = f"1-$\\sigma$ Range for $\\Sigma_{{\\mathrm{{model}}}}(t_Q=10^{{{one_sig_upper:.2f}}})$", color = 'purple')
    plt.legend()
    plt.ylim(-0.1, 1.1)
    plt.xlim(lower_all, 1222)
    plt.savefig('model/stack_model/{0}_stack_upper_{1}_model.pdf'.format(stack, bin_size))
    
    plt.figure()
    flux_covar = cov_interp(one_sig_lower)
    std = np.sqrt(np.diagonal(flux_covar))
    mean_mod = model(waves, one_sig_lower) 
    grid_mod = model(ws, one_sig_lower) 

    ax = plt.axes(None, label = str(bin_size))
    plt.plot(waves, flxx[10], label = "$t_Q=10^2$ years", drawstyle = 'steps')
    plt.plot(waves, flxx[30], label = "$t_Q=10^4$ years", drawstyle = 'steps')
    plt.plot(waves, flxx[70], label = "$t_Q=10^{8}$ years", color = 'red', drawstyle = 'steps')
    plt.xlabel("Rest Frame Wavelength [\\AA]")
    plt.ylabel("Stacked Continuum Normalized Flux")
    plt.axvline(x=1215.67, color = 'grey', linestyle = '--')
    plt.legend()
    plt.savefig('model/diff_models.pdf')
    plt.plot(waves, mean_mod, label = f"$t_Q=10^{{{one_sig_lower:.2f}}}$ years", color = 'purple', drawstyle = 'steps')
    plt.plot(qstack.wave_stack, qstack.flux_stack, label = 'Stacked Flux', color = 'black', drawstyle = 'steps')
    ax.fill_between(ws, grid_mod-std, grid_mod+std, alpha = 0.25, label = f"1-$\\sigma$ Range for $\\Sigma_{{\\mathrm{{model}}}}(t_Q=10^{{{one_sig_lower:.2f}}})$", color = 'purple')
    plt.legend()
    plt.ylim(-0.1, 1.1)
    plt.xlim(lower_all, 1222)
    plt.savefig('model/stack_model/{0}_stack_lower_{1}_model.pdf'.format(stack, bin_size))
    
    plt.figure()

    ax = plt.axes(None, label = str(bin_size))
    #plt.title("Stacked Continuum Normalized Flux Near Ly-$\\alpha$ Transition")
    #plt.plot(ws, std, label = "Noise Vector")
    #plt.plot(qstack.wave_stack, model(qstack.wave_stack, tqmed), linestyle = '--', label = 'Model')
   
    flux2 = model(ws, tqs[10])
    unc2 = np.sqrt(np.diag(mod_covars[10]))
    flux4 = model(ws, tqs[30])
    unc4 = np.sqrt(np.diag(mod_covars[30]))
    flux8 = model(ws, tqs[70])
    unc8 = np.sqrt(np.diag(mod_covars[70]))
    

    plt.plot(waves, flxx[10], label = "$t_Q=10^2$ years", drawstyle = 'steps')
    plt.plot(waves, flxx[30], label = "$t_Q=10^4$ years", drawstyle = 'steps')
    plt.plot(waves, flxx[70], label = "$t_Q=10^{8}$ years", color = 'red', drawstyle = 'steps')
    plt.xlabel("Rest Frame Wavelength [\\AA]")
    plt.ylabel("Stacked Continuum Normalized Flux")
    plt.axvline(x=1215.67, color = 'grey', linestyle = '--')
    plt.plot(waves, mean_mod, label = f"$t_Q=10^{{{tqmed:.2f}}}$ years", color = 'purple', drawstyle = 'steps')
    ax.fill_between(ws, grid_mod-std, grid_mod+std, alpha = 0.25, label = f"1-$\\sigma$ Range for $\\Sigma_{{\\mathrm{{model}}}}(t_Q=10^{{{tqmed:.2f}}})$", color = 'purple')
    ax.fill_between(ws, flux2-unc2, flux2+unc2, alpha = 0.25, label = f"1-$\\sigma$ Range for $\\Sigma_{{\\mathrm{{model}}}}(t_Q=10^{{2}})$", color = 'C0')
    ax.fill_between(ws, flux4-unc4, flux4+unc4, alpha = 0.25, label = f"1-$\\sigma$ Range for $\\Sigma_{{\\mathrm{{model}}}}(t_Q=10^{{4}})$", color = 'C1')
    ax.fill_between(ws, flux8-unc8, flux8+unc8, alpha = 0.25, label = f"1-$\\sigma$ Range for $\\Sigma_{{\\mathrm{{model}}}}(t_Q=10^{{8}})$", color = 'red')
    plt.legend()
    plt.ylim(-0.1, 1.1)
    plt.xlim(lower_all, 1219)
    plt.savefig('model/models_{}_unc.pdf'.format(bin_size))
    plt.close('all')




