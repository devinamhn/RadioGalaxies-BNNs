""" Functions for checking convergence of HMC chains using 
    Gelman Rubin diagnostic and calculating the intergrated 
    autocorrelation time.
"""

import torch
from matplotlib import pyplot as plt
import numpy as np
import radiogalaxies_bnns.inference.utils as utils

#Gelman-Rubin diagnostic
def gelman_rubin(samples1, samples2, N_chains):
    halfway_point = int(len(samples1)/2)
    #remove the first part of each chain
    samples1 = samples1[halfway_point:]
    samples2 = samples2[halfway_point:]

    num_samples= len(samples1)
    #Remove first half of the chain and calculate the mean and variance for each param in the chain
    mean_c1 = np.mean(samples1, axis = 0)
    var_c1 = np.var(samples1, axis = 0) *(1/(num_samples-1)) #(num_samples/(num_samples-1))
    
    mean_c2 = np.mean(samples2, axis = 0)
    var_c2= np.var(samples2, axis = 0) * (1/(num_samples-1)) #(num_samples/(num_samples-1)) 

    #calculate the mean of each param from multiple chains
    mean_of_means = np.mean([mean_c1, mean_c2], axis = 0)
    mean_of_var =  np.mean([var_c1, var_c2], axis = 0)

    #between chain variance - variance of mean values from different chains
    #should converge to 0 as N -> inf
    variance_of_means = ((mean_c1 - mean_of_means)**2 + 
                         (mean_c2 - mean_of_means)**2 ) * (num_samples) / (N_chains -1) #B

    #compare between chain variance to within chain variance 
    # B_j = variance_of_means - mean_of_var/num_samples
    # R_hat = np.sqrt(1+ B_j/abs(variance_of_means))

    R_hat = ( ((num_samples-1)/num_samples) * mean_of_var + (1/num_samples)* variance_of_means) / mean_of_var
    return R_hat

# integrated autocorrelation time
# code from emcee:  https://github.com/dfm/emcee/blob/main/src/emcee/autocorr.py
def next_pow_two(n):
    """Returns the next power of two greater than or equal to `n`"""
    i = 1
    while i < n:
        i = i << 1
    return i

def function_1d(x):
    """Estimate the normalized autocorrelation function of a 1-D series

    Args:
        x: The series as a 1-D numpy array.

    Returns:
        array: The autocorrelation function of the time series.

    """
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= acf[0]
    return acf


def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1


def integrated_time(x, c=5, tol=50, quiet=False):
    """Estimate the integrated autocorrelation time of a time series.

    This estimate uses the iterative procedure described on page 16 of
    `Sokal's notes <https://www.semanticscholar.org/paper/Monte-Carlo-Methods-in-Statistical-Mechanics%3A-and-Sokal/0bfe9e3db30605fe2d4d26e1a288a5e2997e7225>`_ to
    determine a reasonable window size.

    Args:
        x: The time series. If multidimensional, set the time axis using the
            ``axis`` keyword argument and the function will be computed for
            every other axis.
        c (Optional[float]): The step size for the window search. (default:
            ``5``)
        tol (Optional[float]): The minimum number of autocorrelation times
            needed to trust the estimate. (default: ``50``)
        quiet (Optional[bool]): This argument controls the behavior when the
            chain is too short. If ``True``, give a warning instead of raising
            an :class:`AutocorrError`. (default: ``False``)

    Returns:
        float or array: An estimate of the integrated autocorrelation time of
            the time series ``x`` computed along the axis ``axis``.

    Raises
        AutocorrError: If the autocorrelation time can't be reliably estimated
            from the chain and ``quiet`` is ``False``. This normally means
            that the chain is too short.

    """
    x = np.atleast_1d(x)
    if len(x.shape) == 1:
        x = x[:, np.newaxis, np.newaxis]
    if len(x.shape) == 2:
        x = x[:, :, np.newaxis]
    if len(x.shape) != 3:
        raise ValueError("invalid dimensions")

    n_t, n_w, n_d = x.shape
    tau_est = np.empty(n_d)
    windows = np.empty(n_d, dtype=int)

    # Loop over parameters
    for d in range(n_d):
        f = np.zeros(n_t)
        for k in range(n_w):
            f += function_1d(x[:, k, d])
        f /= n_w
        taus = 2.0 * np.cumsum(f) - 1.0
        windows[d] = auto_window(taus, c)
        tau_est[d] = taus[windows[d]]

    # Check convergence
    flag = tol * tau_est > n_t

    # Warn or raise in the case of non-convergence
    if np.any(flag):
        msg = (
            "The chain is shorter than {0} times the integrated "
            "autocorrelation time for {1} parameter(s). Use this estimate "
            "with caution and run a longer chain!\n"
        ).format(tol, np.sum(flag))
        msg += "N/{0} = {1:.0f};\ntau: {2}".format(tol, n_t / tol, tau_est)
        # if not quiet:
        #     raise AutocorrError(tau_est, msg)
        # logger.warning(msg)

    return tau_est

def ess(num_samples, tau):
    return num_samples/tau

paths = utils.Path_Handler()._dict()

path = paths['project']/ 'results'/ 'inits'
path_chains = '/share/nas2/dmohan/mcmc/hamilt/results/inits/thin1000' #final chains loc

device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')


params_hmc_c1 = torch.load(path_chains + '/thin_chain0', map_location=torch.device(device))
params_hmc_c2 = torch.load(path_chains + '/thin_chain1', map_location=torch.device(device))
num_samples = len(params_hmc_c1)

#check convergence for last layer
num_params = 168 
samples1 =np.zeros((num_samples, num_params))
samples2 = np.zeros((num_samples, num_params))
i = 0

for j in range(232274, 232442): #(0, 232444)
    for k in range(num_samples):
        samples1[k][i] = params_hmc_c1[k][j]
        samples2[k][i] = params_hmc_c2[k][j]
    i=i+1
print(len(samples1))

N_chains = 2
burn_in = 0
R_hat = gelman_rubin(samples1[burn_in:], samples2[burn_in:], N_chains)
print(np.where(R_hat<1.0))


plt.figure(dpi=200)
plt.scatter(np.arange(0, 168, 1), R_hat, s =1)
plt.ylabel('R_hat')
plt.xlabel('step')
plt.axhline(y=1, color='gray', linestyle='--')
plt.legend()
plt.savefig(path / 'convergence' / 'rhat.png')

plt.figure(dpi=200)
plt.hist(R_hat)
plt.ylabel('Count')
plt.xlabel('R\u0302')
plt.savefig( path / 'convergence' / 'rhat_hist.png')

exit()
for j in range(5): #9
    print('param', j)
    for i in [0, 10, 20, 30, 50]:
        
        tau = integrated_time(samples1[burn_in:, j], c= i, tol = 50)
        print('Chain 1 --- tau for lag', i , ':', tau, ' ESS:', num_samples/tau)

        tau = integrated_time(samples2[burn_in:, j], c= i, tol = 50)
        print('Chain 2 --- tau for lag', i , ':', tau, ' ESS:', num_samples/tau)
        

    print('Gelman Rubin', R_hat[j])
