"""
Contains common functions for the time-to-event verification.

Reference:

Taggart, R. J., and Loveday, N., and Louis, S. (2026).
"On the evaluation of time-to-event, survival time and first passage time forecasts",
In review.
"""

import numpy as np
import xarray as xr
from scipy.special import gammainc
from scipy.special import beta as beta_func
from scipy.special import gamma as gamma_func
from scipy.stats import gamma


def hyp2f1inc(a, tau, b, c, z, tol=0.00000001, max_n = 10000):
    """
    Calculates the incomplete Gauss hypergeometric function
    See Taggart et al (2026) Eq 35.
    
    This implementation is designed for use when z=-1,
    where the series is guaranteed to converge.
    
    Arguments are 'array-like', meaning either xarray data arrays
    or broadcast numpy arrays.
    
    Args:
        a: array-like positive parameter a.
        tau: array-like positive parameter tau.
        b: array-like positive parameter b.
        c: array-like positive parameter c.
        tol: tolerance factor, so that series calculation is terminated
            when |nth term / nth partial sum| < tol
        max_n: maximum number of terms to calculate           
    
    Returns:
        tuple, consisting of:
        - array-like values of the incomplete Gauss hypergeometric function,
            computed to specified tolerance using the trunctated series
        - maximum absolute error |nth term / nth partial sum| in the calculation
            at the point of truncation
    """
    ga = gamma_func(a)
    # the 0th terms etc
    # NB: the gammainc function below, from scipy, is regularized, so not
    # equal to the the lower incomplete gamma function of Taggart et al (2026) Eq 33.
    
    # initialise terms and sums
    last_pochinc = gammainc(a, tau)  # the incomplete pochhammer symbol
    last_term = last_pochinc.copy()  # the term
    last_sum = last_term.copy()  # the sum

    for n in range(max_n):
        this_pochinc = gammainc(a + n + 1, tau) * gamma_func(a + n + 1) / ga
        this_term = this_pochinc * (b + n) * z * last_term / (last_pochinc * (c + n) * (n + 1))
        this_sum = last_sum + this_term
                 
        error = np.nanmax(np.abs(this_term / this_sum))
        if error < tol:
            break

        last_pochinc = this_pochinc.copy()
        last_term = this_term.copy()
        last_sum = this_sum.copy()
        
    return last_sum, n, error


def _twcrps_gamma(alpha, beta, obs, tau):
    """
    Calculates the threshold-weighted CRPS for the gamma distribution with shape parameter `alpha` and
    rate parameter `beta` and weight of one on the interval [0, `tau`]. See Taggart et al (2026) Eq 36.
    The support of the gamma distribution is on [0, infty).
    
    Inputs are array-like. If some are xr.DataArray, then `fcst_alpha` must also be 
    xr.DataArray, arrays must be broadcast prior to input and remaining arguments be float.
    
    Args:
        alpha: shape parameter
        beta: rate parameter
        obs: the observations (i.e. t of Taggart et al (2026) Eq 36)
        tau: the right-censoring time
        
    Returns:
        array-like values of the threshold-weighted CRPS for each forecast case.
        No means are taken.
    """
    hyp_value = hyp2f1inc(2 * alpha + 1, beta * tau, alpha, alpha + 1, -1)
    if hyp_value[2] > 0.000001:
        raise ValueError('hypergeometric function calculation did not converge')
        
    omega = np.minimum(obs, tau)
    
    result = tau * (1 - gamma.cdf(tau, alpha, scale=1/beta)) ** 2
    result += omega * (2 * gamma.cdf(omega, alpha, scale=1/beta) - 1)
    result += 2 * alpha * (gamma.cdf(tau, alpha+1, scale=1/beta) - gamma.cdf(omega, alpha+1, scale=1/beta)) / beta
    result -= 2 * alpha * gamma_func(2*alpha+1) * hyp_value[0] / (beta * gamma_func(alpha+1)**2)
    
    if isinstance(alpha, xr.DataArray):
        result_xr = xr.ones_like(alpha)
        result_xr.values = result
        return result_xr
    return result


def twcrps_gamma(alpha, beta, loc, obs, tau):
    """
    Calculates the threshold-weighted CRPS for the gamma distribution with shape parameter `alpha` and
    rate parameter `beta` and weight of one on the interval [0, `tau`]. See Taggart et al (2026) Eq 36.
    
    Handles the case where the support of the gamma distribution is on [`loc`, infty).
    
    Inputs are array-like. If some are xr.DataArray, then `fcst_alpha` must also be 
    xr.DataArray, arrays must be broadcast prior to input and remaining arguments be float.
    
    Args:
        alpha: shape parameter
        beta: rate parameter
        loc: location parameter
        obs: the observations (i.e. t of Taggart et al (2026) Eq 36)
        tau: the right-censoring time
        
    Returns:
        array-like values of the threshold-weighted CRPS for each forecast case.
        No means are taken.
    """
    result = np.where(
        tau >= loc,
        _twcrps_gamma(alpha, beta, np.maximum(obs - loc, 0), tau - loc),
        np.maximum(tau - obs, 0)
    )
    return result


def _crps_gamma(alpha, beta, obs):
    """
    Calculates the CRPS for the gamma distribution with shape parameter `alpha` and
    rate parameter `beta`.
    
    Inputs are array like. If some are xr.DataArray, then `alpha` must also be 
    xr.DataArray, arrays must be broadcast prior to input and remaining arguments be float.
    
    Formula from Section 3, Scheuerer, Moller 2015, reproduced in Taggart et al (2026) Eq 34.
    """
    result = obs * (2 * gamma.cdf(obs, alpha, scale=1/beta) - 1)
    result -= alpha * (2 * gamma.cdf(obs, alpha + 1, scale=1/beta) - 1) / beta
    result -= alpha * beta_func(alpha + 0.5, 0.5) / (beta * np.pi)
    
    if isinstance(alpha, xr.DataArray):
        result_xr = xr.ones_like(alpha)
        result_xr.values = result
        return result_xr
    return result


def crps_gamma(alpha, beta, loc, obs):
    """
    Calculates the CRPS for the gamma distribution with shape parameter `alpha`,
    rate parameter `beta` and location parameter `loc`.
    
    See formula from Section 3, Scheuerer, Moller 2015, reproduced in Taggart et al (2026) Eq 34
    for the case when loc=0.
    """
    result1 = _crps_gamma(alpha, beta, obs - loc)
    result2 = loc - obs + _crps_gamma(alpha, beta, obs * 0)
    result = np.where(obs >= loc, result1, result2)
    
    if isinstance(alpha, xr.DataArray):
        result_xr = xr.ones_like(alpha)
        result_xr.values = result
        return result_xr    
    
    return result


def survival_crps_gamma(alpha, beta, loc, obs, tau):
    """
    The right-censored survival-CRPS of Avani et al (2020) for the gamma distribution.
    
    Args:
        alpha: shape parameter
        beta: rate parameter
        loc: location parameter
        obs: observations (may or may not be tau-right-censored)
        tau: right-censoring time            
    """
    score = np.where(
        obs < tau,
        crps_gamma(alpha, beta, loc, obs),
        twcrps_gamma(alpha, beta, loc, obs, tau)
    )
    return score


def log_score_gamma(alpha, beta, loc, obs):
    """
    Calculates the log score for a gamma distribution with shape parameter alpha,
    rate parameter beta and location parameter loc.
    """
    result = -np.log(gamma.pdf(obs, alpha, loc=loc, scale=1/beta))
    
    if isinstance(alpha, xr.DataArray):
        result_xr = xr.ones_like(alpha)
        result_xr.values = result
        return result_xr    
    
    return result


def linear_score_gamma(alpha, beta, loc, obs):
    """
    Calculates the linear score for a gamma distribution with shape parameter alpha,
    rate parameter beta and location parameter loc:
        S(F,obs) = -F'(obs)
    """
    result = -gamma.pdf(obs, alpha, loc=loc, scale=1/beta)
    
    if isinstance(alpha, xr.DataArray):
        result_xr = xr.ones_like(alpha)
        result_xr.values = result
        return result_xr    
    
    return result 


def c_index(fcst, obs, tau):
    """
    The c-index measures the ability of the forecast to discriminate the observed realizations.
    This formula assumes that ideally, the higher the forecast value, the lower the
    observed time-to-event. Thus the forecast might be the probability that the event
    occurs before a fixed time s.
    
    If the fcst is (say) the expected time-to-event, then replace `fcst` with `-fcst`
    in the argument. The c-index is only sensitive to correct ranking of forecast values
    compared with observations, not on the values of forecast values.
    
    See first formula of Section 3, Blanche et al (2018).
    
    Args:
        fcst: typically probability of an event not exceeding time s, for fixed s.
            A data array of one dimension, with dimension name 'case',
            or 1-dimension numpy array
        obs: time of event. A data array of one dimension, with dimension name 'case',
            or 1-dimension numpy array
        tau (float): right-censor time
        
    Returns:
        float with the c-index and 'case' dimension collapsed.
    """
    if isinstance(fcst, np.ndarray):
        fcst = xr.DataArray(data=fcst, dims=['case'], coords={'case': range(len(fcst))})
        
    if isinstance(obs, np.ndarray):
        obs = xr.DataArray(data=obs, dims=['case'], coords={'case': range(len(obs))})
        
    t1 = obs.rename({'case': 'case1'})
    t2 = obs.rename({'case': 'case2'})
    rz1 = fcst.rename({'case': 'case1'})
    rz2 = fcst.rename({'case': 'case2'})

    condition = (t1 < t2) & (t1 <= tau)

    first_term_numerator = (condition & (rz1 > rz2)).sum()
    second_term_numerator = 0.5 * (condition & (rz1 == rz2)).sum()

    result = (first_term_numerator + second_term_numerator) / condition.sum()
    return float(result.values)


def auc_s(fcst, obs, s):
    """
    The AUC_s measures the ability of the forecast to discriminate the observed realizations.
    Forecasts are P(event occurs no later than time s)
    and observations of the time-to-event realizations.
    
    If events are right-censored, then it is assumed that s is no greater than the
    censor time.
    
    See second formula of Section 3, Blanche et al (2018).
    
    Args:
        fcst: typically probability of an event not exceeding time s, for fixed s.
            A data array of one dimension, with dimension name 'case',
            or 1-dimension numpy array
        obs: time of event. A data array of one dimension, with dimension name 'case',
            or 1-dimension numpy array
        t (float): right-censor time
        
    Returns:
        float with the c-index and 'case' dimension collapsed.
    """
    
    if isinstance(fcst, np.ndarray):
        fcst = xr.DataArray(data=fcst, dims=['case'], coords={'case': range(len(fcst))})
        
    if isinstance(obs, np.ndarray):
        obs = xr.DataArray(data=obs, dims=['case'], coords={'case': range(len(obs))})
        
    s1 = obs.rename({'case': 'case1'})
    s2 = obs.rename({'case': 'case2'})
    rz1 = fcst.rename({'case': 'case1'})
    rz2 = fcst.rename({'case': 'case2'})
    
    condition = (s1 <= s) & (s2 > s)
    
    first_term_numerator = (condition & (rz1 > rz2)).sum()
    second_term_numerator = 0.5 * (condition & (rz1 == rz2)).sum()

    result = (first_term_numerator + second_term_numerator) / condition.sum()
    return float(result.values)
