"""
Contains common functions for the time-to-event verification.

Reference:

Taggart, R. J., and Loveday, N., and Louis, S. (2026).
"Evaluating time-to-event forecasts under fixed-horizon right-censoring",
In review.
"""

import numpy as np
import xarray as xr
from scipy.special import gammainc
from scipy.special import beta as beta_func
from scipy.special import gamma as gamma_func
from scipy.stats import gamma


def _twcrps_gamma_wessel(alpha, beta, obs, tau):
    """
    Calculates the left-censored threshold-weighted CRPS (Equation B10 from Wessel et al (2024))
    with left-censoring threshold `tau` for the gamma distribution with shape `alpha`, rate `beta`
    and support on [0,infty).
    
    Inputs are array-like. If some are xr.DataArray, then `alpha` must also be 
    xr.DataArray, arrays must be broadcast prior to input and remaining arguments be float.
    
    Args:
        alpha: shape parameter
        beta: rate parameter
        obs: the observations
        tau: the left-censoring time
        
    Returns:
        array-like values of the threshold-weighted CRPS for each forecast case.
        No means are taken.
    """
    scale = 1 / beta
    vtauy = np.maximum(tau, obs)
    Ftau = gamma.cdf(tau, alpha, scale=scale)
    Fv = gamma.cdf(vtauy, alpha, scale=scale)
    
    result = -tau * Ftau ** 2 + vtauy * (2 * Fv - 1)
    result += alpha / beta * (
        1 - Ftau ** 2 + 2 * Ftau * gamma.cdf(tau, alpha + 1, scale=scale)
        - 2 * gamma.cdf(vtauy, alpha + 1, scale=scale)
    )
    result -= (1 - gamma.cdf(2 * tau, 2 * alpha, scale=scale)) / (beta * beta_func(1 / 2, alpha))
    
    if isinstance(alpha, xr.DataArray):
        result_xr = xr.ones_like(alpha)
        result_xr.values = result
        return result_xr
    return result


def _twcrps_gamma(alpha, beta, obs, tau):
    """
    Calculates the (right-censored) threshold-weighted CRPS for the gamma distribution
    with shape parameter `alpha` and rate parameter `beta` and threshold weighting of
    one on the interval [0, `tau`] and zero elsewhere.
    The support of the gamma distribution is on [0, infty).
    
    Inputs are array-like. If some are xr.DataArray, then `fcst_alpha` must also be 
    xr.DataArray, arrays must be broadcast prior to input and remaining arguments be float.
    
    Args:
        alpha: shape parameter
        beta: rate parameter
        obs: the observations
        tau: the right-censoring time
        
    Returns:
        array-like values of the threshold-weighted CRPS for each forecast case.
        No means are taken.
    """
    return _crps_gamma(alpha, beta, obs) - _twcrps_gamma_wessel(alpha, beta, obs, tau)


def twcrps_gamma(alpha, beta, loc, obs, tau):
    """
    Calculates the threshold-weighted CRPS for the gamma distribution with shape parameter `alpha` and
    rate parameter `beta` and weight of one on the interval [0, `tau`].
    
    Handles the case where the support of the gamma distribution is on [`loc`, infty).
    
    Inputs are array-like. If some are xr.DataArray, then `fcst_alpha` must also be 
    xr.DataArray, arrays must be broadcast prior to input and remaining arguments be float.
    
    Args:
        alpha: shape parameter
        beta: rate parameter
        loc: location parameter
        obs: the observations
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
    
    Formula from Section 3, Scheuerer & Moller (2015).
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
    
    See formula from Section 3, Scheuerer & Moller (2015).
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


def twlogs_gamma(alpha, beta, loc, obs, tau):
    """
    Calculates the threshold-weighted logarithmic score for a gamma distribution
    shape parameter `alpha`, rate parameter `beta`, location parameter `loc`,
    observations `obs` and censoring time `tau`.
    """
    result = -np.where(obs < tau, 1, 0) * np.log(gamma.pdf(obs, alpha, loc=loc, scale=1/beta))
    result -= np.where(obs >= tau, 1, 0) * np.log(1 - gamma.cdf(tau, alpha, loc=loc, scale=1/beta))
    return result
