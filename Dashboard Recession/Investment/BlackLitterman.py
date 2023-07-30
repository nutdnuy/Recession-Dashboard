# load packages
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import datetime
import statsmodels.api as sm
from scipy.optimize import minimize
# not needed, only to prettify the plots.
import matplotlib
from IPython.display import set_matplotlib_formats
from scipy.stats import norm
import pypfopt as pyp
import seaborn as sns
import yfinance as yf
idx = pd.IndexSlice


# ind_return = pd.read_csv("https://raw.githubusercontent.com/nutdnuy/Portfolio_optimization_with_Python/master/data/ind49_m_ew_rets.csv", header=0, index_col=0)/100


def implied_returns(delta, sigma, w):
    """
Obtain the implied expected returns by reverse engineering the weights
Inputs:
delta: Risk Aversion Coefficient (scalar)
sigma: Variance-Covariance Matrix (N x N) as DataFrame
    w: Portfolio weights (N x 1) as Series
Returns an N x 1 vector of Returns as Series
    """
    ir = delta * sigma.dot(w).squeeze() # to get a series from a 1-column dataframe
    ir.name = 'Implied Returns'
    return ir

# Assumes that Omega is proportional to the variance of the prior
def proportional_prior(sigma, tau, p):
    """
    Returns the He-Litterman simplified Omega
    Inputs:
    sigma: N x N Covariance Matrix as DataFrame
    tau: a scalar
    p: a K x N DataFrame linking Q and Assets
    returns a P x P DataFrame, a Matrix representing Prior Uncertainties
    """
    helit_omega = p.dot(tau * sigma).dot(p.T)
    # Make a diag matrix from the diag elements of Omega
    return pd.DataFrame(np.diag(np.diag(helit_omega.values)),index=p.index, columns=p.index)





# inverse matrix
from numpy.linalg import inv



def bl(w_prior, sigma_prior, p, q,
                omega=None,
                delta=2.5, tau=.02):
    """
# Computes the posterior expected returns based on 
# the original black litterman reference model
#
# W.prior must be an N x 1 vector of weights, a Series
# Sigma.prior is an N x N covariance matrix, a DataFrame
# P must be a K x N matrix linking Q and the Assets, a DataFrame
# Q must be an K x 1 vector of views, a Series
# Omega must be a K x K matrix a DataFrame, or None
# if Omega is None, we assume it is
#    proportional to variance of the prior
# delta and tau are scalars
    """
    if omega is None:
        omega = proportional_prior(sigma_prior, tau, p)
    # Force w.prior and Q to be column vectors
    # How many assets do we have?
    N = w_prior.shape[0]
    # And how many views?
    K = q.shape[0]
    # First, reverse-engineer the weights to get pi
    pi = implied_returns(delta, sigma_prior,  w_prior)
    # Adjust (scale) Sigma by the uncertainty scaling factor
    sigma_prior_scaled = tau * sigma_prior  
    # posterior estimate of the mean, use the "Master Formula"
    # we use the versions that do not require
    # Omega to be inverted (see previous section)
    # this is easier to read if we use '@' for matrixmult instead of .dot()
    #     mu_bl = pi + sigma_prior_scaled @ p.T @ inv(p @ sigma_prior_scaled @ p.T + omega) @ (q - p @ pi)
    mu_bl = pi + sigma_prior_scaled.dot(p.T).dot(inv(p.dot(sigma_prior_scaled).dot(p.T) + omega).dot(q - p.dot(pi).values))
    # posterior estimate of uncertainty of mu.bl
#     sigma_bl = sigma_prior + sigma_prior_scaled - sigma_prior_scaled @ p.T @ inv(p @ sigma_prior_scaled @ p.T + omega) @ p @ sigma_prior_scaled
    sigma_bl = sigma_prior + sigma_prior_scaled - sigma_prior_scaled.dot(p.T).dot(inv(p.dot(sigma_prior_scaled).dot(p.T) + omega)).dot(p).dot(sigma_prior_scaled)
    return (mu_bl, sigma_bl)

# for convenience and readability, define the inverse of a dataframe
def inverse(d):
    """
    Invert the dataframe by inverting the underlying matrix
    """
    return pd.DataFrame(inv(d.values), index=d.columns, columns=d.index)

def w_msr(sigma, mu, scale=True):
    """
    Optimal (Tangent/Max Sharpe Ratio) Portfolio weights
    by using the Markowitz Optimization Procedure
    Mu is the vector of Excess expected Returns
    Sigma must be an N x N matrix as a DataFrame and Mu a column vector as a Serie
    """
    w = inverse(sigma).dot(mu)
    if scale:
        w = w/sum(w) # fix: this assumes all w is +ve
    return w


###############
"""
ind_return = pd.read_csv("https://raw.githubusercontent.com/nutdnuy/Portfolio_optimization_with_Python/master/data/ind49_m_ew_rets.csv", header=0, index_col=0)/100

ind_return.index = pd.to_datetime(ind_return.index, format="%Y%m").to_period('M')
df = ind_return [['Agric','Food ','Smoke','Rtail']] 

tickers = ['Agric','Food ','Smoke','Rtail']
s = df.cov()
pi = implied_returns(delta=2.5, sigma=s, w=pd.Series([.30, 0.25, 0.20, 0.25], index=tickers))

mu_exp = pd.Series([.07, .04, .04, .05],index=tickers) 
np.round(w_msr(s, mu_exp)*100, 2)

# Absolute view 1: INTC will return 2%
# Absolute view 2: PFE will return 4%
q = pd.Series({'Agric': 0.07, 'Food ': 0.04, 'Smoke': 0.04, 'Rtail': 0.05})

# The Pick Matrix
# For View 2, it is for PFE
p = pd.DataFrame([
# For View 1, this is for INTC
    {'Agric': 1, 'Food ': 0, 'Smoke': 0, 'Rtail': 0},
# For View 2, it is for PFE
    {'Agric': 0, 'Food ': 1, 'Smoke': 0, 'Rtail': 0},
# For View 3,
    {'Agric': 0, 'Food ': 0, 'Smoke': 1, 'Rtail': 0},
# For View 4,
    {'Agric': 0, 'Food ': 0, 'Smoke': 0, 'Rtail': 1}

    ])

# Find the Black Litterman Expected Returns
bl_mu, bl_sigma = bl(w_prior=pd.Series({'Agric': .30, 'Food ': 0.25, 'Smoke': 0.20, 'Rtail': 0.25}), sigma_prior=s, p=p, q=q)
# Black Litterman Implied Mu
bl_mu


w_msr(bl_sigma, bl_mu)

"""
