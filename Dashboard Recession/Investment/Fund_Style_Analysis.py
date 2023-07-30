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
from joblib import Parallel, delayed
import statsmodels.formula.api as smf



def tracking_error(r_a, r_b):
    """
    Returns the Tracking Error between the two return series
    """
    return np.sqrt(((r_a - r_b)**2).sum())

def portfolio_tracking_error(weights, ref_r, bb_r):
    """
    returns the tracking error between the reference returns
    and a portfolio of building block returns held with given weights
    """
    return tracking_error(ref_r, (weights*bb_r).sum(axis=1))

def style_analysis(dependent_variable, explanatory_variables,method_= "SLSQP"):
    """
    Returns the optimal weights that minimizes the Tracking error between
    a portfolio of the explanatory variables and the dependent variable
    """
    n = explanatory_variables.shape[1]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    solution = minimize(portfolio_tracking_error, init_guess,
                       args=(dependent_variable, explanatory_variables,), method=method_,
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    weights = pd.Series(solution.x, index=explanatory_variables.columns)
    return weights


def regress(dependent_variable, explanatory_variables, alpha=True, model=""):
    """
    Runs a linear regression to decompose the dependent variable into the explanatory variables
    returns an object of type statsmodel's RegressionResults on which you can call
       .summary() to print a full summary
       .params for the coefficients
       .tvalues and .pvalues for the significance levels
       .rsquared_adj and .rsquared for quality of fit
    """
    if alpha:
        explanatory_variables = explanatory_variables.copy()
        explanatory_variables["Alpha"] = 1
    #lm = smf.quantreg(dependent_variable, explanatory_variables).fit()
    #lm = smf.mixedlm(dependent_variable, explanatory_variables).fit()
    lm = sm.OLS(dependent_variable, explanatory_variables).fit()
    return lm




def style_all_time (Factor_ret, Nav_ret, date_col ="date"):
    full_data = pd.merge(Nav_ret, Factor_ret, left_on= date_col, right_on= date_col, how="left").fillna(0)
    full_data.set_index(date_col, inplace=True)
    
    Factor_retcolumns = Factor_ret.columns.tolist()
    Factor_retcolumns.remove(date_col)
    Factor_ret = full_data [Factor_retcolumns]
    
    nav = full_data ["pct"] 
    weights = style_analysis( nav, Factor_ret, )
    
    print(weights)
    
    return weights

def style_time_pal (i,full_data, n ,Factor_ret, Nav_ret, date_col ="date", method_= "SLSQP"):
    data_ = full_data.iloc[ i:i+n]
    
    Factor_retcolumns = Factor_ret.columns.tolist()
    Factor_retcolumns.remove(date_col)
    Factor_ret = data_[Factor_retcolumns]
    
    nav = data_["pct"]
    weights = style_analysis( nav, Factor_ret,method_ )    
    style_time_dic[data_.index[1]] = weights

    return weights


def style_time_series (Factor_ret, Nav_ret, date_col ="date", method_= "SLSQP", n_period=20) :
    
    full_data= pd.merge(Nav_ret, Factor_ret, 
                        left_on= date_col, right_on= date_col, 
                        how="left").fillna(0)
    n = full_data.count().tolist()[0]
    style_time_dic = {}
    
    start = time.time()
    # n_jobs is the number of parallel jobs
    n = full_data.count().tolist()[0]
    lst_style_time  = Parallel(n_jobs=2)(delayed(style_time_pal)(full_data=full_data, n=30 ,i=i,Factor_ret= MSCIdata, Nav_ret= nav_pct) for i in range(0, n-30, 1))
    end = time.time()
    print("run time ",'{:.4f} s'.format(end-start))
    
    df_style_time  = pd.DataFrame(lst_style_time)
    
    return df_style_time



def port_tracking_error_find_period (period,ref_r, bb_r):
    weights = style_time_series (Factor_ret= bb_r , Nav_ret= ref_r)  
    tr_error = tracking_error(ref_r['pct'], (weights*bb_r).sum(axis=1))
    return  tr_error




def opp_n_style_time_series ():
    
    init_guess = 20
    dependent_variable = nav_pct
    explanatory_variables = MSCIdata
    n = explanatory_variables.shape[1]
    #constraints = (weights_sum_to_1,)
    method_= "SLSQP"
    
    
    
    solution = minimize(port_tracking_error_find_period,
                        init_guess,
                        args=(dependent_variable, explanatory_variables,),
                        method=method_,
                        options={'disp': False},
                        #constraints=(weights_sum_to_1,),
                        bounds=bounds)
    
    weights = pd.Series(solution.x, index=explanatory_variables.columns)
    
    return  weights
    