import pandas as pd
import numpy as np
import sklearn.mixture as mix
import scipy.stats as scs
import statsmodels.api as sm

%matplotlib inline

import seaborn as sns


def Markov_switching_autoregression (df, k_regimes =2, order=4):
    
    mod_ = sm.tsa.MarkovAutoregression(
        df, 
        k_regimes=k_regimes, 
        order=order, 
        switching_ar=False)
    res_ = mod_.fit()
    
    return  res_ 
#Markov_switching_autoregression (a).summary()

####################### Hidden_Markov_cluster ############################################


def Hidden_Markov_cluster (df,n_components, covariance_type="full",
                          n_init=100,random_state=7):
    
    
    X_train = X_train.to_numpy()
    model = mix.GaussianMixture(n_components=n_components, 
                                covariance_type=covariance_type, 
                                n_init=n_init, 
                                random_state=random_state).fit(X_train_)
    
    
    
    print("Means and vars of each hidden state")
    for i in range(model.n_components):
        print("{0}th hidden state".format(i))
        print("mean = ", model.means_[i])
        print("var = ", np.diag(model.covariances_[i]))
        print()
    
    dfmean = pd.DataFrame(model.means_ ,columns['mean']).T
    dfmean["factor"] = pd.DataFrame(X_train.columns)
    
    
    return model , dfmean


# hidden_states = model.predict(X_test)

   
# hidden_states = model.predict(X_test)
def regime_plot (model, df):
    from matplotlib import cm
    sns.set(font_scale=1.25)
    style_kwds = {'xtick.major.size': 4, 'ytick.major.size': 3, 'legend.frameon': True}
    sns.set_style('white', style_kwds)
    fig, axs = plt.subplots(model.n_components, sharex=True, sharey=True, figsize=(12,9))
    
    colors = cm.rainbow(np.linspace(0, 1, model.n_components))
    for i, (ax, color) in enumerate(zip(axs, colors)):
            # Use fancy indexing to plot data in each state
            mask = hidden_states == i
            ax.plot_date(df.index.values[mask],
                         df["Close"].values[mask],
                         ".-", c=color)
            ax.set_title("{0}th hidden state".format(i), fontsize=16, fontweight='demi')
            # Format the ticks.
            sns.despine(offset=10)
    plt.tight_layout()

####################### TimeSeriesKMeans ############################################


def TimeSeriesKMeans (df,n_clusters ):
    
    from tslearn.clustering import TimeSeriesKMeans, silhouette_score
    
    
    X_train = df.to_numpy()
    model = TimeSeriesKMeans(n_clusters=n_clusters , metric="dtw")
    hidden_states = model.fit_predict(X_train)
    
    df["hidden_states"] = hidden_states
    
    
    return model ,df

#model ,hidden_states = TimeSeriesKMeans (df=X_test,n_clusters=2 )
#hidden_states = model.fit_predict(X_train)



####################### trend_filtering ############################################



def trend_filtering(data,lambda_value):
    '''Runs trend-filtering algorithm to separate regimes.
        data: numpy array of total returns.'''

    n = np.size(data)
    x_ret = data.to_numpy().reshape(n)

    Dfull = np.diag([1]*n) - np.diag([1]*(n-1),1)
    D = Dfull[0:(n-1),]

    beta = cp.Variable(n)
    lambd = cp.Parameter(nonneg=True)

    def tf_obj(x,beta,lambd):
        return cp.norm(x-beta,2)**2 + lambd*cp.norm(cp.matmul(D, beta),1)

    problem = cp.Problem(cp.Minimize(tf_obj(x_ret, beta, lambd)))

    lambd.value = lambda_value
    problem.solve()

    return beta.value
#filterr =  trend_filtering(data= X_train,lambda_value=14)
#df["Fitter"] = filterr
#df["regime"]= np.where(df["Close"]> df["Fitter"], "up","down")




