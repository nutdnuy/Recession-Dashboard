


## Identify factor betas and store as 'factor_betas' df in risk_model dict
def factor_betas(pca, factor_beta_indices, factor_beta_columns):
        return pd.DataFrame(pca.components_.T, factor_beta_indices, factor_beta_columns)
    
## Identify factor returns and store as 'factor_returns ' df in risk_model dict
def factor_returns(pca, returns, factor_return_indices, factor_return_columns):
    return pd.DataFrame(pca.transform(returns), factor_return_indices, factor_return_columns)
    
## create factor covariance matrix and store in risk_model dict
def factor_cov_matrix(factor_returns, ann_factor):
    return np.diag(factor_returns.var(axis=0, ddof=1)*ann_factor)
    
    
## create idiosyncratic( asset-specific ) risk covariance matrix and store in risk_model dict
def idiosyncratic_var_matrix(returns, factor_returns, factor_betas, ann_factor):
    common_returns = pd.DataFrame(np.dot(factor_returns, factor_betas.T), returns.index, returns.columns)
    residuals = (returns - common_returns)
    return pd.DataFrame(np.diag(np.var(residuals))*ann_factor, returns.columns, returns.columns)


## create idiosyncratic(specific) risk covariance vector and store in risk_model dict
def idiosyncratic_var_vector(returns, idiosyncratic_var_matrix):
    return pd.DataFrame(np.diag(idiosyncratic_var_matrix), returns.columns)


def alpha_vec (dimension = 5 , returns=returns) :
    from sklearn.decomposition import PCA
    
    
    ## instantiate PCA -- reduce dimensionality to n
    num_factor_exposures = dimension
    pca = PCA(n_components=num_factor_exposures,svd_solver='full')
    pca.fit(returns) 
    
    ann_factor = 252 ## to annualize covariance calculation
    
    
    ## create risk model dict to store all data
    risk_model = {}


    risk_model['factor_betas'] = factor_betas(pca, returns.columns.values, np.arange(num_factor_exposures))
    risk_model['factor_returns'] = factor_returns(pca,returns,returns.index,np.arange(num_factor_exposures))
    risk_model['factor_cov_matrix'] = factor_cov_matrix(risk_model['factor_returns'], ann_factor)
    risk_model['idiosyncratic_var_matrix'] = idiosyncratic_var_matrix(returns, risk_model['factor_returns'], risk_model['factor_betas'], ann_factor)
    risk_model['idiosyncratic_var_vector'] = idiosyncratic_var_vector(returns, risk_model['idiosyncratic_var_matrix'])
    ## create alpha vector for objective function
    p1 = returns.T
    p1['alpha_vector'] = p1.median(axis=1)
    alpha_vector = p1[['alpha_vector']].copy()
    
    return risk_model ,  alpha_vector

    

#risk_model ,  alpha_vector =alpha_vec (dimension = 5 , returns=returns)


import cvxpy as cvx

class OptimalHoldings():    
    
    def _get_obj(self, weights, alpha_vector):
                
        part_one = (alpha_vector.values.flatten() - np.mean(alpha_vector.values.flatten()))
        part_two = np.sum(np.abs(alpha_vector.values.flatten()))
        target_weights = part_one / part_two
        objective = cvx.Minimize(cvx.norm(weights - target_weights, p=2))
        
        return objective
    
    def _get_constraints(self, weights, factor_betas, risk):
    
        x = weights
        B = factor_betas
        r = risk
        
        constraints = [
            r <= self.risk_cap**2,
            (B.T)*x <= self.factor_max,
            (B.T)*x >= self.factor_min,
            sum(x.T) == 0.0,
            sum(cvx.abs(x)) <= 1.0,
            x >= self.weights_min,
            x <= self.weights_max
        ]
        
        return constraints

    def __init__(self, risk_cap=0.05, factor_max=10.0, factor_min=-10.0, weights_max=0.55, weights_min=-0.55):
        self.risk_cap=risk_cap
        self.factor_max=factor_max
        self.factor_min=factor_min
        self.weights_max=weights_max
        self.weights_min=weights_min
        
    def _get_risk(self, weights, factor_betas, alpha_vector_index, factor_cov_matrix, idiosyncratic_var_vector):
        f = factor_betas.loc[alpha_vector_index].values.T * weights
        X = factor_cov_matrix
        S = np.diag(idiosyncratic_var_vector.loc[alpha_vector_index].values.flatten())
        return cvx.quad_form(f, X) + cvx.quad_form(weights, S)
    
    def find(self, alpha_vector, factor_betas, factor_cov_matrix, idiosyncratic_var_vector):
        weights = cvx.Variable(len(alpha_vector))
        risk = self._get_risk(weights, factor_betas, alpha_vector.index, factor_cov_matrix, idiosyncratic_var_vector)
        
        obj = self._get_obj(weights, alpha_vector)
        constraints = self._get_constraints(weights, factor_betas.loc[alpha_vector.index].values, risk)

        prob = cvx.Problem(obj, constraints)
        prob.solve(max_iters=500)
        optimal_weights = np.asarray(weights.value).flatten()
        return pd.DataFrame(data=optimal_weights, index=alpha_vector.index)
    
    
class Risk_model():
    def __init__(self, 
                 returns=returns
                 dimension = 5 ,
                 risk_cap=0.05,
                 factor_max=10.0,
                 factor_min=-10.0,
                 weights_max=0.55,
                 weights_min=-0.55):
        
        self.returns = returns
        self.dimension = dimension
        self.risk_cap = risk_cap
        self.factor_max = factor_max
        self.factor_min = factor_min
        self.weights_max = weights_max
        self.weights_min = weights_min
        
        self.risk_model ,  self.alpha_vector =alpha_vec (dimension = 5 , returns=returns)
        
        self.optimal_weights = OptimalHoldings(
            weights_max=0.02,
            weights_min=-0.02,
            risk_cap=0.0015,
            factor_max=0.015,
            factor_min=-0.015).find(alpha_vector, risk_model['factor_betas'], risk_model['factor_cov_matrix'], risk_model['idiosyncratic_var_vector'])
