import numpy as np 

def cov(arr : np.ndarray):
    """Compute the sample covariance matrix for 
    a matrix arr, assumed to be given in (samples x features) 
    form for data fitting applications. 
    
    Parameters
    ----------
    arr : np.ndarray
        matrix in (samples x features) form 
    """
    n, _ = arr.shape
    return arr.T@arr / (n - 1)

def center(arr: np.ndarray): 
    """Center a data matrix, assumedly given in (samples x features)
    form for data fitting applications. 
    
    Parameters
    ----------
    arr : np.ndarray
        matrix in (samples x features) form
    """
    return arr - np.mean(arr, axis=0)