import numpy as np 
import stats
import seaborn as sns
import matplotlib.pyplot as plt


# ----------- # 
# Utils: 
#   * scree_plot
#
#
#
# ----------- # 

def scree_plot(arr : np.ndarray, figsize=(18, 8)): 
    """Generates a scree plot for a given data array. 
    
    Parameters
    ----------
    arr : np.ndarray
        matrix in (samples x features) form 
    figsize : tuple, optional
        by default (18, 8)
    """
    eigvals = np.linalg.eigs(stats.cov(arr))
    plt.figure(figsize=figsize)
    plt.title('Eigenvalue versus magnitude (Scree plot)')
    sns.barplot(x=np.arange(len(eigvals)), y=eigvals, color='blue', saturation=.3)
    plt.ylabel('Magnitude') 
    plt.xlabel('Eigenvalue index')
    plt.show()


# ----------- # 
# Linear Methods: 
#   * pca
#
#
#
# ----------- # 

def pca(data : np.ndarray, dim : int = 2, verbose : bool = False, class_identity=None, return_reconstruction=False) -> np.ndarray:
    """Principal components analysis to form a (dim) dimensional approximation 
    of a given dataset. 
    
    Parameters
    ----------
    data : np.ndarray
        matrix in (samples x features) form 
    dim : int, optional
        approximating dimension, must be less than the data dimension, by default 2
    verbose : bool, optional 
        whether to display reconstruction diagnostics and a scree plot 
    Returns
    -------
    np.ndarray
        dim-dimensional representation of the input data 
    """
    reconstruction=None
    n, p = data.shape 
    assert 0 < dim < p, "projection dimension must be a positive integer less than the number of data dimensions {}".format(p)
    
    # center the data 
    de_meaned = stats.center(data) 

    # optimize for high dimensionality data
    if p > n: 
        if verbose is True: print('High dimensional data detected, optimizing...')

        # X@X.T is (n x n) which we're assuming is actually smaller than (p x p) in this case
        X = de_meaned
        eigvals, X_eigvecs = np.linalg.eig(X @ X.T) 

        # plop the eigenvalues into a diagonal matrix lambda 
        lam = np.diag(eigvals) 

        # compute the eigenvecs
        # transformed_E = X @ X_eigvecs
        lam_inv = np.linalg.inv(lam)
        eigvecs = (X.T @ X_eigvecs @ lam_inv)
    
    else: 
        # generate the data covariance matrix 
        sample_covariance = stats.cov(de_meaned) 

        # extract spectrum of the covariance matrix (its set of eigenvalues and eigenvectors)
        eigvals, eigvecs = np.linalg.eig(sample_covariance)
    
    # arrange the eigenvectors so that they are ordered according to the magnitude (large->small) of their corresponding eigenvalue
    idx = eigvals.argsort()[::-1]   
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    E = eigvecs[:, :dim]
    
    # compute low dimensional representation.  
    Y = data @ E

    if verbose is True:
        # scree plot  
        plt.figure(1, figsize=(18, 8))
        plt.title('Eigenvalue versus magnitude (Scree plot)')
        sns.barplot(x=np.arange(len(eigvals)), y=eigvals, color='blue', saturation=.3)
        plt.ylabel('Magnitude') 
        plt.xlabel('Eigenvalue index')
        plt.show()

        # reconstruction 
        reconstruction = Y @ E.T
        assert reconstruction.shape == data.shape

        reconstruction_params = [data, reconstruction]
        labels = ['Original data', 'Reconstruction']


        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(18, 8))
        for i, ax in zip(range(2), axes.flat):
            # extract given params 
            given_params = reconstruction_params[i]
            ax.set_title(labels[i])
            sns.heatmap(given_params, cmap='Blues_r', alpha=0.65, annot=False, cbar=False, xticklabels=False, yticklabels=False, ax=ax)

        fig.tight_layout()
        plt.show()

        print("Reconstruction error: {}".format(round(np.sum(np.linalg.norm(data - reconstruction, axis=1)), 2)))

        # plot 2D projection 
        if labels:
            to_plot = Y
            if dim != 2:  
                to_plot = pca(data, dim= 2, verbose=False)
            plt.figure(figsize=(10, 8))
            plt.title("2D Data Representation")
            sns.scatterplot(x=to_plot[:, 0], y=to_plot[:, 1], hue=class_identity, legend='full', palette=sns.color_palette('bright'))
            plt.show()
        else: 
            to_plot = Y
            if dim != 2:  
                to_plot = pca(data, dim= 2, verbose=False)
            plt.figure(figsize=(10, 8))
            plt.title("2D Data Representation")
            sns.scatterplot(x=to_plot[:, 0], y=to_plot[:, 1])
            plt.show()

    if return_reconstruction is True and reconstruction is not None: 
        return Y, reconstruction

    elif return_reconstruction is True: 
        reconstruction = Y @ E.T
        return Y, reconstruction

    else: 
        return Y
