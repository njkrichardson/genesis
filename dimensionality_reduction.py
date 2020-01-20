import numpy as np 
import stats
import seaborn as sns
import matplotlib.pyplot as plt


# ----------- # 
# Utils: 
#   * scree_plot
#   * Hinton diagram 
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

def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()

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
    if np.sum(np.isnan(data).astype(int)) > 0: 
        print("missing data detected. consider using pca_missing_data")
        raise(NotImplemented)

    reconstruction=None
    n, p = data.shape 
    assert 0 < dim < p, "projection dimension must be a positive integer less than the number of data dimensions {}".format(p)
    
    # center the data 
    de_meaned = stats.center(data) 

    # optimize for high dimensionality data
    if p > n: 
        print('High dimensional data detected, optimizing...')

        # X@X.T is (n x n) which we're assuming is actually smaller than (p x p) in this case
        X = de_meaned
        eigvals, X_eigvecs = np.linalg.eig(X @ X.T) 

        # plop the eigenvalues into a diagonal matrix lambda 
        lam = np.diag(np.real(eigvals))

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
    eigvals = np.real(np.array(eigvals[idx]))
    eigvecs = np.real(np.array(eigvecs[:, idx]))
    E = eigvecs[:, :dim]
    
    # compute low dimensional representation.  
    Y = np.array(data @ E)

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

        # print("Reconstruction error: {}".format(round(np.sum(np.linalg.norm(data - reconstruction, axis=1)), 2)))

        # plot 2D projection 
        if labels:
            to_plot = Y
            if dim != 2:  
                to_plot = pca(data, dim= 2, verbose=False)
            plt.figure(figsize=(10, 8))
            plt.title("2D Data Representation")
            sns.scatterplot(x=to_plot[:, 0], y=to_plot[:, 1], hue=class_identity, legend='full', \
                                palette=sns.color_palette('bright', n_colors=len(np.unique(class_identity))))
            plt.show()
        else: 
            to_plot = Y
            if dim != 2:  
                to_plot = pca(data, dim= 2, verbose=False)
            plt.figure(figsize=(10, 8))
            plt.title("2D Data Representation")
            sns.scatterplot(x=to_plot[:, 0], y=to_plot[:, 1])
            plt.show()
            
        # Hinton diagram for eigenvalue matrix (slow af though, only use for small matrices)
        if lam.shape[0] < 20: 
            hinton(lam)
            plt.show()

    if return_reconstruction is True and reconstruction is not None: 
        return Y, reconstruction

    elif return_reconstruction is True: 
        reconstruction = Y @ E.T
        return Y, reconstruction

    else: 
        return Y


def pca_missing_data(X : np.ndarray, projection_dim: int=2, max_iters: int=100, tol : float=.01, verbose : bool=False): 
    assert np.sum(np.isnan(X).astype(int)) > 0, "pca_missing_data should only be called on data with missing values"

    def _fill_nan(X : np.ndarray): 
        col_mean = np.nanmean(X, axis=0)

        #Find indicies that you need to replace
        inds = np.where(np.isnan(X))

        #Place column means in the indices. Align the arrays using take
        X[inds] = np.take(col_mean, inds[1])

        return X
        
    d, n = X.shape 
    m = projection_dim 
    G = np.isnan(X).astype(int)
    Y = np.zeros((m, n))

    # initialize B as the imputed basis
    X_imputed = _fill_nan(X)
    B_full, _, _ = np.linalg.svd(X_imputed) 

    B = B_full[:, :m]
    
    old_msqerror=np.nan
    errs = [] 
    
    for iter in range(max_iters): 
        for i in range(n):  
            comprow= np.nonzero((G[:, i]==0).astype(int))[0] # complete rows
            Mn = B[comprow,:].T@B[comprow,:]
            cn=B[comprow,:].T@X[comprow][:, i]
            Y[:, i]=np.linalg.pinv(Mn)@cn
        for j in range(d): # update B (basis)
            goodn=np.nonzero((G[j, :]==0).astype(int))[0] #  complete columns
            Yg = Y[:, goodn]
            Fd = Yg@Yg.T
            md=Y[:,goodn]@(X[j][goodn]).T
            B[j,:]=(np.linalg.pinv(Fd)@md).T

        reconstruction = _fill_nan((X-(B@Y)))
        msqerror = np.linalg.norm(reconstruction) 
        errs.append(msqerror)

        # convergence testing
        if abs(old_msqerror-msqerror) < tol: 
            if verbose is True: 
                print("converged after {} iterations".format(iter))
            break

        else:
            old_msqerror = msqerror

    if verbose is True: 
        plt.figure(figsize=(18, 8))
        plt.title('reconstruction error versus iteration')
        plt.scatter(np.arange(len(errs)), errs)
        plt.xlabel('iteration')
        plt.ylabel('reconstruction error')
        plt.show()

           
    return B, Y