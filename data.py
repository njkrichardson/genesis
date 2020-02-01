import pandas as pd
import scipy.io
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import numpy as np 
from math import sqrt
import numpy.random as npr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# ------------------- #
# Datasets 
# 
#
# Images/Vision 
#   * sculpture faces 
#
#
# Text/Language 
#   * news group 
# ------------------- #


def load_sculpture_faces(path):
  """Load Joshua Tenenbaum\'s sculpture faces dataset. Comprised of 698 images 
  of dimension 64 x 64. 

  Data can be accesse here: https://web.archive.org/web/20160913051505/http://isomap.stanford.edu/datasets.html
  
  Parameters
  ----------
  path : str 
      path to dataset

  Returns
  -------

  """
  mat = scipy.io.loadmat(path)
  df = pd.DataFrame(mat['images']).values.T
  num_images, image_dim = df.shape[0], int(sqrt(df.shape[1]))

  sculpture_faces = [] 

  # this feels dumb but can't think of a faster way to do this? 
  for data_case in df: 
    sculpture_faces.append(data_case.reshape(image_dim, image_dim).T)

  return np.array(sculpture_faces).reshape(num_images, image_dim**2)

def load_news(n_samples : int=None): 
    categories = ['alt.atheism', 'talk.religion.misc','comp.graphics', 'sci.space']
    newsgroups = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'),categories=categories) 
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(newsgroups.data).toarray() 
    targets = newsgroups.target
    
    if n_samples is not None: 
        vectors, mask = random_sample(vectors, n_samples, return_mask=True)
        targets = targets[mask]
        return vectors, targets 
    else: 
        return vectors, targets   


# ------------------- #
# Datasets Utils
# ------------------- #   
    
    
    
def trim_axs(axs, N):
    """little plotting helper to massage the axs list to have correct length...
    often nice for plotting some example images from a dataset. 
    
    see https://matplotlib.org/gallery/lines_bars_and_markers/markevery_demo.html#sphx-glr-gallery-lines-bars-and-markers-markevery-demo-py
    """
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]

def plot_im_examples(arr : np.ndarray, n_to_plot=9): 
    n, p = arr.shape
    image_dim = int(sqrt(p))

    cols = 3
    rows = n_to_plot // cols + 1
    fig1, axs = plt.subplots(rows, cols, figsize=(10, 8), constrained_layout=True)
    axs = trim_axs(axs, n_to_plot)
    for i, ax in enumerate(axs):
        rand_idx = npr.randint(n)
        case = arr[rand_idx]
        ax.imshow(case.reshape(image_dim, image_dim), cmap="Greys")
    plt.show() 
    
def random_sample(arr : np.ndarray, n_samples : int, return_mask : bool=False) -> np.ndarray: 
    n, p = arr.shape
    mask = npr.randint(n, size=n_samples)
    if return_mask is True: 
        return arr[mask, :], mask
    else: 
        return arr[mask, :]

def train_val_test(data : np.ndarray, targets : np.ndarray=None, train_prop : float=0.8): 
    """Wrapper around sklearn's function
    """
    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=(1-train_prop))
    return X_train, X_test, y_train, y_test
    
    
