import pandas as pd
import scipy.io
import numpy as np 
from math import sqrt


def load_sculpture_faces(path):
  """Load Joshua Tenenbaum's sculpture faces dataset. Comprised of 698 images 
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
