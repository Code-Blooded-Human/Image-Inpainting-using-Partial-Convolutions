# ----------------------------------------Keras data generator and augmentor------------------------------------------
# Responsible for creating random batches of images and apply mask on these images.
# Reference: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

from tensorflow import keras
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt

from utils.maskImage import maskImage

class DataGenerator(keras.utils.Sequence):
   def __init__(self, X, y, batch_size=32, dim=(32, 32),
      n_channels=3, shuffle=True):
   
      self.batch_size = batch_size
      self.X = X
      self.y = y
      self.dim = dim
      self.n_channels = n_channels
      self.shuffle = shuffle
      self.on_epoch_end()

   def __len__(self):
      # Denotes the number of batches per epoch
      return int(np.floor(len(self.X) / self.batch_size))

   def __getitem__(self, index):
      # Generate one batch of data
      indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
      X_inputs, y_output = self.__data_generation(indexes)
      return X_inputs, y_output

   def on_epoch_end(self):
      # Updates indexes after each epoch
      self.indexes = np.arange(len(self.X))
      if self.shuffle:
         np.random.shuffle(self.indexes)
   
   def __data_generation(self, idxs):
      # Masked_images is a matrix of masked images used as input
      Masked_images = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels)) # Masked image
      # Mask_batch is a matrix of binary masks used as input
      Mask_batch = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels)) # Binary Masks
      # y_batch is a matrix of original images used for computing error from reconstructed image
      y_batch = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels)) # Original image


      ## Iterate through random indexes
      for i, idx in enumerate(idxs):
         image_copy = self.X[idx].copy()

         ## Get mask associated to that image
         masked_image, mask = maskImage(image_copy)

         Masked_images[i,] = masked_image/255
         Mask_batch[i,] = mask/255
         y_batch[i] = self.y[idx]/255

      ## Return mask as well because partial convolution require the same.
      return [Masked_images, Mask_batch], y_batch

