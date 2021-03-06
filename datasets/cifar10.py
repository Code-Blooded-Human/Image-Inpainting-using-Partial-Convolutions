from tensorflow import keras
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt

def cifar10():
    print('Loading CIFAR-10 dataset')
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    return (x_train, y_train), (x_test, y_test)

def previewDataset(images,labels,numbers=32):
    sample_images = images[:numbers]
    sample_labels = labels[:numbers]

    fig = plt.figure(figsize=(16., 8.))
    grid = ImageGrid(fig, 111, nrows_ncols=(4, 8), axes_pad=0.3)

    for ax, image, label in zip(grid, sample_images, sample_labels):
        ax.imshow(image)
        ax.set_title(label[0])
    plt.show()


def previewGeneratorData(generator):
   nSamples=90
   [maskedImages, masks], originalImages = generator[nSamples]
   previewImage = [None]*(len(maskedImages)+len(masks)+len(originalImages))
   
   previewImage[::3] = originalImages
   previewImage[1::3] = masks
   previewImage[2::3] = maskedImages

   fig = plt.figure(figsize=(17., 8.))
   grid = ImageGrid(fig, 111, nrows_ncols=(4, 9), axes_pad=0.3)

   for ax, image in zip(grid, previewImage):
      ax.imshow(image)

   plt.show()