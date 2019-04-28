import PIL
import numpy as np
import tensorflow as tf
from tensorflow import keras

vgg16 = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None)

