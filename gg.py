import PIL
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from util import *

im,nw,nh = preparetest(r'D:\Users\yl_gong\Desktop\dl\voc\VOC2012\JPEGImages\2008_000309.jpg',224)
vgg16 = keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None)
vgg16.summary()
