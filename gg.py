import PIL
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from util import *

grid = np.meshgrid(np.arange(7),np.arange(7),indexing='xy')
print(np.stack(grid,axis=-1))
