import PIL
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from util import *

aa = tf.placeholder(dtype=tf.float32,shape=[1])
vv = tf.Variable(initial_value=aa,trainable=True)
bb = vv**2
cc = tf.clip_by_value(bb,0,1)
dd = cc**2
mi = tf.train.GradientDescentOptimizer(0.001)
gg = mi.compute_gradients(dd)
ops = mi.apply_gradients(gg)


sess = tf.Session()
sess.run(tf.global_variables_initializer(),feed_dict={aa:[22.]})
for i in range(1600):
    _vc,_, hhh = sess.run([vv,ops, gg])
    print(hhh)
