import tensorflow as tf
from config import *
import numpy as np

detector_inp = tf.placeholder(dtype=tf.float32,shape=[None,None,None,512],name='input')#[None,7,7,512]
ts = tf.shape(detector_inp)[:3]#[None,7,7]
th = tf.cast(tf.shape(detector_inp)[1],tf.float32)
xy_t = tf.placeholder(dtype=tf.float32,shape=[None,None,None,5,2])
wh_t = tf.placeholder(dtype=tf.float32,shape=[None,None,None,5,2])
mask_box = tf.placeholder(dtype=tf.float32,shape=[None,None,None,5,1])
box_num = tf.maximum(tf.reduce_sum(mask_box),1.)
cls_t = tf.placeholder(dtype=tf.float32,shape=[None,None,None,20])
coordxy_t = (xy_t+tf.expand_dims(tf.stack(tf.meshgrid(tf.range(th),tf.range(th)),axis=-1),axis=2))/th#[None,7,7,5,2]
lf_t = coordxy_t-wh_t/2#[None,7,7,5,2]
rt_t = coordxy_t+wh_t/2#[None,7,7,5,2]

detector_out = tf.layers.conv2d(detector_inp,45,(1,1))#[None,7,7,45]
xy = tf.sigmoid(tf.reshape(detector_out[...,0:10],tf.concat([ts,[5,2]],axis=0)))#[None,7,7,5,2]
wh = tf.exp(tf.reshape(detector_out[...,10:20],tf.concat([ts,[5,2]],axis=0))/50)*np.array(anchors)#[None,7,7,5,2]
iou_p = tf.sigmoid(tf.reshape(detector_out[...,20:25],tf.concat([ts,[5]],axis=0)))#[None,7,7,5]
cls = tf.nn.softmax(detector_out[...,25:])#[None,7,7,20]

xyerr = tf.reduce_sum((xy-xy_t)**2 * mask_box)/box_num
wherr = tf.reduce_sum((wh-wh_t)**2 * mask_box)/box_num
clserr = tf.reduce_sum((cls-cls_t)**2 * cls_t)/box_num

coordxy = (xy+tf.expand_dims(tf.stack(tf.meshgrid(tf.range(th),tf.range(th)),axis=-1),axis=2))/th#[None,7,7,5,2]
lf = coordxy-wh/2#[None,7,7,5,2]
rt = coordxy+wh/2#[None,7,7,5,2]
sectwh = tf.minimum(rt_t,rt)-tf.maximum(lf_t,lf)#[None,7,7,5,2]
sect = tf.multiply(*tf.unstack(sectwh,axis=-1))\
       *tf.cast(tf.greater_equal(sectwh[...,0],0),tf.float32)#[None,7,7,5]
union = tf.maximum(tf.multiply(*tf.unstack(rt-lf,axis=-1))+tf.multiply(*tf.unstack(rt_t-lf_t,axis=-1))-sect,1e-5)#[None,7,7,5]
iou_t = sect/union#[None,7,7,5]
mask_box_s = tf.squeeze(mask_box)#[None,7,7,5]
iouerr = tf.reduce_mean((iou_p-iou_t)**2 * tf.where(tf.equal(mask_box_s,1.),mask_box_s,tf.ones_like(mask_box_s)*0.5))

allerr = xyerr+wherr+clserr+iouerr

tf.summary.scalar('xyerr',xyerr)
tf.summary.scalar('wherr',wherr)
tf.summary.scalar('iouerr',iouerr)
tf.summary.scalar('clserr',clserr)
tf.summary.scalar('allerr',allerr)
log_all = tf.summary.merge_all()








