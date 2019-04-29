import tensorflow as tf
from config import *
import numpy as np

detector_inp = tf.placeholder(dtype=tf.float32,shape=[None,None,None,512],name='input')#[None,7,7,512]
ts = tf.shape(detector_inp)[:3]
th = tf.cast(tf.shape(detector_inp)[1],tf.float32)
xy_t = tf.placeholder(dtype=tf.float32,shape=[None,None,None,5,2])
wh_t = tf.placeholder(dtype=tf.float32,shape=[None,None,None,5,2])
mask_box = tf.placeholder(dtype=tf.float32,shape=[None,None,None,5,1])
cls_t = tf.placeholder(dtype=tf.float32,shape=[None,None,None,20])
box_num = tf.reduce_sum(mask_box)
wh_ta = wh_t/np.array(anchors)#[None,7,7,5,2]
logwh_t = tf.log(tf.maximum(wh_ta,1e-2))#[None,7,7,5,2]
#aa = tf.cast(tf.stack(tf.meshgrid(tf.range(th),tf.range(th)),axis=-1),tf.float32)
coordxy_t = (xy_t+tf.expand_dims(tf.stack(tf.meshgrid(tf.range(th),tf.range(th)),axis=-1),axis=2))/th#[None,7,7,5,2]
lf_t = tf.clip_by_value(coordxy_t-wh_t/2,0.,1.)#[None,7,7,5,2]
rt_t = tf.clip_by_value(coordxy_t+wh_t/2,0.,1.)#[None,7,7,5,2]

detector_out = tf.layers.conv2d(detector_inp,45,(1,1))#[None,7,7,45]
xy = tf.nn.sigmoid(tf.reshape(detector_out[...,0:10],tf.concat([ts,[5,2]],axis=0)))#[None,7,7,5,2]
logwh = tf.clip_by_value(tf.reshape(detector_out[...,10:20],tf.concat([ts,[5,2]],axis=0)),-2.,2.)#[None,7,7,5,2]
iou_p = tf.nn.sigmoid(tf.reshape(detector_out[...,20:25],tf.concat([ts,[5]],axis=0)))#[None,7,7,5]
cls = tf.nn.softmax(detector_out[...,25:])#[None,7,7,20]

xyerr = tf.reduce_sum((xy-xy_t)**2 * mask_box)/box_num*5.
wherr = tf.reduce_sum((logwh-logwh_t)**2 * mask_box)/box_num*5.
clserr = tf.reduce_sum((cls-cls_t)**2 * cls_t)/box_num

wh = tf.exp(logwh)*np.array(anchors)#[None,7,7,5,2]
coordxy = (xy+tf.expand_dims(tf.stack(tf.meshgrid(tf.range(th),tf.range(th)),axis=-1),axis=2))/th#[None,7,7,5,2]
lf = tf.clip_by_value(coordxy-wh/2,0.,1.)#[None,7,7,5,2]
rt = tf.clip_by_value(coordxy+wh/2,0.,1.)#[None,7,7,5,2]
sect = tf.nn.relu(tf.multiply(*tf.unstack(tf.minimum(rt_t,rt)-tf.maximum(lf_t,lf),axis=-1)))#[None,7,7,5]
union = tf.maximum(tf.multiply(*tf.unstack(rt-lf,axis=-1))+tf.multiply(*tf.unstack(rt_t-lf_t,axis=-1))-sect,1e-2)#[None,7,7,5]
iou_t = sect/union#[None,7,7,5]
mask_box_s = tf.squeeze(mask_box)#[None,7,7,5]
iouerr = tf.reduce_mean(tf.reduce_sum((iou_p-iou_t)**2 * tf.where(tf.equal(mask_box_s,1.),mask_box_s,tf.ones_like(mask_box_s)*0.5),axis=-1))

allerr = xyerr+wherr+clserr+iouerr

tf.summary.scalar('xyerr',xyerr)
tf.summary.scalar('wherr',wherr)
tf.summary.scalar('iouerr',iouerr)
tf.summary.scalar('clserr',clserr)
tf.summary.scalar('allerr',allerr)
log_all = tf.summary.merge_all()
writer = tf.summary.FileWriter(log_path,graph=tf.get_default_graph())






