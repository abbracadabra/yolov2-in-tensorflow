import numpy as np
from util import *
from config import *
import tensorflow as tf
from tensorflow import keras
from model import *
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess,model_path)

im,nw,nh = preparetest(r'D:\Users\yl_gong\Desktop\dl\voc\VOC2012\JPEGImages\2011_006842.jpg',224)
vgg16 = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None)
_inp = vgg16.predict(keras.applications.vgg16.preprocess_input(np.array([im])),batch_size=1)
_xy,_wh,_iou,_cls = sess.run([xy,wh,iou_p,cls],feed_dict={detector_inp:_inp})

grid=np.meshgrid(np.arange(7),np.arange(7),indexing='ij')
_xy = np.reshape(_xy/7+np.stack((grid[0],grid[1]),axis=-1)*(1/7),[-1,2])
_wh = np.reshape(np.power(_wh,2),[-1,2])
_clsprob = np.max(_cls,axis=-1,keepdims=True)
score = np.reshape(_obj*_clsprob,[-1])
_cls = np.reshape(np.argmax(_cls,axis=-1),[-1])
pack = [z for z in zip(score,_xy,_cls,_wh)]
pack = [z for z in pack if z[0]>0.5 and z[3][0]>0.01 and z[3][1]>0.01]
pack = [pack[i] for i in np.argsort([z[0] for z in pack])[::-1]]
print(pack)
print(len(pack))

def iou(curr, newcurr):
    al = np.clip(curr[1]-curr[3],0,1)
    ar = np.clip(curr[1] + curr[3],0,1)
    bl = np.clip(newcurr[1] - newcurr[3],0,1)
    br = np.clip(newcurr[1] + newcurr[3],0,1)
    sect = (min(ar[0],br[0])-max(al[0],bl[0]))*(min(ar[1],br[1])-max(al[1],bl[1]))
    union = (ar[0]-al[0])*(ar[1]-al[1])+(br[0]-bl[0])*(br[1]-bl[1])-sect
    return sect/union

pos=0
while pos<len(pack):
    curr = pack[pos]
    newpos=pos+1
    while newpos<len(pack):
        newcurr = pack[newpos]
        if curr[2]==newcurr[2]:
            if iou(curr,newcurr)>0.8:
                del pack[newpos]
                newpos-=1
        newpos+=1
    pos+=1

print(pack)
print(len(pack))

im = Image.fromarray(im)
draw  = ImageDraw.Draw(im)
for sc,xy,cs,wh in pack:
    draw.rectangle((tuple((xy-wh/2)*224), tuple((xy+wh/2)*224)),width=1)
    draw.text(tuple((xy-wh/2)*224), str(round(sc,2))+"/"+labels[cs], font=ImageFont.truetype("arial"))
im.show()






