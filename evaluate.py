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

im,nw,nh = preparetest(r'D:\Users\yl_gong\Desktop\dl\voc\VOCtest_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages\000022.jpg',224)
vgg16 = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None)
_inp = vgg16.predict(keras.applications.vgg16.preprocess_input(np.array([im])),batch_size=1)
_xy,_wh,_iou,_cls = sess.run([xy,wh,iou_p,cls],feed_dict={detector_inp:_inp})

grid=np.meshgrid(np.arange(7),np.arange(7),indexing='xy')
_xy = np.reshape(np.clip(_xy,0,1)/7+np.expand_dims(np.stack(grid,axis=-1)/7,axis=2),[-1,2])
_wh = np.reshape(np.clip(_wh,0,1),[-1,2])
_clsprob = np.max(_cls,axis=-1,keepdims=True)#[none,7,7,1]
score = np.reshape(_iou*_clsprob,[-1])#[none,7,7,5]
_cls = np.reshape(np.tile(np.expand_dims(np.argmax(_cls,axis=-1),axis=-1),[1,1,1,5]),[-1])
pack = [z for z in zip(score,_xy,_cls,_wh)]
pack = [z for z in pack if z[0]>0.35]
pack = [pack[i] for i in np.argsort([z[0] for z in pack])[::-1]]
print(pack)
print(len(pack))

def iou(curr, newcurr):
    al = np.clip(curr[1]-curr[3],0,1)
    ar = np.clip(curr[1] + curr[3],0,1)
    bl = np.clip(newcurr[1] - newcurr[3],0,1)
    br = np.clip(newcurr[1] + newcurr[3],0,1)
    sectw = min(ar[0],br[0])-max(al[0],bl[0])
    secth = min(ar[1],br[1])-max(al[1],bl[1])
    sect = sectw*secth*(sectw>=0)
    union = (ar[0]-al[0])*(ar[1]-al[1])+(br[0]-bl[0])*(br[1]-bl[1])-sect
    return sect/union

pos=0
while pos<len(pack):
    curr = pack[pos]
    newpos=pos+1
    while newpos<len(pack):
        newcurr = pack[newpos]
        if curr[2]==newcurr[2]:
            if iou(curr,newcurr)>0.2:
                del pack[newpos]
                newpos-=1
        newpos+=1
    pos+=1

print(pack)
print(len(pack))

im = Image.fromarray(im)
draw  = ImageDraw.Draw(im)
for sc,xy,cs,wh in pack:
    draw.rectangle((tuple(np.clip(xy-wh/2,0,1)*224), tuple(np.clip(xy+wh/2,0,1)*224)),width=1)
    draw.text(tuple(np.clip(xy-wh/2,0,1)*224), str(round(sc,2))+"/"+labels[cs], font=ImageFont.truetype("arial",14),fill="red")
im.show()

# import time
# time.time()
# im.save(os.path.join(r'D:\Users\yl_gong\Desktop\tp',str(round(time.time() * 1000))+'.jpg'))






