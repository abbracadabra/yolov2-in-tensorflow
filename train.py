import numpy as np
import tensorflow as tf
import PIL.Image as Image
import builder as yolo2builder
import xml.etree.ElementTree as ET
import os

class Trainer:
    def __init__(self,labels,anchors,**args):
        self.labels=labels
        self.anchors=anchors
        self.boxnum=len(anchors)//2
        self.classnum = len(labels)
        for n in args:
            self.__setattr__(n,args[n])
        self.transfer_n_build()
        
    def transfer_n_build(self):
        self.graph = tf.Graph()
        self.builder = yolo2builder.GraphBuild(self.graph,r"D:/githubrepo1/darknet/darknet19_448.weights",
                                          classnum = self.classnum,boxnum = self.boxnum,anchors = self.anchors,isTransferLearning=True)
        self.builder.build()
        
    def run(self,epochs,data_generator):
        input_holder = self.builder.input
        truth_holder = self.builder.truth_holder
        output = self.builder.output
        label_holder = self.builder.label_placeholder
        error = self.builder.error
        optimizer = self.builder.optimizer
        with self.graph.as_graph_def():
            with tf.Session() as sess:
                for ep in range(epochs):
                    for i,(imgs,lbs,truths) in data_generator():
                        _,loss_val = sess.run([optimizer,error],feed_dict={input_holder:imgs,label_holder:lbs,truth_holder:truths})
                        print("epoch:{} batch:{} loss:{}".format(ep,i,loss_val))
                    
            
    def start(self):
        self.run(10,self.flowdir)
    
    def flowdir(self):
        imagelist = os.listdir(self.imagedir)
        imagelist = np.random.permutation(imagelist)
        labellist = [os.path.join(self.labeldir,fn.split(".")[0]+'.xml') for fn in imagelist]
        batchsize = 20
        pos=0
        batch=0
        while pos<len(imagelist):
            subimagelist = imagelist[pos:pos+batchsize]
            sublabellist = labellist[pos:pos+batchsize]
            if(batch%10==0):
                dim = np.random.randint(10,19)*32
            pos+=batchsize
            batch+=1
            res = self.load_data(subimagelist,sublabellist,dim)
            if len(res[0])==0:
                continue
            yield res
        
    def load_data(self,impaths,lbpaths,dim=608):
        imlist = []
        lblist = []
        truthlist = []
        for impath,lbpath in zip(impaths,lbpaths):
            #original code apply hue,flip,width&height stretching for image augmentation,which i did not implement here
            #preprocess images
            nim = Image.new('RGB', (dim, dim), (127, 127, 127))
            im = Image.open(impath)
            tree=ET.parse(lbpath)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            if w/dim > h/dim:
                nw = dim
                ratio = dim/w
                nh = int(h * ratio)
                im = im.resize((nw,nh))
            else:
                nh = dim
                ratio = dim/h
                nw = int(w * ratio)
                im = im.resize((nw,nh))
            gap_w = dim-nw
            gap_h = dim-nh
            off_w = np.random.randint(0,gap_w)
            off_h = np.random.randint(0,gap_h)
            nim.paste(im, (off_w,off_h))

            #generate labels
            lb = np.zeros((dim//32,dim//32,self.boxnum*(5+self.classnum)))
            truth = np.zeros((20,4))
            for i,obj in enumerate(root.iter('object')):
                if i >=20:
                    #allow at most 20 truth for an image
                    break;
                cls = obj.find('name').text
                cls_ix = self.labels.index(cls)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text)*ratio+off_w, float(xmlbox.find('xmax').text)*ratio+off_w, 
                     float(xmlbox.find('ymin').text)*ratio+off_h, float(xmlbox.find('ymax').text)*ratio+off_h)

                truth[i,0] = b[0]//32
                truth[i, 1] = b[2] // 32
                truth[i, 2] = b[1] // 32
                truth[i, 2] = b[3] // 32

                x_off = ((b[0] + b[1]) / 2) // 32
                y_off = ((b[2] + b[3]) / 2) // 32
                anchor_reshape = np.reshape(self.anchors * 32,(self.boxnum,2))
                minwh = tf.minimum(anchor_reshape,[b[1]-b[0],b[3]-b[2]])
                sect = minwh[:,0]*minwh[:,1]
                union = anchor_reshape[:,0]*anchor_reshape[:,1] + (b[1]-b[0])*(b[3]-b[2]) - sect
                iou = sect/union
                anchor_ix = np.argmax(iou)
                x = ((b[0] + b[1])/2) / 32 - x_off
                y = ((b[2] + b[3])/2) / 32 - y_off
                lb[y_off, x_off, anchor_ix*(5+self.classnum) ] = x
                lb[y_off, x_off, anchor_ix*(5+self.classnum) + 1] = y
                lb[y_off, x_off, anchor_ix*(5+self.classnum) + 2] = np.log(np.clip((b[1]-b[0])/anchor_reshape[anchor_ix,0],1e-5))
                lb[y_off, x_off, anchor_ix*(5+self.classnum) + 3] = np.log((b[3]-b[2])/anchor_reshape[anchor_ix][1])
                lb[y_off, x_off, anchor_ix*(5+self.classnum) + 4] = 1
                lb[y_off, x_off, anchor_ix*(5+self.classnum) + 5 + cls_ix] = 1
            imlist.append(np.array(nim)/255)
            lblist.append(lb)
            truthlist.append(truth)
        return np.array(imlist),np.array(lblist),np.array(truthlist)
                
    def convert(self,size,box):
        dw = 1./size[0]
        dh = 1./size[1]
        x = (box[0] + box[1])/2.0
        y = (box[2] + box[3])/2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        return (x,y,w,h)

        
        
if __name__ == '__main__':
        Trainer(labels=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog',
                        'horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor'],
                anchors=[1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071],
               imagedir=r'D:\Users\yl_gong\Desktop\dl\voc\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages',
               labeldir=r'D:\Users\yl_gong\Desktop\dl\voc\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\Annotations').start();
        
        
        
        

        
    
        















