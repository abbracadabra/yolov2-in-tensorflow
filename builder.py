import numpy as np
import tensorflow as tf

class WeightLoad:
    offset=16
    path=None
    graph=None
    def __init__(self,path,graph):
        self.path = path
        self.graph = graph
    def nextConvBlock(self,shape,scope,batch_norm):
        _r = {}
        bias = np.memmap(self.path, mode='r', shape=(shape[3]),offset=self.offset, dtype='float32')
        _r['bias'] = bias
        self.offset += shape[3]*4
        if (batch_norm):
            gamma = np.memmap(self.path, mode='r', shape=(shape[3]),offset=self.offset, dtype='float32')
            self.offset += shape[3]*4
            ma = np.memmap(self.path, mode='r', shape=(shape[3]),offset=self.offset, dtype='float32')
            self.offset += shape[3]*4
            mv = np.memmap(self.path, mode='r', shape=(shape[3]),offset=self.offset, dtype='float32')
            self.offset += shape[3]*4
        filter = np.memmap(self.path, mode='r', shape=(shape[3], shape[2], shape[0], shape[1]),offset=self.offset,dtype='float32')
        filter = filter.transpose([2, 3, 1, 0])
        self.offset += shape[3] * shape[2] * shape[0] * shape[1]*4

        with self.graph.as_default():
            with tf.name_scope(scope):
                _r['bias'] = tf.Variable(initial_value=bias,name='bias')
                _r['filter'] = tf.Variable(initial_value=filter, name='filter')
                if(batch_norm):
                    _r['gamma'] = tf.Variable(initial_value=gamma, name='gamma')
                    _r['ma'] = tf.Variable(initial_value=ma, name='ma')
                    _r['mv'] = tf.Variable(initial_value=mv, name='mv')
        return _r

#loader = WeightLoad(path=r"D:/githubrepo1/darknet/yolov2.weights",graph=graph)
class GraphBuild:
    graph=None
    temp=None
    loader=None
    classes=None
    input=None
    output=None
    def __init__(self,graph,path,classnum=80,boxnum=5,anchors=None,isTransferLearning=False):
        self.graph=graph
        self.classnum=classnum
        self.boxnum=boxnum
        self.loader = WeightLoad(path=path, graph=graph)
        self.anchors = anchors
        self.isTransferLearning=isTransferLearning
    def build(self):
        with self.graph.as_default():
            self.label_placeholder = tf.placeholder(dtype='float32', shape=(None, None,None ,self.boxnum*(5+self.classnum)), name='label')
            self.truth_holder = tf.placeholder(dtype='float32',shape=(None,20,4),name='truths')
            self.temp = self.input = tf.placeholder(dtype='float32', shape=(None, None, None, 3), name='input')
            self.buildConv((3,3,3,32),'0-conv')
            self.buildMax('1-max')
            self.buildConv((3,3,32,64), '2-conv')
            self.buildMax('3-max')
            self.buildConv((3, 3, 64, 128), '4-conv')
            self.buildConv((1, 1, 128, 64), '5-conv')
            self.buildConv((3, 3, 64, 128), '6-conv')
            self.buildMax('7-max')
            self.buildConv((3, 3, 128, 256), '8-conv')
            self.buildConv((1, 1, 256, 128), '9-conv')
            self.buildConv((3, 3, 128, 256), '10-conv')
            self.buildMax('11-max')
            self.buildConv((3, 3, 256, 512), '12-conv')
            self.buildConv((1, 1, 512, 256), '13-conv')
            self.buildConv((3, 3, 256, 512), '14-conv')
            self.buildConv((1, 1, 512, 256), '15-conv')
            branch_1 = self.buildConv((3, 3, 256, 512), '16-conv')
            self.buildMax('17-max')
            self.buildConv((3, 3, 512, 1024), '18-conv')
            self.buildConv((1, 1, 1024, 512), '19-conv')
            self.buildConv((3, 3, 512, 1024), '20-conv')
            self.buildConv((1, 1, 1024, 512), '21-conv')
            self.buildConv((3, 3, 512, 1024), '22-conv')
            #####################above is darknet-19 convolutional part
            self.darknet19_end=True
            self.buildConv((3, 3, 1024, 1024), '23-conv',)
            branch_3 = self.buildConv((3, 3, 1024, 1024), '24-conv')
            self.buildRoute('25-route',branch_1)
            self.buildConv((1, 1, 512, 64), '26-conv')
            branch_2 = self.buildReorg('27-reorg')
            self.buildRoute('28-route', branch_2,branch_3)
            self.buildConv((3, 3, 1280, 1024), '29-conv')
            self.buildConv((1, 1, 1024, self.box_num*(5+self.classes)), '30-conv',batch_norm=False,activation='linear')
            self.output=tf.identity(self.temp,name='output')#none,h,w,c
            self.buildloss()
            
    def buildConv(self,shape,scope,batch_norm=True,activation='leaky'):
        with tf.variable_scope(scope):
            if self.isTransferLearning and hasattr(self,'darknet19_end') and self.darknet19_end:
                weights = {}
                weights['filter']=tf.get_variable(name='filter',shape=shape,initializer=tf.contrib.layers.xavier_initializer())
                weights['gamma']=tf.get_variable(name='gamma',shape=(shape[-1]),initializer=tf.keras.initializers.Ones())
                weights['mv']=tf.get_variable(name='moving_variance',shape=(shape[-1]),initializer=tf.keras.initializers.Ones())
                weights['ma']=tf.get_variable(name='moving_mean',shape=(shape[-1]),initializer=tf.keras.initializers.Zeros())
                weights['bias']=tf.get_variable(name='bias',shape=(shape[-1]),initializer=tf.keras.initializers.Zeros())
            else:
                weights = self.loader.nextConvBlock(shape=shape, scope=scope, batch_norm=batch_norm)
            self.temp = tf.nn.conv2d(self.temp, filter=weights['filter'], strides=[1,1,1,1], padding="SAME")
            if(batch_norm):
                self.temp = tf.nn.batch_normalization(self.temp, mean=weights['ma'], variance=weights['mv'],scale=weights['gamma'], variance_epsilon=1e-5, offset=0)
            self.temp = tf.nn.bias_add(self.temp, bias=weights['bias'])
            if(activation=='leaky'):
                self.temp = tf.nn.leaky_relu(self.temp, alpha=0.1)
            return self.temp
    def buildMax(self,scope):
        with tf.name_scope(scope):
            self.temp = tf.nn.max_pool(self.temp, (1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
            return self.temp
    def buildReorg(self,scope):
        with tf.name_scope(scope):
            self.temp = tf.extract_image_patches(self.temp, [1, 2, 2, 1], [1, 2, 2, 1], [1, 1, 1, 1], 'VALID')
            return self.temp
    def buildRoute(self,scope,*branch):
        with tf.name_scope(scope):
            if(len(branch)==1):
                self.temp = tf.identity(branch[0])
                return self.temp
            bs=[]
            for b in branch:
                bs.append(tf.identity(b))
            self.temp = tf.concat(bs,axis=-1)
            return self.temp

    def buildloss(self):
        """

        incur iou loss on assigned predbox: (1-iou)^2 , scale 5
        incur iou loss on unassigned predbox that has lower than threshold iou with any of truths: (iou-0)^2
        incur no iouconf loss on other predbox

        incur box loss on assigned predbox: (predbox-truthbox)^2 , scale 2-truthboxsize
        incur box loss on unassigned predbox: (predbox-[0.5,0.5,0,0])^2 , scale 0.01

        incur class loss on assigned predbox: (predclass-truthclass)^2

        """
        batchsize = tf.shape(self.output)[0]
        outputw = tf.shape(self.output)[1]
        outputh = outputw
        anchorbroadcast = tf.reshape(self.anchors, [self.boxnum, 2])

        _ = tf.reshape(self.label_placeholder,[-1,5+self.classnum])
        boxtruth = _[:,0:4]
        ioutruth = _[:,4]
        classtruth = _[:,5:]
        objmask = tf.cast(tf.not_equal(boxtruth[:,0],0),dtype=tf.int8)
        noobjmask = tf.cast(tf.equal(boxtruth[:,0],0),dtype=tf.int8)

        gridx,gridy = tf.meshgrid(tf.range(outputw),tf.range(outputw),indexing='xy')
        grid = tf.expand_dims(tf.expand_dims(tf.stack([gridx,gridy],axis=-1),axis=2),axis=0)

        _=tf.reshape(self.output,[batchsize,outputw,outputw,self.boxnum,-1])
        ioupred = tf.sigmoid(_[:,:,:,:,4])
        classpred = tf.nn.softmax(_[:,:,:,:,5:])
        xypred = tf.sigmoid(_[:,:,:,:,0:2])
        whpred = _[:,:,:,:,2:4]
        whpredexp = tf.exp(whpred)
        whpredreal = whpredexp*anchorbroadcast
        luxypred = grid+xypred-whpredreal/2
        drxypred = grid + xypred + whpredreal / 2
        luxypred = tf.reshape(luxypred,[batchsize,-1,2])
        drxypred = tf.reshape(drxypred, [batchsize, -1, 2])
        luxypredtile = tf.tile(tf.expand_dims(tf.reshape(luxypred,[batchsize,-1,2]),axis=2),[1,1,tf.shape(self.truth_holder)[1]])
        drxypredtile = tf.tile(tf.expand_dims(tf.reshape(drxypred, [batchsize, -1, 2]), axis=2),[1, 1, tf.shape(self.truth_holder)[1]])

        truths = self.truth_holder
        truths = tf.expand_dims(truths,axis=1)
        lutruth = truths[:,:,:,0:2]
        drtruth = truths[:,:,:,2:4]
        lumax = tf.maximum(luxypredtile,lutruth)
        drmin = tf.minimum(drxypredtile,drtruth)
        sectw = drmin[:,:,:,0] - lumax[:,:,:,0]
        secth = drmin[:, :, :, 1] - lumax[:, :, :, 1]
        sect = sectw*secth
        union = (drxypredtile[:,:,:,0]-luxypredtile[:,:,:,0])*(drxypredtile[:,:,:,1]-luxypredtile[:,:,:,1])+ \
                (drtruth[:,:,:,0]-lutruth[:,:,:,0])*(drtruth[:,:,:,1]-lutruth[:,:,:,1])-sect
        iou = sect/union
        maxiou = tf.reduce_max(iou,axis=-1)
        lowoverlapmask = tf.cast(tf.reshape(tf.less(maxiou,0.5),[-1]),dtype=tf.int8)

        xypred = tf.reshape(xypred, [-1, 2])
        whpred = tf.reshape(whpred, [-1, 2])
        boxpred = tf.concat([xypred,whpred],axis=-1)
        classpred = tf.reshape(classpred,[-1,self.classnum])
        boxtruth = boxtruth[:,2:4]
        truthboxsize = tf.reduce_prod(tf.exp(boxtruth)*tf.tile(anchorbroadcast,[batchsize*outputw*outputw]))/tf.square(outputw)
        iouerror = tf.reduce_sum((lowoverlapmask*noobjmask)*(ioupred**2)+objmask*((1-ioupred)**2)*5)
        boxerror = tf.reduce_sum(noobjmask*((xypred-0.5)**2+whpred**2))*0.01+objmask*tf.reduce_sum((boxtruth-boxpred)**2*(2-truthboxsize))
        classerror = tf.reduce_sum(objmask*((classtruth-classpred)**2))
        self.error = iouerror+boxerror+classerror
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.error)
        return self.error



