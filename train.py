from config import *
from model import *
from util import *
from tensorflow import keras

writer = tf.summary.FileWriter(log_path,graph=tf.get_default_graph())
saver = tf.train.Saver()
ops = tf.train.AdamOptimizer(learning_rate=0.001).minimize(allerr)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.restore(sess,model_path)

vgg16 = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None)

for i in range(epochs):
    for j, (ims,_xy,_wh,_mask,_cls) in enumerate(getbatch()):
        _inp = vgg16.predict(keras.applications.vgg16.preprocess_input(ims),batch_size=len(ims))
        _ii,e1,e2,e3,e4,_err,_log,_ = sess.run([detector_out,iou_p,xy,wh,cls,allerr,log_all,ops],feed_dict={detector_inp:_inp,
                                         xy_t:_xy,
                                         wh_t:_wh,
                                         mask_box:_mask,
                                         cls_t:_cls})
        #print(_ty)
        #print(_ll)
        #print(e1)
        #print(e2)
        #print(e3)
        #print(e4)
        print(_err)
        writer.add_summary(_log)
        if j % 10 == 0:
            saver.save(sess, model_path)
    print("epoch"+str(i))





