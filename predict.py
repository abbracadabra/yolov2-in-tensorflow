import numpy as np
import tensorflow as tf
import cv2 as cv2
import boxes as bx
import matplotlib.pyplot as plt
import builder as yolo2builder

saved = tf.Graph()
builder = yolo2builder.GraphBuild(saved,r"D:/githubrepo1/darknet/yolov2.weights")
builder.build()
inp = builder.input
outp = builder.output

#test
orig = cv2.imread(r'D:\githubrepo1\darknet\data\dog.jpg')
img_normalize = bx.img_prep(np.copy(orig))[...,::-1]#reverse channel back to RGB

with saved.as_default():
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    output = sess.run(saved.get_tensor_by_name('output:0'),feed_dict={saved.get_tensor_by_name('input:0'):[img_normalize]})

box_coord,box_prob,box_class = bx.getBoxes(np.copy(output[0]))
box_coord,box_prob,box_class = bx.thresholdBox(box_coord,box_prob,box_class,threshold=0.5)
box_coord,box_prob,box_class = bx.nms(box_coord,box_prob,box_class,iou_threshold=0.6,max_box=1000)
box_coord = bx.correct_box(orig,box_coord)
print(box_prob)
img_processed = bx.drawRectangle(np.copy(orig),box_coord,box_prob,box_class)
plt.imshow(img_processed[...,::-1])#reverse channel back to RGB
plt.show()
