import os

basedir = os.getcwd()
log_path = os.path.join(basedir,'log')
epochs = 30
anchors=[[0.57273/13, 0.677385/13], [1.87446/13, 2.06253/13], [3.33843/13, 5.47434/13], [7.88282/13, 3.52778/13], [9.77052/13, 9.16828/13]]
model_path = os.path.join(basedir,'mdl','mdl')
vocimdir = r"D:\Users\yl_gong\Desktop\dl\voc\VOC2012\JPEGImages"
voclabeldir= r"D:\Users\yl_gong\Desktop\dl\voc\VOC2012\Annotations"
trainbatch=10
labels=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog',
                        'horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']