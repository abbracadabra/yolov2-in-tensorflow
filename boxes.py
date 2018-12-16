import numpy as np
import tensorflow as tf
import cv2 as cv2

priors = [ [0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434], [7.88282, 3.52778], [9.77052, 9.16828] ]
label = ['person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','sofa','pottedplant','bed','diningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']

def getBoxes(output):
    output = np.reshape(output,(19*19*5,85))    
    #print(np.mean(output,axis=0))
    output[:,0:2] = 1/(1+np.exp(output[:,0:2]*-1))
    output[:,4] = 1/(1+np.exp(output[:,4]*-1))
    #output[:,5:] = output[:,5:] - np.max(output[:,5:],axis=1,keepdims=True)
    output[:,5:] = np.exp(output[:,5:]) / np.sum(np.exp(output[:,5:]),axis=1,keepdims=True)
    output[:,2:4] = np.exp(output[:,2:4])*np.tile(priors,[19*19,1]) / [19,19]
    _x,_y = np.meshgrid(np.arange(19),np.arange(19),indexing='xy')
    output[:,0:2] = (np.reshape(np.tile(np.stack([_x,_y],axis=2),[1,1,5]),(-1,2)) + output[:,0:2]) / [19,19]
    
    ul = output[:,0:2] - output[:,2:4]/2
    lr = output[:,0:2] + output[:,2:4]/2
    coord = np.concatenate([ul,lr],axis=1)
    probs = np.max(output[:,4:5]*output[:,5:],axis=-1)
    cls = np.argmax(output[:,5:],axis=-1)
    return np.reshape(coord,(-1,4)),np.reshape(probs,(-1)),np.reshape(cls,(-1))

def thresholdBox(box_coord,box_prob,box_class,threshold):
    mask = box_prob>=threshold
    return box_coord[mask],box_prob[mask],box_class[mask]

def nms(box_coord,box_prob,box_class,iou_threshold,max_box):
    np.clip(box_coord, 0, 1, box_coord)
    indexing=[]
    ix_sorted = np.argsort(box_prob)[::-1]
    for i in np.arange(max_box):
        if len(ix_sorted)<1:
            break
        indexing.append(ix_sorted[0])
        if len(ix_sorted)==1:
            break
        selected_box = box_coord[ix_sorted[0]]
        ix_sorted = ix_sorted[1:]
        a_min_x = selected_box[0]
        a_min_y = selected_box[1]
        a_max_x = selected_box[2]
        a_max_y = selected_box[3]
        if a_max_x<=a_min_x or a_max_y<=a_min_y:
            continue
        mask = np.zeros_like(ix_sorted)
        for j in np.arange(len(ix_sorted)):
            _b = box_coord[ix_sorted[j]]
            b_min_x = _b[0]
            b_min_y = _b[1]
            b_max_x = _b[2]
            b_max_y = _b[3]
            if b_max_x<=b_min_x or b_max_y<=b_min_y:
                continue
            a_area = (a_max_x - a_min_x)*(a_max_y-a_min_y)
            b_area = (b_max_x - b_min_x) * (b_max_y - b_min_y)
            intersect_area = (np.min((a_max_x,b_max_x))-np.max((a_min_x,b_min_x)))*(np.min((a_max_y,b_max_y))-np.max((a_min_y,b_min_y)))
            iou = intersect_area/(a_area+b_area-intersect_area)
            if iou<iou_threshold:
                mask[j] = 1
        ix_sorted = ix_sorted[mask==1]
    return np.take(box_coord,indexing,axis=0),\
            np.take(box_prob,indexing,axis=0),\
            np.take(box_class,indexing,axis=0)
            
#colors = [ [1,0,1], [0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0] ]


def drawRectangle(img,box_coord,box_prob,box_class):
    iw = img.shape[1]
    ih = img.shape[0]
    box_coord = np.int16(box_coord*[iw,ih,iw,ih])
    for i,(coord,prob,cls) in enumerate(zip(box_coord,box_prob,box_class)):
        thickness=3
#         if prob<0.66 : thickness=2
#         if prob<0.33 : thickness=1
        
#         offset = (cls*123457) % len(label);
#         red = get_color(2,offset,len(label));
#         green = get_color(1,offset,len(label));
#         blue = get_color(0,offset,len(label));
        rgb=(100,100,100)
        cv2.rectangle(img, (coord[0],coord[1]), (coord[2],coord[3]),rgb,thickness)
        
        (w,h),b = cv2.getTextSize(label[cls]+' '+str(np.round(prob,decimals=2)),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.7,3)
        cv2.rectangle(img,(coord[0],coord[1]-h-4),(coord[0]+w+4,coord[1]),rgb,thickness=-1)#leave 2 pixel padding around text
        cv2.putText(img,label[cls]+' '+str(np.round(prob,decimals=2)),(coord[0]+2,coord[1]+2),cv2.Formatter_FMT_DEFAULT,0.7,(0,0,0),3)
    return img
       

        
def get_color(c,x,max):
    ratio = (x/max)*5;
    i = np.floor(ratio).astype(int);
    j = np.ceil(ratio).astype(int);
    ratio -= i;
    r = (1-ratio) * colors[i][c] + ratio*colors[j][c];
    return int(r*255)


def img_prep(img):
    iw = img.shape[1]
    ih = img.shape[0]
    new_img = np.ones((608,608,3))*128
    if 608/iw<608/ih:
        neww = 608
        newh = int(ih*(608/iw))
        img = cv2.resize(img,(neww,newh))
        _d = int((608-newh)/2)
        new_img[_d:_d+newh,:,:] = img
    else:
        newh = 608
        neww = int(iw*(608/ih))
        img = cv2.resize(img,(neww,newh))
        _d = int((608-neww)/2)
        new_img[:,_d:_d+neww,:] = img
    new_img/=255
    return new_img
    
        

def correct_box(img,coord):
    iw = len(img[0])
    ih = len(img)
    if 608/iw<608/ih:
        neww = 608
        newh = int(ih*(608/iw))
        _d = int((608-newh)/2) / 608
        coord[:,1] = (coord[:,1]-_d) / (newh/608)
        coord[:,3] = (coord[:,3]-_d) / (newh/608)
    else:
        newh = 608
        neww = int(iw*(608/ih))
        _d = int((608-neww)/2) / 608
        coord[:,0] = (coord[:,0]-_d)/(neww/608)
        coord[:,2] = (coord[:,2]-_d)/(neww/608)
    np.clip(coord,0,1,out=coord)
    return coord

        
    
        















