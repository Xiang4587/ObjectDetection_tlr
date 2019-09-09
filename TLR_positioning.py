
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
# change this to your caffe root dir
caffe_root = '/home/ssdcaffe'
import os
import sys
sys.path.insert(0, caffe_root + '/python')
import caffe
from google.protobuf import text_format
from caffe.proto import caffe_pb2
import cv2
import yaml
from numpy.linalg import matrix_rank


# In[2]:

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in range(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found
    return labelnames


# In[3]:

def create_mask(image_cp, x1, y1, x2, y2):
    #image_cp = np.zeros(image_cp.shape[:2], dtype = "uint8")
    cv2.rectangle(image_cp, (x1, y1), (x2, y2), 255, -1)  
    #return mask


# In[4]:

def rgb_hist(img, mask):
    #img = cv2.imread('image.jpg')
    
    # 畫出 RGB 三種顏色的分佈圖
    color = ('b','g','r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], mask, [256], [0, 256])
        plt.plot(histr, color = col)
        plt.xlim([0, 256])
    plt.show()


# In[5]:

def HoughCircle(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.medianBlur(gray_img, 5)
    #cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(gray_img,cv2.HOUGH_GRADIENT,1,10,
                              param1=200,param2=50,minRadius=0,maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        #draw the outer circle
        cv2.circle(img, (i[0],i[1]), i[2], (0,255,0), 2)
        #draw the center of the circle
        cv2.circle(img, (i[0],i[1]), 2, (255,0,0), 3)
        
    plt.figure()
    plt.imshow(img)


# In[6]:

def binarization_pro(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img,5)
    img_bin = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,                                   cv2.THRESH_BINARY,11,2)
    img_bin = cv2.cvtColor(img_bin, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(img_bin)
    return img_bin


# In[7]:

#A
'''
f = open('Calibration_revised.yml')
calibration = yaml.load(f, Loader=yaml.FullLoader)
intri = np.array(calibration['CameraMat']['data'])
intri.shape = (3,3)
intri[2][2] = 0.
add_in = np.array([[0],[0],[1]])
intri = np.append(intri, add_in, axis=1)
    
extri = np.array(calibration['CameraExtrinsicMat']['data'])
extri.shape = (4,4)
'''
#A = np.dot(intri, extri)
#B = np.delete(A, (2), axis=1)
#B_inv = np.linalg.inv(B)
#C = np.delete(B, (2), axis=1)
#C = np.delete(C, (2), axis=0)
#C_inv = np.linalg.inv(C)
#Camera_cor = np.array([[210],[167]])
#Camera_cor_de = Camera_cor - np.array([[B[0][2]],[B[2][2]]])


# In[7]:

def Image2World(u,v,calibration):
    #f = open('Calibration_revised.yml')
    #calibration = yaml.load(f, Loader=yaml.FullLoader)
    intri = np.array(calibration['CameraMat']['data'])
    intri.shape = (3,3)
    intri[2][2] = 0.
    add_in = np.array([[0],[0],[1]])
    intri = np.append(intri, add_in, axis=1)
    
    extri = np.array(calibration['CameraExtrinsicMat']['data'])
    extri.shape = (4,4)
    #extri = np.delete(extri, (3), axis=0) #extri.shape = (3,4)
    
    ## Camera_cor = A * Object_cor                              ==> [u,v,1] = A*[X,Y,Z,1]
    A = np.dot(intri, extri) #[u,v,1] = A*[X,Y,Z,1]
    #Camera_cor = np.array([[x],[y],[1]])
    Img_cor = np.array([[u],[v]])
    '''
    #Object_cor = np.array([Xw,Yw,Zw,1])
    ## inv(AT*A)*Camera_cor = AT*A*Object_cor
    B_inv = np.linalg.inv(np.dot(A.T,A)) #inverse (AT*A)
    C_temp = np.dot(B_inv,A.T)
    [X,Y,Z,P] = np.dot(C_temp,Camera_cor) #P=1
    '''
    
    B = np.delete(A, (2), axis=1)                  ##set Zn=0
    #B_inv = np.linalg.inv(B)
    #[X,Y,P] = np.dot(B_inv,Camera_cor)
    Img_cor_de = Img_cor - np.array([[B[0][2]],[B[2][2]]]) ##[u,v]-[a14,a24]
    C = np.delete(B, (2), axis=1)  
    C = np.delete(C, (2), axis=0)  ## [[a11,a12],[a21,a22]]
    C_inv = np.linalg.inv(C)  
    X,Y = np.dot(C_inv,Img_cor_de)
    
    return X,Y


# In[9]:

f = open('Calibration_revised.yml')
calibration = yaml.load(f, Loader=yaml.FullLoader)
intri = np.array(calibration['CameraMat']['data'])
intri.shape = (3,3)
#intri[2][2] = 0.
#add_in = np.array([[0],[0],[1]])
#intri = np.append(intri, add_in, axis=1)
print(intri)   
extri = np.array(calibration['CameraExtrinsicMat']['data'])
extri.shape = (4,4)
extri = np.delete(extri, -1, axis=0)
A = np.dot(intri, extri)
print(extri)
print(A)
#Camera_cor = [[Xw],[Yw],[Zw],1]
#Img_cor = np.array([[u],[v]])


# In[8]:

def World2Image(Xw,Yw,Zw,calibration):
    intri = np.array(calibration['CameraMat']['data'])
    intri.shape = (3,3)
    
    extri = np.array(calibration['CameraExtrinsicMat']['data'])
    extri.shape = (4,4)
    extri = np.delete(extri, -1, axis=0)
    A = np.dot(intri, extri)
    
    World_cor = [[Xw],[Yw],[Zw],[1]]
    U, V, S = np.dot(A,World_cor)
    u = U/S
    v = V/S
    return u,v


# In[33]:

def tlr_detection(detect_image, map_label, is_show=True):
#    model_def = '/home/eric/ssdcaffe/models/VGGNet/TLR/SSD_500x500/deploy.prototxt'
#    model_weights = '/home/eric/ssdcaffe/models/VGGNet/TLR/SSD_500x500/VGG_TLR_SSD_500x500_iter_120000.caffemodel'
  
#    net = caffe.Net(model_def,
#                    model_weights,
#                    caffe.TEST)
    
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape}) # pixel[0-1], [w,h,c], RGB
    
    transformer.set_transpose('data', (2, 0, 1)) 
    
    '''
    transformer.set_mean('data', np.array([104, 117, 123])) 
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0)) # channel: RGB to BGR
    '''
    
    image_resize_width = 500
    image_resize_height = 500
    net.blobs['data'].reshape(1, 3, image_resize_height, image_resize_width)
    #image = caffe.io.load_image(detect_image)    # caffe.io.load image ==> can't read numpyarray

    transformed_image = transformer.preprocess('data', detect_image)
    net.blobs['data'].data[...] = transformed_image

    detections = net.forward()['detection_out']   ### Detection results
    


    # Parse the outputs.
    det_label = detections[0, 0, :, 1]
    det_conf = detections[0, 0, :, 2]
    det_x1 = detections[0, 0, :, 3]
    det_y1 = detections[0, 0, :, 4]
    det_x2 = detections[0, 0, :, 5]
    det_y2 = detections[0, 0, :, 6]

    # Get detections with confidence higher than 0.3
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.3]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(map_label, top_label_indices)
    top_x1 = det_x1[top_indices]
    top_y1 = det_y1[top_indices]
    top_x2 = det_x2[top_indices]
    top_y2 = det_y2[top_indices]

    #image_mat = cv2.imread(detect_image, cv2.IMREAD_COLOR)
    image_mat = detect_image
    image_h = detect_image.shape[0]
    image_w = detect_image.shape[1]
    image_cp =  detect_image.copy()
    image_mask = np.zeros(image_cp.shape[:2], dtype = "uint8")

    color_red = (0, 0, 255)
    color_green = (0, 255, 0)
    color_yellow = (0, 255, 255)
    color_blue = (255, 0, 0)
    color_text = (0, 0, 0)
    
    
    # Load Calibration file
    f = open('Calibration_revised.yml')
    calibration = yaml.load(f, Loader=yaml.FullLoader)
    
    for i in range(0, top_conf.shape[0]):
        font_face = cv2.FONT_HERSHEY_PLAIN
        font_scale = 1
        text_thickness = 1
        box_thickness = 1

        text_origin = (int(round(top_x1[i] * image_w)), int(round(top_y1[i] * image_h)))
        
        x1 = int(round(top_x1[i] * image_w))
        y1 = int(round(top_y1[i] * image_h))
        x2 = int(round(top_x2[i] * image_w))
        y2 = int(round(top_y2[i] * image_h))
        
        ## Chose a point(Px,Py) to derive the traffic light position according to base_link
        Px = x1+(x2-x1)/8
        Py = y1+(y2-y1)/2
        
        #print('Box'+str(i),x1,y1,x2,y2)
        
        if 'Green' in top_labels[i]:
            cv2.rectangle(image_mat, (x1, y1), (x2, y2), color_green, box_thickness)
            text_size, base = cv2.getTextSize(top_labels[i], font_face, font_scale, text_thickness)
            cv2.rectangle(image_mat, text_origin, (text_origin[0]+text_size[0], text_origin[1]-text_size[1]),                          color_green, cv2.FILLED)
            cv2.putText(image_mat, top_labels[i], text_origin,                        font_face, font_scale, color_text, text_thickness, 1)
            cv2.circle(image_mat,(Px, Py), 2, (255, 0, 0), -1)
            #create_mask(image_mask, x1, y1, x2, y2)
            #object_cor = Image2World(Px,Py,calibration)#[:-1]
           
        elif 'Red' in top_labels[i]:
            cv2.rectangle(image_mat, (x1, y1), (x2, y2), color_red, box_thickness)
            text_size, base = cv2.getTextSize(top_labels[i], font_face, font_scale, text_thickness)
            cv2.rectangle(image_mat, text_origin, (text_origin[0]+text_size[0], text_origin[1]-text_size[1]),                          color_red, cv2.FILLED)
            cv2.putText(image_mat, top_labels[i], text_origin,                        font_face, font_scale, color_text, text_thickness, 1)
            cv2.circle(image_mat,(Px, Py), 2, (255, 0, 0), -1)
            #create_mask(image_mask, x1, y1, x2, y2)
            #object_cor = Image2World(Px,Py,calibration)#[:-1]
            
        elif 'traffic light' in top_labels[i]:
            cv2.rectangle(image_mat, (x1, y1), (x2, y2), color_blue, box_thickness)
            text_size, base = cv2.getTextSize(top_labels[i], font_face, font_scale, text_thickness)
            cv2.rectangle(image_mat, text_origin, (text_origin[0]+text_size[0], text_origin[1]-text_size[1]),                          color_yellow, cv2.FILLED)
            cv2.putText(image_mat, top_labels[i], text_origin,                        font_face, font_scale, color_text, text_thickness, 1)
            cv2.circle(image_mat,(Px, Py), 2, (255, 0, 0), -1)
            #create_mask(image_mask, x1, y1, x2, y2)
            #object_cor = Image2World(Px,Py,calibration)#[:-1]
        
        #print 'Box'+str(i),'image('+str(Px)+','+str(Py)+')','(',object_cor[0],',',object_cor[1],')'
    
    
    #image histogram
    
    #masked = cv2.bitwise_and(image_cp, image_cp, mask = image_mask)
    #hist = cv2.calcHist([image_cp], [0], image_mask, [256], [0, 256])
    #plt.figure()
    #masked_rgb = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
    #plt.imshow(masked_rgb)
    
    #plt.figure()
    #plt.bar(range(1,257), hist)
    #plt.show()
    
    
    #rgb_hist(image_cp, image_mask)
    #img_bin = binarization_pro(image_cp)
    #HoughCircle(img_bin)


    
    if is_show:
        cv2.namedWindow('traffic light recognition', cv2.WINDOW_NORMAL)
        cv2.imshow('traffic light recognition', image_mat)
        
        #cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
        #cv2.imshow('mask', image_mask)
        #cv2.imshow('mask', masked)
        
        cv2.waitKey()
        cv2.destroyAllWindows() 
    return image_mat


# In[34]:

if __name__ == '__main__':
    caffe.set_device(0)
    caffe.set_mode_gpu()
    print('Check Caffe OK!')

    # load label map file
    map_file_name = '/home/eric/ssdcaffe/data/TLR/labelmap_tlr.prototxt'
    map_file = open(map_file_name, 'r')
    label_map = caffe_pb2.LabelMap()
    a = text_format.Merge(str(map_file.read()), label_map)
    
    model_def = '/home/eric/ssdcaffe/models/VGGNet/TLR/SSD_500x500/deploy.prototxt'
    model_weights = '/home/eric/ssdcaffe/models/VGGNet/TLR/SSD_500x500/VGG_TLR_SSD_500x500_iter_120000.caffemodel'
  
    net = caffe.Net(model_def,
                    model_weights,
                    caffe.TEST)

    # Load Video
    VideoPath = '/home/eric/Documents/Autoware/ImageFile/20190220/output.mp4'
    cap = cv2.VideoCapture(VideoPath)
    #cap = cv2.VideoCapture(0)
    '''                       
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('TLR_positioning.mp4',fourcc, 20.0, (640,480))
    '''                       
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        img_mat = tlr_detection(frame, label_map, is_show=False)
        #out.write(cv2.cvtColor(img_mat, cv2.COLOR_RGB2BGR))
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# In[55]:

get_ipython().system('jupyter nbconvert --to script TLR_positioning.ipynb')


# In[14]:

import cv2


# In[ ]:

img = cv2.imread()

