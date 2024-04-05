import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def show_BGR_opencv_image(image:np.ndarray):
    '''
    image:(h,w,3) RGB channel image
    '''
    plt.imshow(X=image)

def show_annotation_poly(lineList:list,origin_image:np.ndarray,points_type:str):
    '''
        points:(x,y)*n 
        origin_image:(h,w,3) BGR channel image
    '''
    copyed_img = np.copy(origin_image)
    assert not (origin_image is copyed_img)
    for line in lineList:
      if line['ignore'] != 1:
        cv.polylines(img=copyed_img,pts=np.expand_dims(line[points_type],axis=0),isClosed=True,color=(0,255,0))
    plt.imshow(X=copyed_img)