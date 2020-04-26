import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

dir_path = "dataset/"

for root, dirs, files in os.walk("dataset"):
    for file in files:
        # membaca gambar sebagai grayscale
        img = cv.imread(dir_path +file,0)
        kernel = np.ones((5,5),np.uint8)

        #noise removing
        dst = cv.fastNlMeansDenoising(img,None,3,7,21)
        plt.subplot(151),plt.imshow(img)
        plt.subplot(152),plt.imshow(dst)

        #Binarization
        _,tresholded_img = cv.threshold(dst,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        plt.subplot(153)
        plt.imshow(tresholded_img)
        # plt.show()
        # break

        #Edge Detection
        img_canny = cv.Canny(tresholded_img,100,200)

        #sobel
        img_sobelx = cv.Sobel(tresholded_img,cv.CV_8U,1,0,ksize=5)
        img_sobely = cv.Sobel(tresholded_img,cv.CV_8U,0,1,ksize=5)
        img_sobel = img_sobelx + img_sobely

        plt.subplot(154)
        plt.imshow(img_sobel)

        #Dilation 
        dilation = cv.dilate(tresholded_img,kernel,iterations = 1)
        h, w = tresholded_img.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv.floodFill(tresholded_img, mask, (1,0), 255);
        im_floodfill_inv = cv.bitwise_not(tresholded_img)
        plt.subplot(155)
        plt.imshow(dilation+im_floodfill_inv)
        # plt.show()
        # break
        
        lines = cv2.HoughLines(dilation+im_floodfill_inv,1,np.pi/1000, 55)
        try:
            d1 = OrderedDict()
        for i in range(len(lines)):
            for rho,theta in lines[i]:
                deg = np.rad2deg(theta)
                if deg in d1:
                    d1[deg] += 1
                else:
                    d1[deg] = 1
                    
        t1 = OrderedDict(sorted(d1.items(), key=lambda x:x[1] , reverse=False))
        print(list(t1.keys())[0],'Angle' ,thresh.shape)
        non_zero_pixels = cv2.findNonZero(thresh)
        center, wh, theta = cv2.minAreaRect(non_zero_pixels)
        angle=list(t1.keys())[0]
        if angle>160:
            angle=180-angle
        if angle<160 and angle>20:
            angle=12        
        root_mat = cv2.getRotationMatrix2D(center, angle, 1)
        rows, cols = img.shape
        rotated = cv2.warpAffine(img, root_mat, (cols, rows), flags=cv2.INTER_CUBIC)
        
    except:
        rotated=img
        pass
    return rotated

        
        

        