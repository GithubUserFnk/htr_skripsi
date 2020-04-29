import os
import cv2 as cv
import numpy as np
import pandas as pd
from collections import OrderedDict
from matplotlib import pyplot as plt
from segmentation import segmentImage

dir_path = "dataset/"
subplot = 170
fn = 0
words_img_count = 0

class PreprocessImageDataset(object):
    def __init__(self):
        self.dir_path = "dataset/"
        self.subplot = 160
        self.kernel = np.ones((5,5),np.uint8)

    def preprocessor(self):
        for root, dirs, files in os.walk("dataset"):
            for file in files:
                img = cv.imread(dir_path + file)
                gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                plt.subplot(subplot+1)
                plt.imshow(gray_img)

                denoised_img = self._denoising(gray_img)
                plt.subplot(subplot+2)
                plt.imshow(denoised_img)

                binarized_img =  self._binarize(denoised_img)
                plt.subplot(subplot+3)
                plt.imshow(binarized_img)
  
                sobel_img = self._slant_corrector(binarized_img)
                plt.subplot(subplot+4)
                plt.imshow(sobel_img)

                dilated_img = self._dilate_image(sobel_img)
                plt.subplot(subplot+6)
                plt.imshow(dilated_img)

                centered_img = self._deskew(dilated_img)
                plt.subplot(subplot+7)
                plt.imshow(centered_img)

                Segmentation(self._deskew(img)).process()
                img_session += 1
                # plt.show()
                # break
    
    def _denoising(self, img):
        return cv.fastNlMeansDenoising(img,None,3,7,21)

    def _binarize(self, img):
        # _, treshold_img = cv.threshold(img,0,255, cv.THRESH_OTSU)
        th, treshold_img = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
        return treshold_img

    def _slant_corrector(self, img):
        edges = cv.Canny(img, 100, 200)
        img_sobelx = cv.Sobel(edges, cv.CV_8U, 1, 0, ksize=5)
        img_sobely = cv.Sobel(edges, cv.CV_8U, 0, 1, ksize=5)
        img_sobel = img_sobelx + img_sobely
        return img_sobel

    def _dilate_image(self, img):
        # dilated_img = cv.dilate(img, self.kernel, iterations = 1)
        dilated_img = cv.dilate(img, None ,iterations = 4)
        return dilated_img

    def _deskew(self, img):
        thresh=img
        edges = cv.Canny(thresh,50,200,apertureSize = 3)
        
        lines = cv.HoughLines(edges,1,np.pi/1000, 55)
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
            non_zero_pixels = cv.findNonZero(thresh)
            center, wh, theta = cv.minAreaRect(non_zero_pixels)
            angle=list(t1.keys())[0]
            if angle>160:
                angle=180-angle
            if angle<160 and angle>20:
                angle=12        
            root_mat = cv.getRotationMatrix2D(center, angle, 1)
            rows, cols = img.shape
            rotated = cv.warpAffine(img, root_mat, (cols, rows), flags=cv2.INTER_CUBIC)
        except:
            rotated=img
            pass
        return rotated


class Segmentation(object):
    def __init__(self, img):
        self.img = img
        self.min_pixel_threshold = 25
        self.min_separation_threshold = 35
        self.min_round_letter_threshold = 190
        

    def process(self):
        textLines = self.line_segment()
        imgList = self.word_segment(textLines)
        print ('No. of Words',len(imgList))
        counter = 0
        for letterGray in imgList:
            print ('LetterGray shape: ',letterGray.shape)
            gray = cv.cvtColor(letterGray, cv.COLOR_BGR2GRAY)
            th, letterGray = cv.threshold(
                gray,
                127,
                255,
                cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
            letter2 = letterGray.copy()
            letterGray = cv.dilate(letterGray,None,iterations = 5)
            upoints, dpoints = self._find_cap_points(letterGray)
            upper_baseline, lower_baseline = self.baselines(
                upoints,
                dpoints,
                letterGray,
                letter2
            )
            seg = self.visualize(letterGray, letter2, upper_baseline, lower_baseline)
            words = self.segment_characters(seg,letterGray)
            result_word_path = "./result/characters/{}.jpeg"
            for word in words:
                print('word', word)
                cv.imwrite(
                    result_word_path.format(counter),
                    word
                )
                counter += 1

    def line_segment(self):
        gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        th, threshed = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
    
        upper=[]
        lower=[]
        flag=True
        for i in range(threshed.shape[0]):
            col = threshed[i:i+1,:]
            cnt=0
            if flag:
                cnt=np.count_nonzero(col == 255)
                if cnt >0:
                    upper.append(i)
                    flag=False
            else:
                cnt=np.count_nonzero(col == 255)
                if cnt <2:
                    lower.append(i)
                    flag=True
        text_lines = []
        if len(upper)!= len(lower):lower.append(threshed.shape[0])
        for i in range(len(upper)):
            timg = self.img[upper[i]:lower[i],0:]
            
            if timg.shape[0]>5:
                timg=cv.resize(timg,((timg.shape[1]*5,timg.shape[0]*8)))
                text_lines.append(timg)

        return text_lines

    def word_segment(self, text_lines):
        wordImgList=[]
        counter=0
        cl=0
        for txt_line in text_lines:
            print('txt line', txt_line)
            gray = cv.cvtColor(txt_line, cv.COLOR_BGR2GRAY)
            th, threshed = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
            final_thr = cv.dilate(threshed,None,iterations = 20)
            
            contours, hierarchy = cv.findContours(
                final_thr,
                cv.RETR_EXTERNAL,
                cv.CHAIN_APPROX_SIMPLE
            )
            boundingBoxes = [cv.boundingRect(c) for c in contours]
            (contours, boundingBoxes) = zip(
                *sorted(
                    zip(contours, boundingBoxes), key=lambda b: b[1][0], reverse=False))
        
            for cnt in contours:
                area = cv.contourArea(cnt)
                if area > 10000:
                    print ('Area= ',area)
                    x,y,w,h = cv.boundingRect(cnt)
                    print (x,y,w,h)
                    letterBgr = txt_line[0:txt_line.shape[1],x:x+w]
                    wordImgList.append(letterBgr)
                    words_result_path = "./result/words/{}.jpg".format(words_img_count)
                    cv.imwrite(words_result_path,letterBgr)
                    words_img_count += 1
            cl=cl+1
        return wordImgList

    def _fit_to_size(self, thresh1):
        mask = thresh1 > 0
        coords = np.argwhere(mask)

        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top
        cropped = thresh1[x0:x1,y0:y1]
        return cropped

    def _find_cap_points(self, img):
        cpoints = []
        dpoints = []
        for i in range(img.shape[1]):
            col = img[:,i:i+1]
            k = col.shape[0]
            while k > 0:
                if col[k-1]==255:
                    dpoints.append((i,k))
                    break
                k-=1
            
            for j in range(col.shape[0]):
                if col[j]==255:
                    cpoints.append((i,j))
                    break
        return cpoints,dpoints

    def baselines(self, upoints, dpoints, letter, letter2):
        ##-------------------------Creating upper baseline-------------------------------##
        colu = []
        for i in range(len(upoints)):
            colu.append(upoints[i][1])
        
        maxyu = max(colu)
        minyu = min(colu)
        avgu = (maxyu + minyu) // 2
        meanu = np.around(np.mean(colu)).astype(int)
        print('Upper:: Max, min, avg, mean:: ',maxyu, minyu, avgu, meanu)
            
        ##-------------------------------------------------------------------------------##
        ##-------------------------Creating lower baseline process 1--------------------------##
        cold = []
        for i in range(len(dpoints)):
            cold.append(dpoints[i][1])
        
        maxyd = max(cold)
        minyd = min(cold)
        avgd = (maxyd + minyd) // 2
        meand = np.around(np.mean(cold)).astype(int)
        print('Lower:: Max, min, avg, mean:: ',maxyd, minyd, avgd, meand)

        ##-------------------------------------------------------------------------------##
        ##-------------------------Creating lower baseline process 2---------------------------##
        cn = []
        count = 0

        h = letter.shape[0]
        w = letter.shape[1]

        for i in range(h):
            for j in range(w):
                if(letter[i,j] == 255):
                    count+=1
            if(count != 0):
                cn.append(count)
                count = 0    
        maxindex = cn.index(max(cn))
        print('Max pixels at: ',maxindex)
            
        ##------------------Printing upper and lower baselines-----------------------------##     
        cv.line(letter2,(0,meanu),(w,meanu),(255,0,0),2)
        lb = 0
        if(maxindex > meand):
            lb = maxindex
            cv.line(letter2,(0,maxindex),(w,maxindex),(255,0,0),2)
        else:
            lb = meand
            cv.line(letter2,(0,meand),(w,meand),(255,0,0),2)
        return meanu, lb

    def histogram(self, letter2, cropped):
        ##------------Making Histograms (Default)------------------------######
        colcnt = np.sum(cropped==255, axis=0)
        x = list(range(len(colcnt)))
        return colcnt

    def visualize(self, letter, letter2, upper_baseline, lower_baseline):
        seg = []
        seg1 = []
        seg2 = []
        h = letter.shape[0]
        w = letter.shape[1]

        cropped = letter2[upper_baseline:lower_baseline,0:w]
        colcnt = self.histogram(letter2, cropped)
        ## Check if pixel count is less than min_pixel_threshold, add segmentation point
        for i in range(len(colcnt)):
            if(colcnt[i] < self.min_pixel_threshold):
                seg1.append(i)
            
        ## Check if 2 consequtive seg points are greater than min_separation_threshold in distance
        for i in range(len(seg1)-1):
            if(seg1[i+1]-seg1[i] > self.min_separation_threshold):
                seg2.append(seg1[i])

        # Modified segmentation for removing circles----------------------------###            
        arr=[]
        for i in (seg2):
            arr1 = []
            j = upper_baseline
            while(j <= lower_baseline):
                if(letter[j,i] == 255):
                    arr1.append(1)
                else:
                    arr1.append(0)
                j+=1
            arr.append(arr1)
        print('At arr Seg here: ', seg2)
        
        ones = []
        for i in (arr):
            ones1 = []
            for j in range(len(i)):
                if (i[j] == 1):
                    ones1.append([j])
            ones.append(ones1)
        
        diffarr = []
        for i in (ones):
            diff = i[len(i)-1][0] - i[0][0]
            diffarr.append(diff)
        print('Difference array: ',diffarr)
        
        for i in range(len(seg2)):
            if(diffarr[i] < self.min_round_letter_threshold):
                seg.append(seg2[i])

        ## Make the Cut 
        for i in range(len(seg)):
            letter3 = cv.line(letter2,(seg[i],0),(seg[i],h),(255,0,0),2)
        
        print("Does it work::::")

        return seg

    def segment_characters(self, seg, lettergray):
        s = 0
        wordImgList = []
        fn = 0
        for i in range(len(seg)):
            if i==0:
                s=seg[i]
                if s > 15:
                    wordImg = lettergray[0:,0:s]
                    cntx=np.count_nonzero(wordImg == 255) 
                    print ('count',cntx)
                    fn+=1
                else:
                    continue
            elif (i != (len(seg)-1)):
                if seg[i]-s > 15:
                    wordImg = lettergray[0:,s:seg[i]]
                    cntx=np.count_nonzero(wordImg == 255) 
                    print ('count',cntx)
                    fn+=1
                    s=seg[i]
                else:
                    continue
            else:
                wordImg = lettergray[0:,seg[len(seg)-1]:]
                cntx=np.count_nonzero(wordImg == 255) 
                print ('count',cntx)
                fn=fn+1
            wordImgList.append(wordImg)

        return wordImgList



PreprocessImageDataset().preprocessor()      
        

        