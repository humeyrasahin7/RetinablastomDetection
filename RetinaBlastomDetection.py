
import numpy as np
import cv2
from math import hypot
import math
import imutils

img = cv2.imread('C:/Users/Humeyra/Desktop/imgs/3.jpg')
#face detection
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]


    rect = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    imgcr = cv2.resize(img[y:y+h, x:x+w],(150,150),interpolation = cv2.INTER_AREA)
 
except NameError:
    print("Can not detect the face try another photo")
    exit()




cv2.imshow("croped", imgcr)
cv2.imwrite("C:/Users/Humeyra/Desktop/croped.jpg",imgcr)
#rgb maske

rgb_maske = imgcr.copy()
b, g, r = imgcr[:,:,0], imgcr[:,:,1], imgcr[:,:,2]
_,treshb = cv2.threshold(b,127,255,cv2.THRESH_BINARY_INV)
cv2.imshow('maske',treshb)
cv2.waitKey(0)

_,treshg = cv2.threshold(g,130,255,cv2.THRESH_BINARY_INV)
cv2.imshow('maske',treshg)
cv2.waitKey(0)

_,treshr = cv2.threshold(r,130,255,cv2.THRESH_BINARY_INV)
cv2.imshow('maske',treshr)
cv2.waitKey(0)

rg = cv2.bitwise_and(treshg,treshg,mask=treshr)
rgb = cv2.bitwise_and(rg,rg,mask=treshb)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2)) #se

dilated_rgb = cv2.dilate(rgb,kernel)
cv2.imshow('dilated rgb',dilated_rgb)
cv2.waitKey(0)
cv2.imwrite("C:/Users/Humeyra/Desktop/dilatedRGB.jpg",dilated_rgb)


median_drgb = cv2.medianBlur(dilated_rgb,5)
cv2.imshow('median rgb',median_drgb)
cv2.waitKey(0)

#imfill deneme
#im_floodfill = median_drgb.copy()
#h, w = median_drgb.shape[:2]
#mask = np.zeros((h+2, w+2), np.uint8)
#cv2.floodFill(im_floodfill, mask, (0,0), 255);
#im_floodfill_inv = cv2.bitwise_not(im_floodfill)
#im_out = median_drgb | im_floodfill_inv
#cv2.imshow("filled ", im_out)
#cv2.waitKey(0)

#median_filled = cv2.medianBlur(im_out,5)
#cv2.imshow('median filled',median_filled)
#cv2.waitKey(0)



kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4)) #se
dilated2 = cv2.dilate(median_drgb,kernel2)

median_filled2 = cv2.medianBlur(dilated2,3)
cv2.imshow('median filled dilated 2 ',dilated2)
cv2.waitKey(0)

cv2.imshow('median filled2 ',median_filled2)
cv2.waitKey(0)
cv2.imwrite("C:/Users/Humeyra/Desktop/medianfilled2.jpg",median_filled2)



opening = cv2.morphologyEx(median_filled2, cv2.MORPH_OPEN, kernel2)
_,thopen = cv2.threshold(b,166,255,cv2.THRESH_BINARY)

### bwareopen deneme 
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(median_filled2, connectivity=8)
sizes = stats[1:, -1]; nb_components = nb_components - 1

min_size = 200 

bwop1 = np.zeros((output.shape))

for i in range(0, nb_components):
    if sizes[i] >= min_size:
        bwop1[output == i + 1] = 255
bwop1 = np.uint8(bwop1)


cv2.imshow('bwopen deneme',bwop1)
cv2.waitKey(0)
cv2.imwrite("C:/Users/Humeyra/Desktop/bwop1.jpg",bwop1)




maskedFinal = cv2.bitwise_and(imgcr,imgcr,mask=bwop1)
cv2.imshow('rgb masked',maskedFinal)
cv2.waitKey(0)
cv2.imwrite("C:/Users/Humeyra/Desktop/rgbmaskedfinal.jpg",maskedFinal)


ycbcr = cv2.cvtColor(maskedFinal,cv2.COLOR_BGR2YCrCb)
finalY = ycbcr[:,:,0]

grayl = cv2.cvtColor(maskedFinal,cv2.COLOR_BGR2GRAY)
hsvF = cv2.cvtColor(maskedFinal,cv2.COLOR_BGR2HSV)
finalH = hsvF[:,:,0]
finalS = hsvF[:,:,1]
finalG = maskedFinal[:,:,1]

_,hF = cv2.threshold(finalH,102,255,cv2.THRESH_BINARY_INV)
_,sF = cv2.threshold(finalH,8,255,cv2.THRESH_BINARY)
_,sF2 = cv2.threshold(finalH,114,255,cv2.THRESH_BINARY_INV)
finalS = cv2.bitwise_and(sF,sF,mask = sF2)
_,yF = cv2.threshold(finalY,102,255,cv2.THRESH_BINARY)
_,grayF = cv2.threshold(grayl,127,255,cv2.THRESH_BINARY)
_,greenF = cv2.threshold(finalG,140,255,cv2.THRESH_BINARY)

hs = cv2.bitwise_and(finalS,finalS,mask = hF)
hsy = cv2.bitwise_and(hs,hs,mask = yF)
hsygray = cv2.bitwise_and(hsy,hsy,mask= grayF)
ortakmask = cv2.bitwise_and(hsygray,hsygray,mask=greenF)

cv2.imshow('ortak maske',ortakmask)
cv2.waitKey(0)
cv2.imwrite("C:/Users/Humeyra/Desktop/ortakmaske.jpg",ortakmask)



median_ortak = cv2.medianBlur(ortakmask,5)

cv2.imshow('ortak maske median',median_ortak)
cv2.waitKey(0)



dilated_ortak = cv2.dilate(median_ortak,kernel2)
cv2.imshow('final dilated',dilated_ortak)
cv2.waitKey(0)
cv2.imwrite("C:/Users/Humeyra/Desktop/morphOrtak.jpg",dilated_ortak)



##final
final = cv2.bitwise_and(imgcr,imgcr, mask = dilated_ortak)
cv2.imshow('final ',final)
cv2.waitKey(0)
cv2.imwrite("C:/Users/Humeyra/Desktop/GozSon.jpg",final)




#### yüz segmentasyonu ###
ycbcr2 = cv2.cvtColor(imgcr,cv2.COLOR_BGR2YCrCb)
yc = ycbcr2[:,:,0]
cr = ycbcr2[:,:,1]
cb = ycbcr2[:,:,2]

_, maskY = cv2.threshold(yc,80,255,cv2.THRESH_BINARY)
_, maskCr1 = cv2.threshold(cr,135,255,cv2.THRESH_BINARY)
_, maskCr2 = cv2.threshold(cr,180,255,cv2.THRESH_BINARY_INV)
_, maskCb1 = cv2.threshold(cb,135,255,cv2.THRESH_BINARY_INV)
_, maskCb2 = cv2.threshold(cb,85,255,cv2.THRESH_BINARY)

mcr = cv2.bitwise_and(maskCr1,maskCr1,mask=maskCr2)
mcb = cv2.bitwise_and(maskCb1,maskCb1,mask=maskCb2)
mcry = cv2.bitwise_and(mcr,mcr,mask=maskY)
fM = cv2.bitwise_and(mcry,mcry,mask=mcb)
cv2.imshow("mask ycrcb",fM)
cv2.waitKey(0) 
cv2.imwrite("C:/Users/Humeyra/Desktop/maskycrcb.jpg",fM)



medianY = cv2.medianBlur(fM,5)
cv2.imshow("medianfm",medianY)
cv2.waitKey(0) 

nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(medianY, connectivity=8)
sizes = stats[1:, -1]; nb_components = nb_components - 1

min_size = 500 

bwop2 = np.zeros((output.shape))

for i in range(0, nb_components):
    if sizes[i] >= min_size:
        bwop2[output == i + 1] = 255
bwop2 = np.uint8(bwop2)

cv2.imshow("bwop2",bwop2)
cv2.waitKey(0) 



#imfill deneme
im_floodfill = bwop2.copy()
h, w = bwop2.shape[:2]
maskfill = np.zeros((h+2, w+2), np.uint8)
cv2.floodFill(im_floodfill, maskfill, (0,0), 255)
im_floodfill_inv = cv2.bitwise_not(im_floodfill)
m_out = bwop2 | im_floodfill_inv
cv2.imshow("filled ", m_out)
cv2.waitKey(0)
cv2.imwrite("C:/Users/Humeyra/Desktop/filledface.jpg",m_out)


kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)) #se
erode_son = cv2.erode(m_out,kernel3)

cv2.imshow("erode son", erode_son)
cv2.waitKey(0)
cv2.imwrite("C:/Users/Humeyra/Desktop/erodedface.jpg",erode_son)


son = cv2.bitwise_and(dilated_ortak,dilated_ortak, mask = erode_son)
cv2.imshow('finalllll ',son)
cv2.waitKey(0)
#cv2.imwrite("C:/Users/Humeyra/Desktop/yuzsegmen.jpg",son)



nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(son, connectivity=8)
sizes = stats[1:, -1]; nb_components = nb_components - 1

min_size = 10 

bwop2 = np.zeros((output.shape))

for i in range(0, nb_components):
    if sizes[i] >= min_size:
        bwop2[output == i + 1] = 255
son = np.uint8(bwop2)

cv2.imshow("son2",son)
cv2.waitKey(0) 
cv2.imwrite("C:/Users/Humeyra/Desktop/enSon.jpg",son)




nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(son, connectivity=8)
sizes = stats[1:, -1]; 
nb_components1 = nb_components - 1
image = cv2.cvtColor(imgcr, cv2.COLOR_RGB2BGR)
if(nb_components1<=2 and nb_components1>0):
    print("retinablastom might be detected")
    contours, _ = cv2.findContours(son, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0,255,0), 1)
    res = cv2.resize(image,(300,300),interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    cv2.imshow("", bgr)
    cv2.waitKey(0)
    cv2.imwrite("C:/Users/Humeyra/Desktop/bgr.jpg",bgr)

else:
    print("your eyes are healthy")





