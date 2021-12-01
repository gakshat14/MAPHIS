import numpy as np
from imutils import grab_contours
import matplotlib.pyplot as plt

from cv2 import MORPH_RECT, getStructuringElement, dilate, erode, findContours, RETR_TREE, CHAIN_APPROX_SIMPLE, contourArea, drawContours

def dilation(src:np.float32, dilateSize=1):
    element = getStructuringElement(MORPH_RECT, (2 * dilateSize + 1, 2 * dilateSize + 1),
                                    (dilateSize, dilateSize))
    return dilate(src.astype('uint8'), element)

def erosion(src, dilateSize=1):
    element = getStructuringElement(MORPH_RECT, (2 * dilateSize + 1, 2 * dilateSize + 1),
                                    (dilateSize, dilateSize))
    return erode(src.astype('uint8'), element)

def coloriseMap(segmentedMap:np.float32, trees=False) -> np.uint8:
    segmentedMap = np.where(segmentedMap>0.5,1,0)
    segmentedMap = erosion(segmentedMap,3)
    contours = findContours(segmentedMap, RETR_TREE, CHAIN_APPROX_SIMPLE)
    contours = grab_contours(contours)

    colorisedMap = np.ones((np.shape(segmentedMap)[0], np.shape(segmentedMap)[1],3), dtype=np.uint8)*128
    for i, cnt in enumerate(contours):
        area = contourArea(cnt)
        if trees:
            drawContours(colorisedMap, contours, i, (0,255,0), -1) 
        else:
            if area<6000:
                drawContours(colorisedMap, contours, i, (255,0,0), -1) 
            elif 6000<area and area < 10000:
                drawContours(colorisedMap, contours, i, (0,255,0), -1) 
            else:
                drawContours(colorisedMap, contours, i, (0,0,255), -1) 
            
    return colorisedMap
