import numpy as np
from imutils import grab_contours
import matplotlib.pyplot as plt
import pathlib
import cv2
import json

def dilation(src:np.float32, dilateSize=1):
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilateSize + 1, 2 * dilateSize + 1),
                                    (dilateSize, dilateSize))
    return cv2.dilate(src.astype('uint8'), element)

def erosion(src, dilateSize=1):
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilateSize + 1, 2 * dilateSize + 1),
                                    (dilateSize, dilateSize))
    return cv2.erode(src.astype('uint8'), element)

def extractShapes(segmentedMap:np.float32, savePath:pathlib.Path):
    segmentedMap = np.where(segmentedMap>0.5,1,0)
    segmentedMap = erosion(segmentedMap, 3)
    contours = cv2.findContours(segmentedMap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = grab_contours(contours)
    shapeDict = {}
    i = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 4000 : 
            pass
        else:
            shapeDict[f'{i}'] ={}
            shapeDict[f'{i}']['area'] = area
            shapeDict[f'{i}']['savePath'] = str(savePath / f'{i}.npy')
            np.save(savePath / f'{i}.npy', contour)
            M = cv2.moments(contour)
            shapeDict[f'{i}']['xTile'] = int(M["m10"] / M["m00"])
            shapeDict[f'{i}']['yTile'] = int(M["m01"] / M["m00"])
            perimeter = cv2.arcLength(contour,True)
            shapeDict[f'{i}']['perimeter'] = perimeter
            _, radiusEnclosingCircle = cv2.minEnclosingCircle(contour)
            areaCircle = 3.1416 * radiusEnclosingCircle * radiusEnclosingCircle
            circleness = area/areaCircle
            shapeDict[f'{i}']['circleness'] = circleness
            (x, y, w, h) = cv2.boundingRect(contour)
            shapeDict[f'{i}']['rectangleness'] = area / (w*h)
            i += 1

    print(shapeDict)

    with open(savePath / f'shapeDict.json', 'w') as outfile:
        json.dump(shapeDict, outfile)


def coloriseMap(segmentedMap:np.float32, savePath:pathlib.Path) -> np.uint8:
    segmentedMap = np.where(segmentedMap>0.5,1,0)
    segmentedMap = erosion(segmentedMap, 3)
    contours = cv2.findContours(segmentedMap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = grab_contours(contours)
   
    colorisedMap = np.ones((np.shape(segmentedMap)[0], np.shape(segmentedMap)[1],3), dtype=np.uint8)*128
    for i, contour in enumerate(contours):
        area =  cv2.contourArea(contour)
        if area<6000:
                cv2.drawContours(colorisedMap, contours, i, (255,0,0), -1) 
        elif 6000<area and area < 10000:
                cv2.drawContours(colorisedMap, contours, i, (0,255,0), -1) 
        else:
                cv2.drawContours(colorisedMap, contours, i, (0,0,255), -1) 
            
    return colorisedMap
