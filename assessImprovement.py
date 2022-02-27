import pathlib
import numpy as np
import matplotlib.pyplot as plt
import cv2
from imutils import grab_contours
import json
from PIL import Image

areaShapes = {'labels':0, 'trees':0, 'buildings':100}
colorDict = {'labels':(255,0,0), 'trees':(0,255,0), 'buildings':(0,0,255)}

def dilation(src:np.float32, dilateSize=1):
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilateSize + 1, 2 * dilateSize + 1),
                                    (dilateSize, dilateSize))
    return cv2.dilate(src.astype('uint8'), element)

def erosion(src, dilateSize=1):
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilateSize + 1, 2 * dilateSize + 1),
                                    (dilateSize, dilateSize))
    return cv2.erode(src.astype('uint8'), element)

featureNames = ['labels', 'buildings', 'trees']

colorisedMap = np.zeros((7590,11400,3), np.uint8)

for featureName in featureNames:

    cityName = 'Luton'

    tileName = '0105033010241'

    filePath = pathlib.Path(f'datasets/layers/{featureName}/{cityName}')

    background = np.where(np.array(Image.open( f'datasets/cities/{cityName}/500/tp_1/{tileName}.jpg').convert('L'), np.float32) <100, 0, 1)

    targetMask = np.load(filePath.joinpath(f'{tileName}_mask.npy'))
    segmentedMask = np.load(filePath.joinpath(f'{tileName}_segmented.npy'))[157:7590+157, 100:11400+100]

    if featureName == 'buildings':
        segmentedMask = erosion(segmentedMask, 3)

    contours = cv2.findContours(segmentedMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = grab_contours(contours)

    

    nContours = 0

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area <= areaShapes[featureName]:
            pass
        else:        
            M = cv2.moments(contour)
            perimeter = cv2.arcLength(contour,True)
            _, radiusEnclosingCircle = cv2.minEnclosingCircle(contour)
            areaCircle = 3.1416 * radiusEnclosingCircle * radiusEnclosingCircle
            circleness = area/areaCircle
            (x, y, W, H) = cv2.boundingRect(contour)
            rectangleness =  area / (W*H)

            cv2.drawContours(colorisedMap, [contour], 0, colorDict[featureName], -1) 
            nContours += 1

    print(featureName)
    jsonDict = json.load(open(filePath.joinpath(f'{tileName}.json')))
    print(f'Target number of contours : {len(jsonDict[f"{featureName}"])} / detected number of contours : {nContours}')
    areaTarget = targetMask.sum()/255
    detectedTarget = colorisedMap.sum()/255
    print(f'Target area : {areaTarget} / Detected area : {detectedTarget} : {detectedTarget/areaTarget}')

    plt.matshow(colorisedMap)
    plt.show()
