from multiprocessing.sharedctypes import Value
from typing import Tuple
import numpy as np
import math
import cv2
import random
from scipy import ndimage
from skimage.draw import line, disk, ellipse_perimeter, circle_perimeter, rectangle_perimeter
from pathlib import Path, PurePath
import matplotlib.pyplot as plt
import glob
import argparse
from typing import Tuple
from pyprojroot import here
from PIL import Image
import pandas as pd
import json

from threading import Thread
from queue import Queue

q = Queue()

FEATURENAMES = [ 'trees', 'buildings', 'labels' ]

PSMALL  = 0.3
PMEDIUM = 0.3
PLARGE  = 0.3
PHUGE = 0.1

MARGIN  = 5
SPACING = 7
NLINESMAX = 8

smallSizes = [1,2,4,8,16]

def dilation(src:np.float32, dilateSize=1):
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilateSize + 1, 2 * dilateSize + 1),
                                    (dilateSize, dilateSize))
    return cv2.dilate(src.astype('uint8'), element)

def erosion(src, dilateSize=1):
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilateSize + 1, 2 * dilateSize + 1),
                                    (dilateSize, dilateSize))
    return cv2.erode(src.astype('uint8'), element)

def crop(mat:float, MARGIN:int, sizeImg:int, center=True) -> float :
    if center:
        return mat[MARGIN:MARGIN+sizeImg,MARGIN:MARGIN+sizeImg]
    else:
        raise NotImplementedError ("Non-centered Crops are not implemented")

def generateStripePattern(sizeImg:int) -> np.float32:
    enclosingSquareLength = int(sizeImg*math.sqrt(2))
    lines = np.ones((int(enclosingSquareLength),int(enclosingSquareLength)), dtype=np.float32)
    for i in range(1, enclosingSquareLength-SPACING, SPACING):
        for j in [i-1, i, i+1]:
            rr, cc = line(j,0,j,enclosingSquareLength-1)
            lines[rr, cc] = 0
    rotationAngle = random.randint(20,90-20) + random.randint(0,1)*90
    rotatedImage = ndimage.rotate(lines, rotationAngle, reshape=True)
    toCrop = np.shape(rotatedImage)[0]-sizeImg
    return rotatedImage[int(toCrop/2):int(toCrop/2)+sizeImg, int(toCrop/2):int(toCrop/2)+sizeImg]

def generate_ellipsoid(maxLength:int) -> Tuple[list,list]:
    radiusX = random.randint(int(maxLength/4), int(maxLength/3))
    radiusY = random.randint(int(maxLength/4), int(maxLength/3))
    centerX = random.randint(radiusX, maxLength-radiusX )
    centerY = random.randint(radiusY, maxLength-radiusY )
    rr, cc   = ellipse_perimeter(centerX,centerY, radiusX, radiusY)
    return rr, cc
    
def generate_circle(maxLength:int) -> Tuple[list,list]:
    radius = random.randint(int(maxLength/4), int(maxLength/3))
    centerX = random.randint(radius, maxLength-radius )
    centerY = random.randint(radius, maxLength-radius )
    rr, cc   = circle_perimeter(centerX,centerY, radius)
    return rr, cc

def generate_rectangle(maxLength:int) -> Tuple[list,list]:
    extent_x = random.randint(int(maxLength/4), int(maxLength/3))
    extent_y = random.randint(int(maxLength/4), int(maxLength/3))      
    start_x = random.randint(extent_x, maxLength-extent_x)
    start_y = random.randint(extent_y, maxLength-extent_y)        
    start = (start_x, start_y)
    extent = (extent_x, extent_y)
    rr, cc = rectangle_perimeter(start, extent=extent)
    return rr, cc

def generateThickShape(maxRotatedLength:int) -> Tuple[np.float32, np.float32]:
    shapeVar = random.choices(['rectangle', 'circle'], [0.85,0.15])[0]
    if shapeVar == 'rectangle':
        shapeLength = random.randint(int((maxRotatedLength-1)*0.5), maxRotatedLength-1)
        pattern = generateStripePattern(shapeLength)
        mask = np.ones((shapeLength,shapeLength),np.float32)
        mask[0:MARGIN,:] = 0
        mask[shapeLength-MARGIN:,:] = 0
        mask[:,0:MARGIN] = 0
        mask[:,shapeLength-MARGIN:] = 0
        image = mask*pattern
    else:
        pattern = generateStripePattern(maxRotatedLength)
        mask = np.zeros((maxRotatedLength,maxRotatedLength), np.float32)
        maskBackground = np.ones((maxRotatedLength,maxRotatedLength))
        rr, cc = disk((int(maxRotatedLength/2), int(maxRotatedLength/2)), math.ceil(maxRotatedLength/2))
        maskBackground[rr,cc] = 0
        rr, cc = disk((int(maxRotatedLength/2), int(maxRotatedLength/2)), math.ceil(maxRotatedLength/2)-MARGIN)
        mask[rr,cc] = 1
        image = mask*pattern + maskBackground
    rotationAngle = random.randint(0,180)
    rotatedImage = ndimage.rotate(image, rotationAngle, reshape=True, mode='constant', cval=1)
    rotatedMaskSegment = ndimage.rotate(mask, rotationAngle, reshape=True, mode='constant', cval=0)
    return rotatedImage, rotatedMaskSegment

def generateFeature(patternDF:pd.DataFrame, background, tbSizeVar) -> Tuple[np.float32, np.float32]:    
    choice = random.choice(FEATURENAMES)
    while  patternDF[choice][f'{tbSizeVar}'].empty:
        choice = random.choice(FEATURENAMES)            
    element = patternDF[choice][f'{tbSizeVar}'].sample(n=1).iloc[0]    
    mask = np.zeros((element['H'], element['W']), np.float32)
    bkg = np.ones((element['H'], element['W']), np.float32)
    cv2.drawContours(mask,[np.load(here() / element['savePath'])], 0, 1, -1, offset = ( -element['xTile'], -element['yTile']))
    cv2.drawContours(bkg,[np.load(here() / element['savePath'])], 0, 0, -1, offset = ( -element['xTile'], -element['yTile']))
    toDraw = mask*background[element['yTile']:element['yTile']+element['H'], element['xTile']:element['xTile']+element['W']] + bkg
    rotationAngle = random.randint(0,180)
    rotatedImage = ndimage.rotate(toDraw, rotationAngle, reshape=True, mode='constant', cval=1)
    rotatedMask = ndimage.rotate(mask, rotationAngle, reshape=True, mode='constant', cval=0)
    #rotatedImage = toDraw
    #rotatedMask = mask
    return rotatedImage, rotatedMask, choice

def fillThumbnail(thumbnailSize:int, pattern:np.float32, mask:np.float32, boundRowLow:int, boundRowHigh:int, boundColLow:int, boundColHigh:int, imageToFill:np.float32, maskToFill:np.float32)-> Tuple[np.float32, np.float32]:
    thumbnail = np.ones((thumbnailSize,thumbnailSize), np.float32)
    maskToReturn =  np.zeros((thumbnailSize,thumbnailSize), np.float32)
    posX = random.randint(0, thumbnailSize-np.shape(pattern)[0])
    posY = random.randint(0, thumbnailSize-np.shape(pattern)[1])
    thumbnail[posX:posX+np.shape(pattern)[0], posY:posY+np.shape(pattern)[1]] *= pattern
    maskToReturn[posX:posX+np.shape(mask)[0], posY:posY+np.shape(mask)[1]] += mask
    imageToFill[boundRowLow:boundRowHigh, boundColLow:boundColHigh] *= thumbnail
    maskToFill[boundRowLow:boundRowHigh, boundColLow:boundColHigh] += maskToReturn
    return imageToFill, maskToFill

def generateFeatureOrStripe(tbSizeVar:int, patternsDict:dict, boundRowLow:int, boundRowHigh:int, boundColLow:int, boundColHigh:int, image:np.float32, masksDict:dict, pReal=0.8, background=None)-> Tuple[np.float32, np.float32, np.float32]:
    patternVar = random.choices([0,1], [pReal,1-pReal])[0]
    if patternVar ==0:
        pattern, mask, choice = generateFeature(patternsDict, background, tbSizeVar)

    else:
        choice = 'buildings'
        pattern, mask = generateThickShape(int(tbSizeVar/math.sqrt(2)))

    image, mask = fillThumbnail(tbSizeVar, pattern, mask, boundRowLow, boundRowHigh,boundColLow, boundColHigh, image, masksDict[choice])
    masksDict[choice] = mask
    return image, masksDict

def generateBlockOfFlats(sizeImg:int) -> Tuple[np.float32, np.float32, np.float32]:
    MARGIN = 3
    enclosingSquareLength = int(sizeImg*math.sqrt(2))
    mask = np.zeros((enclosingSquareLength,enclosingSquareLength), dtype=np.float32)
    band = np.ones((enclosingSquareLength,enclosingSquareLength), dtype=np.float32)
    middle = int(enclosingSquareLength/2)
    width = random.choices([64,128], [0.5,0.5])[0]
    widthMargin = enclosingSquareLength%width
    band[:, middle-width-MARGIN:middle+width+MARGIN] = 0
    mask[MARGIN*2:widthMargin-MARGIN, middle-width+MARGIN:middle+width-MARGIN] = 1
    for i in range(enclosingSquareLength//width):
        mask[widthMargin+width*i+MARGIN:widthMargin+width*(i+1)-MARGIN, middle-width+MARGIN:middle+width-MARGIN] = 1
    rotationAngle = random.randint(0,180)
    rotatedBand = ndimage.rotate(band, rotationAngle, reshape=True, mode='constant', cval=1)
    rotatedImage = ndimage.rotate(mask*generateStripePattern(enclosingSquareLength), rotationAngle, reshape=True, mode='constant', cval=1)
    rotatedMask = ndimage.rotate(mask, rotationAngle, reshape=True, mode='constant', cval=0)
    cropMargin = int((enclosingSquareLength-sizeImg)/2)
    return crop(rotatedImage+rotatedBand, cropMargin, sizeImg), crop(rotatedMask, cropMargin, sizeImg), crop(rotatedBand, cropMargin, sizeImg)

def is_cell_available(grid, indexRow, indexCol):
    return not bool(grid[indexRow, indexCol])

def try_square_size(grid, indexRow, indexCol, n):
    smSizes = smallSizes.copy()
    while True:
        potentialSize = random.choice(smSizes)
        if indexRow + potentialSize <= n and indexCol + potentialSize <= n:
            if np.count_nonzero(grid[indexRow:indexRow+potentialSize, indexCol:indexCol+potentialSize])==0:
                return potentialSize
        else:
            smSizes.remove(potentialSize)

def addLines(image:np.float32, sizeImg=512):
    lines = np.ones((sizeImg,sizeImg), dtype=np.float32)
    for i in range(random.randint(0,NLINESMAX)):
        if random.randint(0,1) == 0:
            r0, r1 = 0, sizeImg-1
            c0, c1 = random.randint(0,sizeImg-1), random.randint(0,sizeImg-1)
        else:
            c0, c1 = 0,sizeImg-1
            r0, r1 = random.randint(0,sizeImg-1), random.randint(0,sizeImg-1)
        rr, cc = line(r0,c0,r1,c1)
        lines[rr, cc] = 0

    _, image = cv2.threshold(image, 0.2, 1, cv2.THRESH_BINARY)
    return lines*image

def generateFeaturesAndMask(patternsDict:dict, sizeImg=512, background=None, minSize = 32)-> Tuple[np.float32, np.float32, np.float32]:
    image = np.ones((sizeImg,sizeImg), np.float32)
    # CHECK MASKS TO ENSURE THE MEMORY IS NOT SHARED WHEN ALLOCATING
    masksDict = {featureName:np.zeros((sizeImg,sizeImg), np.float32) for featureName in  patternsDict.keys()}
    gridStep = int(sizeImg/minSize)
    grid = np.zeros((gridStep,gridStep), np.float32)
    for indexRow in range(gridStep):
        for indexCol in range(gridStep):
            if is_cell_available(grid, indexRow, indexCol):
                blockSize = try_square_size(grid, indexRow, indexCol, gridStep)
                if blockSize ==smallSizes[-1]:
                    blockOfFlats = random.choices([0,1], [0.5,0.5])[0]
                    if blockOfFlats == 0:
                        imageBloc, maskBloc , band  = generateBlockOfFlats(sizeImg)
                        image = image*band + imageBloc* (1-band)
                        masksDict['buildings']  = masksDict['buildings']*band + maskBloc
                        masksDict['trees'] = masksDict['trees']*band   
                        masksDict['labels'] = masksDict['labels']*band   
                        image = addLines(image)
        
                        return image, masksDict
               
                image, masksDict  = generateFeatureOrStripe(blockSize*minSize, patternsDict, indexRow*minSize, (indexRow+blockSize)*minSize , indexCol*minSize, (indexCol+blockSize)*minSize, image, masksDict, background=background)
                grid[indexRow:indexRow+blockSize, indexCol:indexCol+blockSize] = 1

            else:
                pass
    image = addLines(image)
        
    return image, masksDict

def makeBatch(batchSize, patternsDict, background):
    batch = np.zeros((batchSize, 1, 512, 512))
    batchMasks = {featureName:np.zeros((batchSize, 1, 512, 512)) for featureName in  patternsDict.keys()}
    for index in range(batchSize):
        image, masksDict = generateFeaturesAndMask(patternsDict, background=background)
        batch[index,0] = image
        for key, value in batchMasks.items():
            value[index,0] = masksDict[key]
    return batch, batchMasks

def genMap(savePath, patternsDict, background):
    global q
    Path(f'{savePath}').mkdir(parents=True ,exist_ok=True)
    while True:
        counter = q.get()
        image, masksDict = generateFeaturesAndMask(patternsDict, background=background)
        np.save(f'{savePath}/image_{counter}', image)
        for key, value in masksDict.items():
            np.save(f'{savePath}/mask_{key}_{counter}.npy', value)
        q.task_done()
        
def main(args):
    mapName = '0105033010241'
    cityName = 'Luton'
    
    sizes = [32,64,128,256,512]
    
    background = np.where(np.array(Image.open( here() / f'datasets/cities/{cityName}/500/tp_1/{mapName}.jpg').convert('L'), np.float32) <100, 0, 1)
    patternsDict = {}
    sq2 = math.sqrt(2)
    for featureName in FEATURENAMES:
        patternsDict[featureName] = {}
        fullDf = pd.DataFrame(json.load(open(here() / f'datasets/layers/{featureName}/Luton/0105033010241.json'))[f'{featureName}']).transpose() 
        for size in sizes:
            sLow = size/2
            sHigh = size/sq2
            df1 = fullDf.query('@sLow<H<@sHigh & W<H ')
            df2 = fullDf.query('@sLow<W<@sHigh & H<W ')
            patternsDict[featureName][f'{size}'] = df1.append(df2)

    if args.treatment == "show":    
        counter = len(glob.glob(f'{args.savePath}/image*'))
        for i in range(args.nSamples):
            image, masksDict = generateFeaturesAndMask(patternsDict, background=background, sizeImg=512)
            plt.matshow(image)
            plt.show()
            
            
    elif args.treatment == "save":
        for i in range(args.nSamples):
            q.put(i)
        for t in range(args.maxThreads):
            worker = Thread(target = genMap, args = (args.savePath, patternsDict, background))
            worker.daemon = True
            worker.start()
        q.join()
    
    else:
        raise NotImplementedError ("Can only save or show")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tree Generation')
    parser.add_argument('--datasetPath', required=False, type=PurePath, default = here().joinpath('datasets/patterns'))
    parser.add_argument('--nSamples', required=False, type=int, default = 100)
    parser.add_argument('--savePath', required=False, type=PurePath, default = here().joinpath('datasets/syntheticCities'))
    parser.add_argument('--imageSize', required=False, type=int, default = 512)
    parser.add_argument('--treatment', required=False, type=str, default='show')
    parser.add_argument('--maxThreads', required=False, type=int, default=6)
    args = parser.parse_args()
    
    savePath = Path(args.savePath)
    savePath.mkdir(parents=True, exist_ok=True)
    
    main(args)
