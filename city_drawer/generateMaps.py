from typing import Tuple, Dict
import numpy as np
import math
import cv2
import random
from scipy import ndimage
from skimage.draw import line, disk, ellipse_perimeter, circle_perimeter, rectangle_perimeter
from pathlib import  PurePath
import matplotlib.pyplot as plt
import argparse
from typing import Tuple
from pyprojroot import here
from PIL import Image
import pandas as pd
import json


FEATURENAMES = ( 'trees', 'buildings', 'labels' )
SIZES = (32,64,128,256,512)

PSMALL  = 0.3
PMEDIUM = 0.3
PLARGE  = 0.3
PHUGE = 0.1

MARGIN  = 5
SPACING = 7
NLINESMAX = 8

smallSizes = [1,2,4,8,16]


def generate_ellipsoid(maxLength:int) -> Tuple[list,list]:
    """Generates the coordinates of an ellipsoid in a square

    Args:
        maxLength (int): square dimensions

    Returns:
        Tuple[list,list]: x and y coordinates of the ellipse
    """
    radiusX = random.randint(int(maxLength/4), int(maxLength/3))
    radiusY = random.randint(int(maxLength/4), int(maxLength/3))
    centerX = random.randint(radiusX, maxLength-radiusX )
    centerY = random.randint(radiusY, maxLength-radiusY )
    rr, cc   = ellipse_perimeter(centerX,centerY, radiusX, radiusY)
    return rr, cc
    
def generate_circle(maxLength:int) -> Tuple[list,list]:
    """Generates the coordinates of a circle in a square

    Args:
        maxLength (int): square dimensions

    Returns:
        Tuple[list,list]: x and y coordinates of the circle
    """
    radius = random.randint(int(maxLength/4), int(maxLength/3))
    centerX = random.randint(radius, maxLength-radius )
    centerY = random.randint(radius, maxLength-radius )
    rr, cc   = circle_perimeter(centerX,centerY, radius)
    return rr, cc

def generate_rectangle(maxLength:int) -> Tuple[list,list]:
    """Generates the coordinates of a rectangle in a square

    Args:
        maxLength (int): square dimensions

    Returns:
        Tuple[list,list]: x and y coordinates of the rectangle
    """
    extent_x = random.randint(int(maxLength/4), int(maxLength/3))
    extent_y = random.randint(int(maxLength/4), int(maxLength/3))      
    start_x = random.randint(extent_x, maxLength-extent_x)
    start_y = random.randint(extent_y, maxLength-extent_y)        
    start = (start_x, start_y)
    extent = (extent_x, extent_y)
    rr, cc = rectangle_perimeter(start, extent=extent)
    return rr, cc

def dilation(src:np.float32, dilateSize=1):
    """Dilatation operation, can be applied to the image for map "cleaning"

    Args:
        src (np.float32): source image
        dilateSize (int, optional): Dilatation kernel size. Defaults to 1.

    Returns:
        _type_: dilated image
    """
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilateSize + 1, 2 * dilateSize + 1),
                                    (dilateSize, dilateSize))
    return cv2.dilate(src.astype('uint8'), element)

def erosion(src, dilateSize=1):
    """Erosion operation, can be applied to the image for map "cleaning"

    Args:
        src (np.float32): source image
        dilateSize (int, optional): Erosion kernel size. Defaults to 1.

    Returns:
        _type_: eroded image
    """
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilateSize + 1, 2 * dilateSize + 1),
                                    (dilateSize, dilateSize))
    return cv2.erode(src.astype('uint8'), element)

def crop(mat:np.float32, MARGIN:int, sizeImg:int) -> np.float32 :
    """Center crops a matrix

    Args:
        mat (np.float32): matrix to crop
        MARGIN (int): number of pixels to crop
        sizeImg (int): size of the image to keep

    Returns:
        np.float32: cropped matrix
    """
    return mat[MARGIN:MARGIN+sizeImg,MARGIN:MARGIN+sizeImg]

def generateStripePattern(sizeImg:int) -> np.float32:
    """generates a strippe pattern 

    Args:
        sizeImg (int): generates a strippe pattern 

    Returns:
        np.float32: image to crop
    """
    enclosingSquareLength = int(sizeImg*math.sqrt(2))
    lines = np.ones((int(enclosingSquareLength),int(enclosingSquareLength)), dtype=np.float32)
    for i in range(1, enclosingSquareLength-SPACING, SPACING):
        for j in [i-1, i, i+1]:
            rr, cc = line(j,0,j,enclosingSquareLength-1)
            lines[rr, cc] = 0
    rotationAngle = random.randint(20,90-20) + random.randint(0,1)*90
    rotatedImage = ndimage.rotate(lines, rotationAngle, reshape=True)
    toCrop = np.shape(rotatedImage)[0]-sizeImg
    return crop(rotatedImage,int(toCrop/2), sizeImg) 

def generateThickShape(maxRotatedLength:int) -> Tuple[np.float32, np.float32]:
    """Draw a synthetic feature

    Args:
        maxRotatedLength (int): Maximum lenght for the rotated shape

    Returns:
        Tuple[np.float32, np.float32]: the image, the mask
    """
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
    # Apply random rotation
    rotationAngle = random.randint(0,180)
    rotatedImage = ndimage.rotate(image, rotationAngle, reshape=True, mode='constant', cval=1)
    rotatedMaskSegment = ndimage.rotate(mask, rotationAngle, reshape=True, mode='constant', cval=0)
    return rotatedImage, rotatedMaskSegment

def generateFeature(patternDF:Dict, background:np.float32, thumbnailSize:int) -> Tuple[np.float32, np.float32, str]:  
    """Draws a feature in a small box, later to be placed randomly in the thumbnail 

    Args:
        patternDF (Dict): dictionnary of dataframe (per size per feature)
        background (np.float32): the background of the tile considered
        thumbnailSize (int): size of the thumbnail on which to draw

    Returns:
        Tuple[np.float32, np.float32, str]: the image, the mask and the feature name
    """
    # First, choose randomly a feature to generate  
    choice = random.choice(FEATURENAMES)
    while  patternDF[choice][f'{thumbnailSize}'].empty:
        choice = random.choice(FEATURENAMES)            
    element = patternDF[choice][f'{thumbnailSize}'].sample(n=1).iloc[0]  
    # declare the mask for the feature   
    mask = np.zeros((element['H'], element['W']), np.float32)
    # draw the mask: get 1 value where the shape is
    cv2.drawContours(mask,[np.load(here() / element['savePath'])], 0, 1, -1, offset = ( -element['xTile'], -element['yTile']))
    toDraw = mask*background[element['yTile']:element['yTile']+element['H'], element['xTile']:element['xTile']+element['W']] + (1-mask)
    # apply random rotation
    rotationAngle = random.randint(0,180)
    rotatedImage = ndimage.rotate(toDraw, rotationAngle, reshape=True, mode='constant', cval=1)
    rotatedMask = ndimage.rotate(mask, rotationAngle, reshape=True, mode='constant', cval=0)
    return rotatedImage, rotatedMask, choice

def fillThumbnail(thumbnailSize:int, pattern:np.float32, mask:np.float32, boundRowLow:int, boundRowHigh:int, boundColLow:int, boundColHigh:int, imageToFill:np.float32, maskToFill:np.float32)-> Tuple[np.float32, np.float32]:
    """given a pattern, randomly places it in a thumbnail

    Args:
        thumbnailSize (int): Size of the thumbnail to draw
        pattern (np.float32): pattern to draw
        mask (np.float32): _description_
        boundRowLow (int): lower row bound of the thumbnail in the image
        boundRowHigh (int): higher row bound of the thumbnail in the image
        boundColLow (int): lower col bound of the thumbnail in the image
        boundColHigh (int): higher col bound of the thumbnail in the image
        imageToFill (np.float32): image to complete
        maskToFill (np.float32): mask to complete

    Returns:
        Tuple[np.float32, np.float32]: image, associated mask
    """
    # Once we get the pattern, we draw it at a random position in the thumbnail
    thumbnail = np.ones((thumbnailSize,thumbnailSize), np.float32)
    maskToReturn =  np.zeros((thumbnailSize,thumbnailSize), np.float32)
    posX = random.randint(0, thumbnailSize-np.shape(pattern)[0])
    posY = random.randint(0, thumbnailSize-np.shape(pattern)[1])
    thumbnail[posX:posX+np.shape(pattern)[0], posY:posY+np.shape(pattern)[1]] *= pattern
    maskToReturn[posX:posX+np.shape(mask)[0], posY:posY+np.shape(mask)[1]] += mask
    imageToFill[boundRowLow:boundRowHigh, boundColLow:boundColHigh] *= thumbnail
    maskToFill[boundRowLow:boundRowHigh, boundColLow:boundColHigh] += maskToReturn
    return imageToFill, maskToFill

def generateFeatureOrStripe(thumbnailSize:int, patternsDict:dict, boundRowLow:int, boundRowHigh:int, boundColLow:int, boundColHigh:int, image:np.float32, masksDict:dict, pReal=0.8, background=None)-> Tuple[np.float32, Dict]:
    """Draws a shape on a thumbnail (sub-part) of the image

    Args:
        thumbnailSize (int): Size of the thumbnail to draw
        patternsDict (Dict):  A dictionnary containing all the features of all the sizes considered (not the shapes as objects but the meta information) 
        background (np.float32): the background, loaded as a binary matrix, but float 32 is necessary for later operations (is it tho?)
        boundRowLow (int): lower row bound of the thumbnail in the image
        boundRowHigh (int): higher row bound of the thumbnail in the image
        boundColLow (int): lower col bound of the thumbnail in the image
        boundColHigh (int): higher col bound of the thumbnail in the image
        image (np.float32): Image on which to draw
        masksDict (dict): dictionnary of associated masks (one per feature)
        pReal (float, optional): Probability to draw a real feature (vs a synthetic one). Defaults to 0.8.

    Returns:
        Tuple[np.float32, Dict]: image, dictionnary of associated masks (one per feature)
    """
    # randomly choose the pattern, real or synthetic
    patternVar = random.choices([0,1], [pReal,1-pReal])[0]
    if patternVar ==0:
        pattern, mask, choice = generateFeature(patternsDict, background, thumbnailSize)
    else:
        choice = 'buildings'
        pattern, mask = generateThickShape(int(thumbnailSize/math.sqrt(2)))
    image, mask = fillThumbnail(thumbnailSize, pattern, mask, boundRowLow, boundRowHigh,boundColLow, boundColHigh, image, masksDict[choice])
    masksDict[choice] = mask
    return image, masksDict

def generateBlockOfFlats(sizeImg:int) -> Tuple[np.float32, np.float32, np.float32]:
    """Generates a block of houses to replicate semi-detached houses

    Args:
        sizeImg (int): Size of the image

    Returns:
        Tuple[np.float32, np.float32, np.float32]: (the image, the mask and the band (=the surface occupied if the house were not detached))
    """
    # declare margin to use at the end of blocks
    MARGIN = 3
    # declare the maximum enclosing square that can be drawn in the image considered
    enclosingSquareLength = int(sizeImg*math.sqrt(2))
    # declare the mask and the band
    mask = np.zeros((enclosingSquareLength,enclosingSquareLength), dtype=np.float32)
    band = np.ones((enclosingSquareLength,enclosingSquareLength), dtype=np.float32)
    #declare the middle of the images to locate the block of flats in the center of the image
    middle = int(enclosingSquareLength/2)
    #randomly select width
    width = random.choices([int(sizeImg/8),int(sizeImg/4)], [0.5,0.5])[0]
    widthMargin = enclosingSquareLength%width
    # draw the shapes
    band[:, middle-width-MARGIN:middle+width+MARGIN] = 0
    mask[MARGIN*2:widthMargin-MARGIN, middle-width+MARGIN:middle+width-MARGIN] = 1
    for i in range(enclosingSquareLength//width):
        mask[widthMargin+width*i+MARGIN:widthMargin+width*(i+1)-MARGIN, middle-width+MARGIN:middle+width-MARGIN] = 1
    rotationAngle = random.randint(0,180)
    # apply random rotation
    rotatedBand = ndimage.rotate(band, rotationAngle, reshape=True, mode='constant', cval=1)
    rotatedImage = ndimage.rotate(mask*generateStripePattern(enclosingSquareLength), rotationAngle, reshape=True, mode='constant', cval=1)
    rotatedMask = ndimage.rotate(mask, rotationAngle, reshape=True, mode='constant', cval=0)
    cropMargin = int((enclosingSquareLength-sizeImg)/2)
    return crop(rotatedImage+rotatedBand, cropMargin, sizeImg), crop(rotatedMask, cropMargin, sizeImg), crop(rotatedBand, cropMargin, sizeImg)

def is_cell_available(grid:np.uint8, indexRow:int, indexCol:int) -> bool:
    """Checks if the cell at position (indexRow, indexCol) on the grid is available

    Args:
        grid (np.uint8): the grid 
        indexRow (int): the row index on the surrogate grid
        indexCol (int): the column index on the surrogate grid

    Returns:
        bool: the availability of the cell at position (indexRow, indexCol)
    """
    return not bool(grid[indexRow, indexCol])

def try_square_size(grid:np.uint8, indexRow:int, indexCol:int, gridSize:int) ->int:
    """When randomly partitionning a square into smaller squares, use a surrogate grid (faster) to assess if there is space left to draw in the actual image

    Args:
        grid (np.uint8): the grid 
        indexRow (int): the row index on the surrogate grid
        indexCol (int): the column index on the surrogate grid
        gridSize (int): the grid size

    Returns:
        int: the grid-level size of the shape that can be drawned
    """
    # local copy of the small sizes to consider
    smSizes = smallSizes.copy()
    while True:
        potentialSize = random.choice(smSizes)
        # assert we are not out of the grid
        if indexRow + potentialSize <= gridSize and indexCol + potentialSize <= gridSize:
            # assert the grid is empty for the shape we want to draw
            if np.count_nonzero(grid[indexRow:indexRow+potentialSize, indexCol:indexCol+potentialSize])==0:
                return potentialSize
        else:
            smSizes.remove(potentialSize)

def addLines(image:np.float32, sizeImg=512) ->  np.float32:
    """Adds random lines to the final image

    Args:
        image (np.float32): image without noise
        sizeImg (int, optional):  Size of the image. Defaults to 512. don't mess around with that

    Returns:
        np.float32: image + noise
    """
    # Declare the lines image: lines have value 0, not 1.
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

    # Once lines are drawn, simple element-wise product of the two matrices
    return lines*image

def generateFeaturesAndMask(patternsDict:dict, background:np.float32, sizeImg=512, minSize = 32) -> Tuple[np.float32, Dict] :
    """Generates one image and the associated masks

    Args:
        patternsDict (Dict):  A dictionnary containing all the features of all the sizes considered (not the shapes as objects but the meta information) 
        background (np.float32): the background, loaded as a binary matrix, but float 32 is necessary for later operations (is it tho?)
        sizeImg (int, optional): Size of the image. Defaults to 512. don't mess around with that
        minSize (int, optional): Minimum shape size to consider. Defaults to 32.

    Returns:
        Tuple[np.float32, Dict]: image, dictionnary of associated masks (one per feature)
    """
    image = np.ones((sizeImg,sizeImg), np.float32)
    masksDict = {featureName:np.zeros((sizeImg,sizeImg), np.float32) for featureName in  patternsDict.keys()}
    gridSize = int(sizeImg/minSize)
    grid = np.zeros((gridSize,gridSize), np.uint8)
    for indexRow in range(gridSize):
        for indexCol in range(gridSize):
            if is_cell_available(grid, indexRow, indexCol):
                blockSize = try_square_size(grid, indexRow, indexCol, gridSize)
                # If the block size is the biggest (i.e it fits the whole image), draw a row of semi-detached houses with a probability p=0.5
                if blockSize == smallSizes[-1]:
                    blockOfFlats = random.choices([0,1], [0.5,0.5])[0]
                    if blockOfFlats == 0:
                        imageBloc, maskBloc , band  = generateBlockOfFlats(sizeImg)
                        image = image*band + imageBloc* (1-band)
                        masksDict['buildings']  = masksDict['buildings']*band + maskBloc
                        masksDict['trees'] = masksDict['trees']*band   
                        masksDict['labels'] = masksDict['labels']*band   
                        _, image = cv2.threshold(image, 0.2, 1, cv2.THRESH_BINARY)
                        for key, mask in masksDict.items():
                            _, mask_ = cv2.threshold(mask, 0.2, 1, cv2.THRESH_BINARY)
                            masksDict[key] = mask_
                        return image, masksDict
               
                image, masksDict  = generateFeatureOrStripe(blockSize*minSize, patternsDict, indexRow*minSize, (indexRow+blockSize)*minSize , indexCol*minSize, (indexCol+blockSize)*minSize, image, masksDict, background=background)
                grid[indexRow:indexRow+blockSize, indexCol:indexCol+blockSize] = 1

            else:
                pass

    # Add noise (in form of lines) to finale image        
    image = addLines(image)
    _, image = cv2.threshold(image, 0.2, 1, cv2.THRESH_BINARY)
    for key, mask in masksDict.items():
        _, mask_ = cv2.threshold(mask, 0.2, 1, cv2.THRESH_BINARY)
        masksDict[key] = mask_
    return image, masksDict

def makeBatch(batchSize:int, patternsDict:Dict, background:np.float32) ->Tuple[np.float32, Dict]:
    """Make a batch for image processing

    Args:
        batchSize (int): size of the batch
        patternsDict (Dict):  A dictionnary containing all the features of all the sizes considered (not the shapes as objects but the meta information) 
        background (np.float32): the background, loaded as a binary matrix, but float 32 is necessary for later operations (is it tho?)

    Returns:
        Tuple[np.float32, Dict]: (the image, the dictionnary of per-feature masks)
    """
    # Create empty batch for the image
    batch = np.zeros((batchSize, 1, 512, 512))
    # Create a dictionnary of empty batches for the masks associated to the images
    batchMasks = {featureName:np.zeros((batchSize, 1, 512, 512)) for featureName in  patternsDict.keys()}
    # Populate the image batch and the mask batches with the appropriate data
    for index in range(batchSize):
        image, masksDict = generateFeaturesAndMask(patternsDict, background=background)
        batch[index,0] = image
        for key, value in batchMasks.items():
            value[index,0] = masksDict[key]
    return batch, batchMasks
        
def backgroundLoadingFunction(cityName:str, tileName:str, threshold=100) -> np.float32:
    """Boiler plate code for loading and converting the background

    Args:
        cityName (str): city name
        tileName (str): tile name
        threshold (int, optional): Threshold for binarisation (uint8 to binary) Defaults to 100.

    Returns:
        np.float32: the background, loaded as a binary matrix, but float 32 is necessary for later operations (is it tho?)
    """
    # Load tile using PIL.Image
    background = Image.open( here() / f'datasets/cities/{cityName}/500/tp_1/{tileName}.jpg')
    # Convert RGB to grayscale
    background = background.convert('L')
    # Return binarised (thresholded) image
    return np.where(np.array(background, np.float32) <threshold, 0, 1)

def patternExtraction(featureNames:Tuple, sizes:Tuple, cityName:str, tileName:str) -> Dict:
    """parse a dataframe into a dictionnary because it's more convenient

    Args:
        featureNames (Tuple): Names of the features to consider
        sizes (Tuple): all the possible pattern sizes
        cityName (str): city name
        tileName (str): tile name

    Returns:
        Dict: A dictionnary of dataframes containing all the features of all the sizes considered (not the shapes as objects but the meta information)
    """
    # Declare pattern dictionnary
    patternsDict = {}
    # Declare sqrt(2) for pd query
    sq2 = math.sqrt(2)
    # Iterate through features and sizes to query the features of the given sizes (included rotated) from the dataframe 
    for featureName in featureNames:
        patternsDict[featureName] = {}
        fullDf = pd.DataFrame(json.load(open(here() / f'datasets/layers/{featureName}/{cityName}/{tileName}.json'))[f'{featureName}']).transpose() 
        for size in sizes:
            sLow = size/2
            sHigh = size/sq2
            df1 = fullDf.query('@sLow<H<@sHigh & W<H ')
            df2 = fullDf.query('@sLow<W<@sHigh & H<W ')
            patternsDict[featureName][f'{size}'] = df1.append(df2)
    return patternsDict

def main(args):
    
    background   = backgroundLoadingFunction(args.cityName, args.tileName)
    patternsDict = patternExtraction(FEATURENAMES, SIZES, args.cityName, args.tileName)

    for i in range(args.nSamples):
        image, masksDict = generateFeaturesAndMask(patternsDict, background=background, sizeImg=512)
        plt.matshow(image)
        plt.show()
        for mask in masksDict.values():
            plt.matshow(mask)
            plt.show()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tree Generation')
    parser.add_argument('--datasetPath', required=False, type=PurePath, default = here().joinpath('datasets/patterns'), help='Where the json file of the features is stored')
    parser.add_argument('--nSamples', required=False, type=int, default = 100, help='the number of samples to be displayed')
    parser.add_argument('--imageSize', required=False, type=int, default = 512, help='the size of the images to be generated DONT CHANGE IT RN.')
    parser.add_argument('--cityName', required=False, type=str, default = 'Luton', help='The name of the city in which to take the features from.')
    parser.add_argument('--tileName', required=False, type=str, default = '0105033010241', help='The name of the tile of args.cityName in which to take the features from.')
    args = parser.parse_args()
    
    main(args)
