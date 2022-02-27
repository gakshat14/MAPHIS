#!/usr/bin/env python
# coding: utf-8
from os import read
from pathlib import Path
from torch.utils.data import Dataset
import glob
from PIL import Image
import random
import torch.nn as nn
import torch
from torchvision.transforms import ToTensor
import numpy as np
import cv2
import pandas as pd
import json
from PIL.TiffTags import TAGS
from torch.nn.functional import one_hot

def dilation(src:np.float32, dilateSize=1):
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilateSize + 1, 2 * dilateSize + 1),
                                    (dilateSize, dilateSize))
    return cv2.dilate(src.astype('uint8'), element)

def erosion(src, dilateSize=1):
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilateSize + 1, 2 * dilateSize + 1),
                                    (dilateSize, dilateSize))
    return cv2.erode(src.astype('uint8'), element)


def csv_from_excel(fileName):
    pd.read_excel('./'+fileName).to_csv('./'+fileName[:fileName.index('.')]+'.csv', index=False)
    
def normalise(tensor):
    return (tensor-tensor.min())/(tensor.max()-tensor.min()) 

def identity(tensor):
    return tensor

class syntheticCity(Dataset):
    def __init__(self, datasetPath, train=True, fileFormat='npy', transform = None):
        self.fileFormat = fileFormat
        lenDataset = len(glob.glob(datasetPath+"/maskTrees_*."+fileFormat))
        if train:
            self.maskTrees   = glob.glob(datasetPath+"/maskTrees_*."+fileFormat)[int(lenDataset*0.1):-1]
            self.maskStripes = glob.glob(datasetPath+"/maskStripes_*."+fileFormat)[int(lenDataset*0.1):-1]
            self.images = glob.glob(datasetPath+"/image_*."+fileFormat)[int(lenDataset*0.1):-1]
        else:
            self.maskTrees   = glob.glob(datasetPath+"/maskTrees_*."+fileFormat)[0:int(lenDataset*0.1)]
            self.maskStripes = glob.glob(datasetPath+"/maskStripes_*."+fileFormat)[0:int(lenDataset*0.1)]
            self.images = glob.glob(datasetPath+"/image_*."+fileFormat)[0:int(lenDataset*0.1)]

        self.transform= transform

    def __len__(self): 
        return len(self.images)

    def __getitem__(self, index):  
        image = np.load(self.images[index])
        maskTree = np.load(self.maskTrees[index]) 
        maskStripes = np.load(self.maskStripes[index]) 
        i, j = random.randint(0,255), random.randint(0,255)
        if self.transform:
            image, maskTree, maskStripes = self.transform(image), self.transform(maskTree), self.transform(maskStripes)
        return image[:,i:i+256,j:j+256], maskTree[:,i:i+256,j:j+256], maskStripes[:,i:i+256,j:j+256]

class Maps(Dataset):
    def __init__(self, datasetPath:Path, cityName:str, fileFormat='.jpg', transform=None):
        self.mapsPath   = list(datasetPath.glob(f'cities/{cityName}/*/*/*{fileFormat}'))
        if fileFormat == '.jpg': 
            self.height = 7590
            self.width  = 11400  
        elif fileFormat == '.tif':
            self.height = getTiffProperties(Image.open(self.mapsPath[0]), returnDict=True)['ImageLength'][0]
            self.width  = getTiffProperties(Image.open(self.mapsPath[0]), returnDict=True)['ImageWidth'][0]
        else:
            raise Exception('Wrong File format : only png and tif accepted')        
        self.fileFormat = fileFormat
        
        self.evaluationData = list(datasetPath.glob(f'cities/{cityName}/*/*/*.csv'))
        projectionData = list(datasetPath.glob(f'cities/{cityName}/*/*/*.prj'))
        tfwData = list(datasetPath.glob(f'cities/{cityName}/*/*/*.tfw'))
        self.datasetPath = datasetPath      
        self.projections = projectionData
        self.tfwData = tfwData
        self.transform = transform

    def __getitem__(self, index:int):                
        if self.fileFormat == '.tif':
            properties =  getTiffProperties(Image.open(self.mapsPath[index]), returnDict=True)
        else: 
            properties =  'No properties with '+self.fileFormat+' format.'
        map = ToTensor()(Image.open(self.mapsPath[index]))
        projection = open(self.projections[index], 'r').read() 
        metaData = extractMetaData(open(self.tfwData[index], 'r').read())
        boundaries = getBoundaries(metaData, self.height, self.width)
        sample = {'map': map.unsqueeze_(0),
                  'properties':properties,
                  'projection':projection,
                  'metaData':metaData,
                  'boundaries':boundaries,
                  'tilePath':str(self.mapsPath[index]),
                  'mapName':self.mapsPath[index].name
                  }

        if self.transform:
            sample = self.transform(sample)
        return sample    

    def __len__(self): 
        return len(self.mapsPath)
            
class Tiles(Dataset):
    def __init__(self, datasetPath:Path, cityName:str, mapName='0105033050201', transform=None, fromCoordinates=False, mapfileFormat='.jpg', thumbnailFileFormat='.npy', colored=False) -> None:
        self.tilingParameters = json.load(open(datasetPath / 'tilingParameters.json'))
        self.tilesCoordinates = self.tilingParameters['coordinates']
        self.mapName  = mapName
        self.transform=transform
        self.fromCoordinates = fromCoordinates
        self.mapfileFormat = mapfileFormat
        self.thumbnailFileFormat = thumbnailFileFormat
        tfwData = list(datasetPath.glob(f'cities/{cityName}/*/*/{mapName}.tfw'))[0]
        self.projectionData = list(datasetPath.glob(f'cities/{cityName}/*/*/{mapName}.tfw'))[0]
        self.properties =  getTiffProperties(Image.open(next(datasetPath.glob(f'cities/{cityName}/*/*/{mapName}.tif'))), returnDict=True)
        self.colored=colored
        self.boundaries = getBoundaries(extractMetaData(open(tfwData, 'r').read()), 7590, 11400)
        if colored:
            self.classifiedPath = json.load(open(datasetPath / f'classifiedMaps/{cityName}/{mapName}.json'))
            self.cityfolderPath = next(datasetPath.glob(f'coloredMaps/{cityName}') )
            if fromCoordinates:
                im = openfile(self.cityfolderPath / f'{self.mapName}{self.mapfileFormat}', self.mapfileFormat)
                dilated = dilation(im)
                eroded = erosion(im)
                self.fullMap = ToTensor()(np.concatenate((im, dilated, eroded),1))
        else:
            self.cityfolderPath = next(datasetPath.glob(f'cities/{cityName}/*/*') )
            if fromCoordinates:
                self.paddingMap = nn.ConstantPad2d((self.tilingParameters['paddingX'],self.tilingParameters['paddingX'], self.tilingParameters['paddingY'],self.tilingParameters['paddingY']),1)
                im = openfile(self.cityfolderPath / f'{self.mapName}{self.mapfileFormat}', self.mapfileFormat)
                self.fullMap = self.paddingMap(ToTensor()(im))
    
    def __len__(self):
        return len(self.tilesCoordinates)        

    def __getitem__(self, index):
        coordDict = self.tilesCoordinates[f'{index}']
        sample = {'coordDict': coordDict}
        if self.fromCoordinates:
            sample['tile'] = self.fullMap[:,coordDict['yLow']:coordDict['yHigh'], coordDict['xLow']:coordDict['xHigh']]
        else:
            sample['tile'] = ToTensor()(openfile(self.cityfolderPath / f'{self.mapName}_{index}{self.thumbnailFileFormat}', self.thumbnailFileFormat))
        
        if self.transform:
            sample['tile'] = self.transform(sample)

        if self.colored:
            sample['labels'] = one_hot(torch.tensor(self.classifiedPath[f'{index}']),5)
        return sample

class unfold(object):
    def __init__(self):
        self.height = 7590
        self.width = 11400
        self.kernelSize = 512
        self.stride = (354,363)
        self.unfold = nn.Unfold(kernel_size=self.kernelSize, stride = self.stride)
        self.hRatio = int((self.height-self.kernelSize+2)/self.stride[0])
        self.wRatio = int((self.width-self.kernelSize+2)/self.stride[1])
        
    def __call__(self, sample):
        a = self.unfold(sample['map']).reshape(self.kernelSize,self.kernelSize,self.hRatio*self.wRatio)
        return {'tiledMap': a.permute(2,0,1),
                'map': sample['map'],
                'tilePath':sample['tilePath'],
                'properties':sample['properties'],
                'projection':sample['projection'],
                'metaData':sample['metaData'],
                'boundaries':sample['boundaries'],
                'mapName':sample['mapName']
                }

class pad(object):
    def __init__(self, paddingX=45, paddingY=188):
        self.paddingMap = nn.ConstantPad2d((paddingX,paddingX, paddingY,paddingY),1)
        
    def __call__(self, sample):
        return {'map': self.paddingMap(sample['map']),
                'tilePath':sample['tilePath'],
                'properties':sample['properties'],
                'projection':sample['projection'],
                'metaData':sample['metaData'],
                'boundaries':sample['boundaries'],
                'mapName':sample['mapName']
                }

def openfile(filePath, fileExtension):
    if fileExtension =='.npy':
        return np.load(filePath)
    elif fileExtension =='.jpg':
        return normalise(np.array(Image.open(filePath).convert('L')))
    elif fileExtension =='.json':
        return json.load(open(filePath))
    else:
        raise ValueError ('Wrong fileExtension string')

def matchKeyToName(pathToJsonfile:str, key : str):
    cityKeysFile = openfile(pathToJsonfile,'.json')
    return cityKeysFile[key]['Town']

def getTiffProperties(tiffImage, showDict = False, returnDict=False):    
    meta_dict = {TAGS[key] : tiffImage.tag[key] for key in tiffImage.tag.keys()}
    if showDict:
        for key, value in meta_dict.items():
            print(' %s : %s' % (key, value))
    if returnDict:
        return meta_dict

def normalise(x:np.float32):
    return (x-x.min()) / (x.max()-x.min())

def extractMetaData(tfwRaw) ->dict:
    xDiff = float(tfwRaw.split("\n")[0])
    yDiff = float(tfwRaw.split("\n")[3])
    westBoundary = float(tfwRaw.split("\n")[4])
    northBoundary = float(tfwRaw.split("\n")[5])
    return {'xDiff':xDiff, 'yDiff':yDiff, 'westBoundary':westBoundary, 'northBoundary':northBoundary}

def getBoundaries(metaData:dict, imageHeight:int, imageWidth:int) -> dict:
    eastBoundary = metaData['westBoundary'] + (imageWidth - 1) * metaData['xDiff']
    southBoundary = metaData['northBoundary'] + (imageHeight - 1) * metaData['yDiff']
    return {'westBoundary':metaData['westBoundary'], 'northBoundary':metaData['northBoundary'],
            'eastBoundary':eastBoundary, 'southBoundary':southBoundary, 
            'xDiff':metaData['xDiff'], 'yDiff':metaData['yDiff'] }