#!/usr/bin/env python
# coding: utf-8
import csv
import pathlib
from torch.utils.data import Dataset
from PIL import Image
import torch.nn as nn
import torch
from torchvision.transforms import ToTensor
import numpy as np
import json
from PIL.TiffTags import TAGS
from torch.nn.functional import one_hot
import morphological_tools as morph_tools
import constants

def normalise(tensor):
    return (tensor-tensor.min())/(tensor.max()-tensor.min()) 

def identity(tensor):
    return tensor

class Maps(Dataset):
    def __init__(self, datasetPath:pathlib.Path, cityName:str, fileFormat='.jpg', transform=None):
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
            
class Thumbnails(Dataset):
    def __init__(self, datasetPath:pathlib.Path, cityName:str,  tileName='0105033050201', transform=None, tileFileFormat='.jpg', featureName='trees') -> None:
        self.tilingParameters = json.load(open(datasetPath / 'tilingParameters.json'))
        self.tilesCoordinates = self.tilingParameters['coordinates']
        self.cityName = cityName
        self.mapName  = tileName
        self.featureName = featureName
        self.tileFileFormat = tileFileFormat
        self.cityfolderPath = next(datasetPath.glob(f'cities/{cityName}/*/*') )

        self.paddingMapBackground = nn.ConstantPad2d((self.tilingParameters['paddingX'],self.tilingParameters['paddingX'], self.tilingParameters['paddingY'],self.tilingParameters['paddingY']),1)
        self.paddingMapMask = nn.ConstantPad2d((self.tilingParameters['paddingX'],self.tilingParameters['paddingX'], self.tilingParameters['paddingY'],self.tilingParameters['paddingY']),0)
        background = openfile(self.cityfolderPath / f'{self.mapName}.jpg')
        self.paddedBackground = np.where(self.paddingMapBackground(ToTensor()(background))!=0,1,0)

        trees_mask = openfile(datasetPath / f'layers/trees/{cityName}/{self.mapName}_mask.npy')
        self.padded_trees_mask = torch.where(self.paddingMapMask(ToTensor()(trees_mask))!=0,1,0)

        buildings_mask = openfile(datasetPath / f'layers/buildings/{cityName}/{self.mapName}_mask.npy')
        self.padded_buildings_mask = torch.where(self.paddingMapMask(ToTensor()(buildings_mask))!=0,1,0)

        labels_mask = openfile(datasetPath / f'layers/labels/{cityName}/{self.mapName}_mask.npy')
        self.padded_labels_mask = torch.where(self.paddingMapMask(ToTensor()(labels_mask))!=0,1,0)

    def __len__(self):
        return len(self.tilesCoordinates)        

    def __getitem__(self, index):
        coordDict = self.tilesCoordinates[f'{index}']
        sample = {'coordDict': coordDict}
        sample['background'] = self.paddedBackground[:,coordDict['yLow']:coordDict['yHigh'], coordDict['xLow']:coordDict['xHigh']]
        if self.featureName =='labels':
            sample['mask'] = self.padded_labels_mask[:,coordDict['yLow']:coordDict['yHigh'], coordDict['xLow']:coordDict['xHigh']] 
            sample['overlap_mask'] = torch.clamp(self.padded_trees_mask[:,coordDict['yLow']:coordDict['yHigh'], coordDict['xLow']:coordDict['xHigh']]+self.padded_buildings_mask[:,coordDict['yLow']:coordDict['yHigh'], coordDict['xLow']:coordDict['xHigh']], 0 , 1)
        elif self.featureName =='trees':
            sample['mask'] = self.padded_trees_mask[:,coordDict['yLow']:coordDict['yHigh'], coordDict['xLow']:coordDict['xHigh']] 
            sample['overlap_mask'] = torch.clamp(self.padded_labels_mask[:,coordDict['yLow']:coordDict['yHigh'], coordDict['xLow']:coordDict['xHigh']]+self.padded_buildings_mask[:,coordDict['yLow']:coordDict['yHigh'], coordDict['xLow']:coordDict['xHigh']], 0 , 1)
        elif self.featureName =='buildings':
            sample['mask'] = self.padded_buildings_mask[:,coordDict['yLow']:coordDict['yHigh'], coordDict['xLow']:coordDict['xHigh']] 
            sample['overlap_mask'] = torch.clamp(self.padded_trees_mask[:,coordDict['yLow']:coordDict['yHigh'], coordDict['xLow']:coordDict['xHigh']]+self.padded_labels_mask[:,coordDict['yLow']:coordDict['yHigh'], coordDict['xLow']:coordDict['xHigh']], 0 , 1)
        else:
            raise ValueError ('Wrong featureName')
        return sample


class Tiles(Dataset):
    def __init__(self, datasetPath:pathlib.Path, cityName:str, mapName='0105033050201', transform=None, fromCoordinates=False, mapfileFormat='.jpg', thumbnailFileFormat='.npy', colored=False, feature=None) -> None:
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
                im = openfile(self.cityfolderPath / f'{self.mapName}{self.mapfileFormat}')
                dilated = morph_tools.dilation(im)
                eroded = morph_tools.erosion(im)
                self.fullMap = ToTensor()(np.concatenate((im, dilated, eroded),1))
        else:
            self.cityfolderPath = next(datasetPath.glob(f'cities/{cityName}/*/*') )
            if fromCoordinates:
                self.paddingMap = nn.ConstantPad2d((self.tilingParameters['paddingX'],self.tilingParameters['paddingX'], self.tilingParameters['paddingY'],self.tilingParameters['paddingY']),1)
                im = openfile(self.cityfolderPath / f'{self.mapName}{self.mapfileFormat}')
                self.fullMap = self.paddingMap(ToTensor()(im))
    
    def __len__(self):
        return len(self.tilesCoordinates)        

    def __getitem__(self, index):
        coordDict = self.tilesCoordinates[f'{index}']
        sample = {'coordDict': coordDict}
        if self.fromCoordinates:
            sample['tile'] = self.fullMap[:,coordDict['yLow']:coordDict['yHigh'], coordDict['xLow']:coordDict['xHigh']]
        else:
            sample['tile'] = ToTensor()(openfile(self.cityfolderPath / f'{self.mapName}_{index}{self.thumbnailFileFormat}'))
        
        if self.transform:
            sample['tile'] = self.transform(sample)

        if self.colored:
            sample['labels'] = one_hot(torch.tensor(self.classifiedPath[f'{index}']),5)
        return sample

class unfold(object):
    def __init__(self):
        self.height = constants.TILEHEIGHT
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

def openfile(filePath:pathlib.Path):
    fileExtension = filePath.suffix
    if fileExtension =='.npy':
        return np.load(filePath)
    elif fileExtension =='.jpg':
        return normalise(np.array(Image.open(filePath).convert('L')))
    elif fileExtension =='.json':
        return json.load(open(filePath))
    else:
        raise ValueError ('Wrong fileExtension string')

def matchKeyToName(pathToJsonfile:str, key : str):
    cityKeysFile = openfile(pathToJsonfile)
    return cityKeysFile[key]['Town']

def getTiffProperties(tiffImage, showDict = False, returnDict=False):    
    meta_dict = {TAGS[key] : tiffImage.tag[key] for key in tiffImage.tag.keys()}
    if showDict:
        for key, value in meta_dict.items():
            print(' %s : %s' % (key, value))
    if returnDict:
        return meta_dict

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