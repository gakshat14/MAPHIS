import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2

'''
dataframe = pd.read_json('C:/Users/hx21262/MAPHIS/datasets/extractedShapes/Luton/0105033010241/shapeDict.json')
dataframe = dataframe.drop(['savePath','xTile','yTile']).transpose().astype(float)
dataframe.hist( color='steelblue', edgecolor='black', linewidth=1.0, xlabelsize=8, ylabelsize=8, grid=False)
plt.show()
'''

featureName = 'trees'

def getClassDistributionFromDataframe(dataframe:pd.DataFrame, distribution:np.float32)->np.float32:
    distribution = distribution.copy()
    for value in dataframe.values():
        distribution[value] +=1
    return distribution

def getLabels(dataframe:pd.DataFrame, x:int, y:int, distance:int) -> pd.DataFrame:
    allLabels = dataframe.query('xTile-@distance<@x<xTile+@distance and yTile-@distance<@y<yTile+@distance')
    return allLabels[allLabels['class'] != 0]['class']

classes = json.load(open(Path(f'datasets/classifiedLayers/classes.json')))
labelsDataframe = pd.read_json('datasets/classifiedLayers/Luton/0105033010241.json').transpose()
print(labelsDataframe)

allShapesDict = json.load(open(f'datasets/layers/{featureName}/Luton/0105033010241.json'))[f'{featureName}']
tilingParameters = json.load(open(f'datasets/layers/tilingParameters.json'))

colorisedMap = np.zeros((tilingParameters['height'], tilingParameters['width'],3), np.uint8)

for shapeDict in allShapesDict.values():
    if shapeDict['circleness'] < 0.2 or shapeDict['rectangleness']<0.2:
        pass
    else:   
        if getLabels(labelsDataframe, shapeDict['xTile'], shapeDict['yTile'], 256).empty:
            pass
        else:
            cv2.drawContours(colorisedMap, [np.load(shapeDict['savePath'])], 0, (255,255,255), -1) 

plt.matshow(colorisedMap)
plt.show()