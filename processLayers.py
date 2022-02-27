import cv2
import pathlib
from imutils import grab_contours
import numpy as np
import json
from tqdm import tqdm

cityName = 'Luton'

featuresDict = {'labels':{'thresh':106}, 'buildings':{'thresh':76}, 'trees':{'thresh':166}}

pathToShapeFolder = pathlib.Path(f'datasets/extractedShapes/{cityName}')

pathsToLayers = pathlib.Path(f'datasets/layers/raw/{cityName}').iterdir()

tilingParameters = {"height": 7590, "width": 11400, "kernelSize": 512, "paddingX": 0, "paddingY": 0, "strideX": 0, "strideY": 0, "nCols": 204, "nRows": 16}

with open(f'datasets/layers/tilingParameters.json', 'w') as outfile:
    json.dump(tilingParameters, outfile)

for filePath in pathsToLayers: 

    pathToShape = pathToShapeFolder / filePath.stem

    for featureName, featureDict in featuresDict.items():
        shapeSavePath = pathToShape / f'{featureName}'
        shapeSavePath.mkdir(parents=True, exist_ok=True)

        jsonSavePath = pathlib.Path(f'datasets/layers/{featureName}/{cityName}')
        jsonSavePath.mkdir(parents=True, exist_ok=True)

        featureLayer = cv2.imread(str(filePath.joinpath(f'{featureName}.jpg')), cv2.IMREAD_GRAYSCALE)

        featureLayer = np.where(featureLayer<=featureDict['thresh']+10,255,0)

        featureLayer = np.uint8(featureLayer)

        maskShapePath = pathlib.Path(f'datasets/layers/{featureName}/{cityName}')
        if maskShapePath.joinpath(f'{filePath.stem}_mask.npy').is_file():
            print(f'Mask already created for {featureName}')
        else:
            print(f'Saving mask for {featureName}')
            np.save(maskShapePath.joinpath(f'{filePath.stem}_mask.npy'), featureLayer)

        contours = cv2.findContours(featureLayer, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = grab_contours(contours)

        shapeDict = {"mapName":f'{filePath.stem}', f'{featureName}':{}}

        for i, contour in tqdm(enumerate(contours)):
            area = cv2.contourArea(contour)
            if area ==0:
                pass
            else:
                shapeDict[f'{featureName}'][f'{i}'] ={}

                shapeDict[f'{featureName}'][f'{i}']['area'] = area
                M = cv2.moments(contour)
                '''shapeDict[f'{featureName}'][f'{i}']['xTile'] = int(M["m10"] / M["m00"])
                shapeDict[f'{featureName}'][f'{i}']['yTile'] = int(M["m01"] / M["m00"])'''
                perimeter = cv2.arcLength(contour,True)
                shapeDict[f'{featureName}'][f'{i}']['perimeter'] = perimeter
                _, radiusEnclosingCircle = cv2.minEnclosingCircle(contour)
                areaCircle = 3.1416 * radiusEnclosingCircle * radiusEnclosingCircle
                circleness = area/areaCircle
                shapeDict[f'{featureName}'][f'{i}']['circleness'] = circleness
                (x, y, W, H) = cv2.boundingRect(contour)
                shapeDict[f'{featureName}'][f'{i}']['rectangleness'] = area / (W*H)
                shapeDict[f'{featureName}'][f'{i}']['H'] = H 
                shapeDict[f'{featureName}'][f'{i}']['W'] = W
                shapeDict[f'{featureName}'][f'{i}']['xTile'] = x
                shapeDict[f'{featureName}'][f'{i}']['yTile'] = y
                shapeDict[f'{featureName}'][f'{i}']['savePath'] = str(pathToShape / f'{featureName}/{i}.npy')
                np.save(pathToShape / f'{featureName}/{i}.npy', contour)

        with open(jsonSavePath /f'{filePath.stem}.json', 'w') as outfile:
            json.dump(shapeDict, outfile)

       


