import pathlib
import numpy as np
import cv2
from imutils import grab_contours
import constants
import argparse
from datasets.datasetsFunctions import matchKeyToName

AREASHAPES = constants.AREASHAPES
COLORDICT = constants.COLORDICT
FEATURENAMES = constants.FEATURENAMES

def main(args):
    cityName = matchKeyToName(f'{args.datasetPath}/{args.cityKey}.json')
    allTilesPaths = list(pathlib.Path(f'{args.datasetPath}/cities/{cityName}').glob(f'*/*/*.jpg'))

    for tilePath in allTilesPaths:
        tileName = tilePath.stem
        print(f'Processing Tile {tileName}')

        colorisedMap = np.ones((constants.TILEHEIGHT,constants.TILEWIDTH,3), np.uint8)
        background = cv2.imread(f'{tilePath}')

        for featureName in FEATURENAMES:
            filePath = pathlib.Path(f'datasets/layers/{featureName}/{cityName}')       

            # Load segmented image and remove padding
            segmentedMask = np.load(filePath.joinpath(f'{tileName}_segmented.npy'))[157:7590+157, 100:11400+100]

            contours = cv2.findContours(segmentedMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = grab_contours(contours)

            nContours = 0

            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area <= AREASHAPES[featureName]:
                    pass
                else:        
                    cv2.drawContours(colorisedMap, [contour], 0, COLORDICT[featureName], -1) 
                    nContours += 1
    pathlib.Path(f'{args.savePath}/{cityName}').mkdir(exist_ok=True, parents=True)
    print(f'Saving colored tile {tileName} at {args.savePath}/{cityName}/{tileName}_colored.jpg')
    overlayed = cv2.addWeighted(background, 0.5, colorisedMap, 0.5)
    cv2.imwrite(f'{args.savePath}/{cityName}/{tileName}_colored.jpg', overlayed)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetPath', required=False, type=str, default='datasets')
    parser.add_argument('--savePath', required=False, type=str, default='datasets/visuals')
    parser.add_argument('--cityKey', required=False, type=str, default = '36')
    args = parser.parse_args()
    main(args)