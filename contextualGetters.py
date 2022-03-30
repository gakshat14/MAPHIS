import argparse
import pandas as pd
from pyprojroot import here
import pathlib
from pathlib import Path
from datasets.datasetsFunctions import matchKeyToName, openfile
import matplotlib.pyplot as plt

def getLabels(dataframe:pd.DataFrame, x:int, y:int, distance:int) -> list:
    allLabels = dataframe.query('xTile-@distance<@x<xTile+@distance and yTile-@distance<@y<yTile+@distance')
    meaningfullLabels = allLabels[allLabels['class'] != 0]['class']
    return meaningfullLabels

def main(args):
    cityName = matchKeyToName(pathlib.Path(f'{args.datasetPath}/cityKey.json'), args.cityKey)
    classes = openfile(Path(f'{args.datasetPath}/classifiedLayers/classes.json'))
    dataframe = pd.read_json(here().joinpath(f'datasets/classifiedLayers/{cityName}/{args.tileName}.json'))
    x, y = 7290,4696
    distance = 500
    neighbouringLabels = getLabels(dataframe.transpose(), x,y, distance)
    plt.hist(neighbouringLabels, bins=len(classes))
    plt.xticks([i for i in range(len(classes))], classes, rotation=45)
    #plt.imshow(openfile(pathlib.Path(f'{args.datasetPath}/cities/{cityName}/500/tp_1/{args.tileName}.jpg'))[y-constants.HEIGHTPADDING-distance:y-constants.HEIGHTPADDING+distance, x-constants.WIDTHPADDING-distance:x-constants.WIDTHPADDING+distance])
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetPath', required=False, type=str, default='datasets')
    parser.add_argument('--cityKey', required=False, type=str, default = '36')
    parser.add_argument('--tileName', required=False, type=str, default= '0105033010241')
    args = parser.parse_args()
    main(args)