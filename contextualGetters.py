import argparse
import pandas as pd
from pyprojroot import here
import numpy as np
from pathlib import Path
from datasets.datasetsFunctions import matchKeyToName, openfile
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--datasetPath', required=False, type=str, default='datasets')
parser.add_argument('--cityKey', required=False, type=str, default = '36')
parser.add_argument('--classifType', required=False, type=str, default = 'Labels')
args = parser.parse_args()

cityName = matchKeyToName(f'{args.datasetPath}/cityKey.json', args.cityKey)

def getClassDistributionFromDataframe(dataframe:pd.DataFrame, distribution:np.float32)->np.float32:
    distribution = distribution.copy()
    for value in dataframe.values():
        distribution[value] +=1
    return distribution

def getLabels(dataframe:pd.DataFrame, x:int, y:int, distance:int) -> list:
    allLabels = dataframe.query('xTile-@distance<@x<xTile+@distance and yTile-@distance<@y<yTile+@distance')
    meaningfullLabels = allLabels[allLabels['class'] != 0]['class']
    getClassDistributionFromDataframe
    return meaningfullLabels

classes = openfile(Path(f'{args.datasetPath}/classified{args.classifType}/classes.json'), '.json')
dataframe = pd.read_json(here().joinpath('datasets/classifiedLabels/Luton/0105033010241.json'))

fig, axs = plt.subplots(1, 1, tight_layout=True)

# We can set the number of bins with the *bins* keyword argument.
axs.hist(getLabels(dataframe.transpose(), 7290, 4696, 4000), bins=len(classes))

plt.show()