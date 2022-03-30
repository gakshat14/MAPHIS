import numpy as np
from pathlib import Path, PurePath
from csv import writer

from numpy.lib.shape_base import tile

cityNames = ['Barrow_in_Furness', 'Luton']
tileNames = ['12345', '56789']

saveFolder = 'rawFolder'

for cityName in cityNames:
    savePath = Path(PurePath(saveFolder).joinpath(cityName))
    savePath.mkdir(parents=True, exist_ok=True)
    savePathString = saveFolder+'/'+cityName+'/'
    with open(f"{savePath.joinpath('rawFeatures.csv')}",'w', newline="") as csvFile:
        csvWriter = writer(csvFile)
        for tileName in tileNames:
            tileNameSave = savePathString + tileName+ '_mat'
            for i in range(5):
                mat = np.random.rand(10,10)
                np.savez(tileNameSave + str(i), mat)
                csvWriter.writerow(np.random.rand(10))