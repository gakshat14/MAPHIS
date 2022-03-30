import json
import zipfile
import glob
import os
from pathlib import Path, PurePath

def matchKeyToName(pathToJsonfile:str, key : str):
    cityKeysFile = json.load(open(pathToJsonfile))
    return cityKeysFile[key]['Town']

def createFolders(rawFolderPath='rawFolder', classifiedFolderPath='classifiedFolder', fileExtension='.zip'):

    listOfZips = glob.glob(rawFolderPath+'/*'+fileExtension)

    for zipName in listOfZips:
        ## Get dir name for folder creation in classified
        dirName = os.path.splitext(zipName)[0]
        Path(PurePath(classifiedFolderPath).joinpath(PurePath(dirName).name)).mkdir(parents=True, exist_ok=True)
        ## Extract compressed in raw

        with zipfile.ZipFile(zipName, 'r') as zip_ref:
            zip_ref.extractall(dirName)
        
